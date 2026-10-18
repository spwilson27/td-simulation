"""Reference FLIP transfer, binning, and the liquid checks that do not need TouchDesigner."""

import numpy as np

import flip


def _interior_block():
    # Domain [0, 1]^3, h = 1/16. Particles sit two cells in, so every kernel is complete.
    origin = np.zeros(3)
    size = np.ones(3)
    h = 1.0 / 16.0
    spacing = h / 2.0
    state = flip.seed_box(origin, size, (0.2, 0.2, 0.2), (0.8, 0.8, 0.8), spacing, velocity=(1.25, -0.4, 0.15))
    return state, origin, size, h, (16, 16, 16)


def test_quadratic_weights_partition_unity_inside():
    h = 0.1
    origin = np.zeros(3)
    # A particle at an arbitrary interior point. Sum of weights on the 3^3 neighbourhood is 1.
    rng = np.random.default_rng(0)
    for _ in range(20):
        pos = rng.uniform(0.3, 0.7, size=3)
        cell = np.floor(pos / h).astype(int)
        total = 0.0
        for dz in range(-2, 3):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    c = cell + np.array([dx, dy, dz])
                    centre = origin + (c + 0.5) * h
                    total += float(flip.weight((pos - centre) / h))
        assert abs(total - 1.0) < 1e-9


def test_bitonic_schedule_sorts():
    rng = np.random.default_rng(1)
    keys = rng.integers(0, 50, size=(32, 2)).astype(np.int32)
    order = flip.bitonic_sort(keys)
    got = keys[order]
    expect = keys[np.lexsort((keys[:, 1], keys[:, 0]))]
    assert np.array_equal(got, expect)
    assert len(flip.bitonic_passes(32)) == 15  # 5*6/2


def test_bins_are_grouped_in_particle_order():
    linear = np.array([2, 0, 2, 1, 0, 2], dtype=np.int32)
    active = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
    bins = flip.bin_particles(linear, active, n_cells=4)
    # Cell 2 holds particles 0, 2, 5 (index 3 is inactive, index order).
    start = int(bins["start"][2])
    count = int(bins["count"][2])
    got = bins["order"][start:start + count]
    assert list(got) == [0, 2, 5]
    assert int(bins["count"][1]) == 0
    assert int(bins["count"][0]) == 2


def test_p2g_conserves_mass_and_momentum_and_round_trips_uniform_velocity():
    state, origin, _size, h, res = _interior_block()
    transferred = flip.p2g(state["pos"], state["vel"], state["mass"], origin, h, res)
    assert abs(transferred["mass"].sum() - state["mass"].sum()) / state["mass"].sum() < 1e-8
    particle_mom = (state["mass"][:, None] * state["vel"]).sum(axis=0)
    grid_mom = transferred["momentum"].reshape(-1, 3).sum(axis=0)
    assert np.max(np.abs(grid_mom - particle_mom)) / np.linalg.norm(particle_mom) < 1e-8
    # PIC of a uniform field, no projection: every particle comes back with the same velocity.
    back, _c = flip.g2p(state["pos"], state["vel"], state["mass"], transferred["vel"], transferred["vel"],
                        origin, h, flip_blend=0.0)
    assert np.max(np.abs(back - state["vel"])) < 1e-8
    flip_back, _c = flip.g2p(state["pos"], state["vel"], state["mass"], transferred["vel"], transferred["vel"],
                             origin, h, flip_blend=0.95)
    assert np.max(np.abs(flip_back - state["vel"])) < 1e-8


def test_apic_recovers_a_linear_velocity_gradient():
    state, origin, _size, h, res = _interior_block()
    # v = (x, 0, 0). The quadratic APIC matrix is the velocity gradient, about diag(1, 0, 0).
    state["vel"] = np.zeros_like(state["pos"])
    state["vel"][:, 0] = state["pos"][:, 0]
    state["C"] = np.zeros((state["pos"].shape[0], 3, 3))
    transferred = flip.p2g(state["pos"], state["vel"], state["mass"], origin, h, res, apic=state["C"])
    _back, matrices = flip.g2p(state["pos"], state["vel"], state["mass"], transferred["vel"], transferred["vel"],
                               origin, h, apic=True)
    # Interior particles, away from the block edge where the kernel is one-sided on the *block*
    # but still complete on the grid. The gradient of v=x is 1.
    # Mass-weighted P2G of v = x_particle is not exactly the cell-centre coordinate, so the
    # recovered gradient sits near 0.9 rather than 1. The off-diagonal blocks stay at 0.
    mean_c = matrices.mean(axis=0)
    assert abs(mean_c[0, 0] - 1.0) < 0.15
    assert np.max(np.abs(mean_c[1:])) < 0.05


def test_hydrostatic_tank_stays_still_and_keeps_its_volume():
    origin = np.zeros(3)
    size = np.ones(3)
    res = (16, 16, 16)
    h = 1.0 / 16.0
    spacing = h / 2.0
    state = flip.seed_box(origin, size, (0.0, 0.0, 0.0), (1.0, 0.45, 1.0), spacing)
    volume0 = flip.liquid_volume(state["mass"], state["density"])
    dt = 1.0 / 60.0
    peak = 0.0
    for _ in range(12):
        state = flip.step_particles(state, origin, size, h, res, dt, flip=0.85, cycles=3, smooth=5, coarse=20)
        speed = np.linalg.norm(state["vel"], axis=1)
        peak = max(peak, float(speed.max()))
    heights = flip.column_heights(state["pos"], state["mass"], origin, size, bins=4)
    volume1 = flip.liquid_volume(state["mass"], state["density"])
    # Gravity wants about g*dt = 0.16 m/s per step. The projection has to put that back.
    # A few surface particles keep a fraction of that; the bulk and the height stay put.
    assert peak < 0.35
    # g*dt is 0.16. The fastest particle stays within about two impulses; the mean is the bulk.
    assert float(np.linalg.norm(state["vel"], axis=1).max()) < 0.30
    assert float(np.linalg.norm(state["vel"], axis=1).mean()) < 0.05
    assert abs(volume1 - volume0) / volume0 < 1e-12
    assert float(np.std(heights)) < 0.08


def test_dam_break_front_advances_inside_the_ritter_band():
    """Martin & Moyce / Ritter: a dry-bed front moves at order sqrt(g H), and only forward."""
    origin = np.zeros(3)
    size = np.ones(3)
    res = (16, 16, 16)
    h = 1.0 / 16.0
    spacing = h / 2.0
    height = 0.55
    width = 0.30
    state = flip.seed_box(origin, size, (0.0, 0.0, 0.0), (width, height, 1.0), spacing)
    dt = 1.0 / 60.0
    g = 9.81
    fronts = [flip.front_x(state["pos"], state["mass"], y_max=0.25)]
    for _ in range(16):
        state = flip.step_particles(state, origin, size, h, res, dt, flip=0.9, cycles=3, smooth=4, coarse=16)
        fronts.append(flip.front_x(state["pos"], state["mass"], y_max=height))
    early = np.asarray(fronts)
    # The toe moves forward. One cell of jitter is the lattice, not a reversal.
    assert np.all(np.diff(early) >= -h)
    assert fronts[-1] > fronts[0] + 0.25
    assert fronts[-1] <= 1.0 + 1e-6
    # Versus 2 sqrt(g H), the Ritter dry-bed speed for an infinite reservoir. A finite column on a
    # 16^3 grid is slower and is not the Martin & Moyce curve; it has to sit in that band.
    elapsed = 16 * dt
    moved = fronts[-1] - fronts[0]
    ritter = 2.0 * np.sqrt(g * height) * elapsed
    assert 0.15 * ritter < moved < 1.25 * ritter
