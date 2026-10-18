"""The reference projection has to match the shader: adjoint divergence, and a V-cycle that
actually removes it. These are small grids; the GPU check is validate_phase4."""

import numpy as np

import pressure3d as pressure


def _smooth_velocity(n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    vel = np.zeros((n, n, n, 3))
    vel[..., 0] = 0.15 * np.sin(2 * np.pi * xx) * np.cos(np.pi * yy)
    vel[..., 1] = 0.10 * np.cos(np.pi * xx) * np.sin(2 * np.pi * yy)
    vel[..., 2] = 0.04 * np.sin(np.pi * zz)
    vel += 0.005 * rng.normal(size=vel.shape)
    return vel


def _solid(n):
    solid = np.zeros((n, n, n), dtype=bool)
    solid[n // 3:n // 3 + max(3, n // 7), n // 3:n // 2, n // 4:n // 2] = True
    return solid


def test_divergence_is_the_laplacian_of_the_gradient():
    rng = np.random.default_rng(1)
    pressure_field = rng.normal(size=(12, 12, 12))
    solid = _solid(12)
    pressure_field[solid] = 0.0
    got = pressure.divergence(pressure.gradient(pressure_field, solid=solid), solid=solid)
    want = pressure.laplacian(pressure_field, solid=solid)
    assert np.max(np.abs(got - want)) == 0.0


def test_vcycle_reduces_smooth_divergence_tenfold():
    vel = _smooth_velocity(32)
    solid = _solid(32)
    obstacle = np.zeros_like(vel)
    obstacle[..., 1] = 0.2
    rhs = pressure.remove_mean(pressure.divergence(vel, solid=solid, obstacle_vel=obstacle), solid)
    before = pressure.rms(pressure.divergence(vel, solid=solid, obstacle_vel=obstacle), solid)
    jacobi = pressure.jacobi(np.zeros_like(rhs), rhs, solid=solid, iterations=12, omega=1.0)
    after_jacobi = pressure.rms(
        pressure.divergence(pressure.project(vel, jacobi, solid=solid, obstacle_vel=obstacle),
                            solid=solid, obstacle_vel=obstacle),
        solid,
    )
    solved = pressure.solve(rhs, solid=solid, cycles=2, smooth=4, coarse_iters=32)
    after = pressure.rms(
        pressure.divergence(pressure.project(vel, solved, solid=solid, obstacle_vel=obstacle),
                            solid=solid, obstacle_vel=obstacle),
        solid,
    )
    assert before / after >= 10.0
    assert before / after > before / after_jacobi


def test_liquid_free_surface_stencil_matches_jacobi():
    """Air is Dirichlet (p = 0). div(grad p) has to be the stencil Jacobi inverts, on liquid cells."""
    rng = np.random.default_rng(4)
    n = 12
    pressure_field = rng.normal(size=(n, n, n))
    air = np.zeros((n, n, n), dtype=bool)
    air[:, n // 2:, :] = True
    solid = np.zeros((n, n, n), dtype=bool)
    solid[0, :, :] = True
    pressure_field[air | solid] = 0.0
    got = pressure.laplacian(pressure_field, solid=solid, air=air)
    # One Jacobi residual step at omega 1 from 0 is not the operator; build it from the neighbour sum.
    acc = np.zeros_like(pressure_field)
    for axis in range(3):
        for delta in (1, -1):
            nb = pressure._sample(pressure_field, axis, delta, fill=0.0)
            nb_liquid = pressure._neighbour_fluid(~air & ~solid, axis, delta)
            nb_air = pressure._neighbour_fluid(air, axis, delta)
            acc += np.where(nb_liquid, nb, np.where(nb_air, 0.0, pressure_field))
    stencil = (acc - 6.0 * pressure_field)
    liquid = ~air & ~solid
    assert np.max(np.abs(got[liquid] - stencil[liquid])) < 1e-9


def test_mgpcg_hydrostatic_column_is_at_rest():
    """A tank with a floor and an open top: CG+V-cycle must recover p = g dt * depth and kill the velocity."""
    n = 32
    h = 1.0 / n
    liquid = np.zeros((n, n, n), dtype=bool)
    liquid[:, : n // 2, :] = True
    air = ~liquid
    gdt = 9.81 / 60.0
    vel = np.zeros((n, n, n, 3))
    vel[..., 1] = np.where(liquid, -gdt, 0.0)
    rhs = pressure.divergence(vel, h, air=air)
    solved, history = pressure.mgpcg(rhs, h, air=air, iterations=6, smooth=1, coarse_iters=16)
    out = pressure.project(vel, solved, h, air=air)
    div = pressure.rms(pressure.divergence(out, h, air=air), air=air)
    assert history[0] == 1.0
    assert history[-1] < 0.02
    assert history[-1] < history[1]
    assert abs(float(out[..., 1][liquid].mean())) < 1e-3
    assert div <= 0.03
    # Bottom pressure is g dt times the liquid depth.
    assert abs(float(solved[n // 2, 0, n // 2]) - gdt * (n // 2) * h) < 0.01


def test_liquid_vcycle_cuts_a_free_surface_divergence():
    # Dyadic grid: the free-surface restriction is un-normalised, and a 24^3 grid does not halve
    # onto an even coarse mask. Two cycles on 16^3 still take the divergence down by about 8x.
    vel = _smooth_velocity(16)
    air = np.zeros(vel.shape[:3], dtype=bool)
    air[:, 10:, :] = True
    vel[air] = 0.0
    report = pressure.reduction(vel, air=air, cycles=2, smooth=4, coarse_iters=30)
    assert report["before"] > 0.0
    assert report["factor"] >= 8.0
