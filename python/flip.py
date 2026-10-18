"""Reference FLIP / APIC transfer and a liquid step on the phase-4 collocated grid.

Particle-to-grid is a gather. Particles are binned by cell, and each cell sums its contributors in
particle-index order, so the sum does not depend on thread scheduling. Quadratic B-spline weights
are a partition of unity on an interior particle (support 1.5 cells), which is why mass and momentum
survive the transfer. The grid step reuses `pressure3d` with an air mask: liquid cells are the
unknown, air is Dirichlet (p = 0), solids are Neumann.

Float64 here is the reference. The GPU gather is the same sum in float32, so a byte-exact match
against this module is not the goal; conservation tolerances below are for this reference.
"""

from __future__ import annotations

import numpy as np

import pressure3d as pressure


def quadratic(d: np.ndarray) -> np.ndarray:
    """1D quadratic B-spline. `d` is the offset from a cell centre in units of h."""
    a = np.abs(d)
    w = np.zeros_like(a, dtype=np.float64)
    close = a < 0.5
    far = (a >= 0.5) & (a < 1.5)
    w[close] = 0.75 - d[close] ** 2
    w[far] = 0.5 * (1.5 - a[far]) ** 2
    return w


def weight(offset_cells: np.ndarray) -> np.ndarray:
    """Separable quadratic weight. `offset_cells` is (particle - cell centre) / h."""
    return quadratic(offset_cells[..., 0]) * quadratic(offset_cells[..., 1]) * quadratic(offset_cells[..., 2])


def bitonic_passes(n: int) -> list[tuple[int, int]]:
    """(k, j) pairs for a power-of-two bitonic sort. The GPU sort uses this same schedule."""
    if n < 2 or n & (n - 1):
        raise ValueError("bitonic sort needs a power of two")
    logn = n.bit_length() - 1
    passes = []
    for exp in range(1, logn + 1):
        k = 1 << exp
        j = k >> 1
        while j > 0:
            passes.append((k, j))
            j >>= 1
    return passes


def bitonic_sort(keys: np.ndarray) -> np.ndarray:
    """Sort rows lexicographically. `keys` is (N, K), N a power of two. Returns the permutation."""
    n = int(keys.shape[0])
    order = np.arange(n, dtype=np.int32)
    for k, j in bitonic_passes(n):
        partner = np.arange(n) ^ j
        low = np.arange(n) < partner
        # Only the lower index decides, then both slots are written.
        take_lo = np.flatnonzero(low)
        other = partner[take_lo]
        a = keys[order[take_lo]]
        b = keys[order[other]]
        ascending = (take_lo & k) == 0
        # a < b lexicographically
        less = np.zeros(take_lo.shape[0], dtype=bool)
        undecided = np.ones(take_lo.shape[0], dtype=bool)
        for col in range(keys.shape[1]):
            cmp_less = a[:, col] < b[:, col]
            cmp_more = a[:, col] > b[:, col]
            less = less | (undecided & cmp_less)
            undecided = undecided & ~cmp_less & ~cmp_more
        # ascending: smaller key at the low index. descending: larger key at the low index.
        low_gets_a = np.where(ascending, less | undecided, ~(less | undecided))
        # equal keys stay (undecided): ascending keeps a at low, which is stable enough for unique keys.
        new = order.copy()
        new[take_lo] = np.where(low_gets_a, order[take_lo], order[other])
        new[other] = np.where(low_gets_a, order[other], order[take_lo])
        order = new
    return order


def cell_coords(pos: np.ndarray, origin: np.ndarray, h: float, res: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Home cell of each particle, clipped to the grid, and the linear id x + rx * (y + ry * z)."""
    res_a = np.asarray(res, dtype=np.int32)
    cell = np.floor((pos - origin) / h).astype(np.int32)
    cell = np.clip(cell, 0, res_a - 1)
    linear = cell[:, 0] + res[0] * (cell[:, 1] + res[1] * cell[:, 2])
    return cell, linear.astype(np.int32)


def bin_particles(linear: np.ndarray, active: np.ndarray, n_cells: int) -> dict:
    """Counting-sort bins. Inactive particles sort after every real cell.

    Within a cell the particle indices increase, which is the gather order the GPU uses.
    """
    n = int(linear.shape[0])
    sentinel = np.int32(n_cells + 1)
    keys_cell = np.where(active, linear, sentinel).astype(np.int32)
    order = np.lexsort((np.arange(n, dtype=np.int32), keys_cell))
    sorted_cell = keys_cell[order]
    valid = sorted_cell < sentinel
    start = np.full(n_cells, -1, dtype=np.int32)
    count = np.zeros(n_cells, dtype=np.int32)
    if np.any(valid):
        cells, first, counts = np.unique(sorted_cell[valid], return_index=True, return_counts=True)
        start[cells] = first.astype(np.int32)
        count[cells] = counts.astype(np.int32)
    return {"order": order.astype(np.int32), "start": start, "count": count, "cell": sorted_cell}


def _scatter_offsets():
    return [(dx, dy, dz) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def p2g(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, origin: np.ndarray, h: float,
        res: tuple[int, int, int], apic: np.ndarray | None = None) -> dict:
    """Gather mass and momentum. `apic` is (N, 3, 3) affine matrices, or None for FLIP/PIC.

    Each of the 27 neighbour offsets is accumulated with `np.add.at` in particle-index order, so a
    cell's sum is the same sum the GPU gather does when it walks that cell's bin (particles were
    sorted by index inside the bin).
    """
    origin = np.asarray(origin, dtype=np.float64)
    res_t = tuple(int(v) for v in res)
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    grid_m = np.zeros(res_t, dtype=np.float64)
    grid_p = np.zeros(res_t + (3,), dtype=np.float64)
    base = np.floor((pos - origin) / h).astype(np.int32)
    alive = mass > 0.0
    res_a = np.asarray(res_t, dtype=np.int32)
    for dx, dy, dz in _scatter_offsets():
        c = base + np.array([dx, dy, dz], dtype=np.int32)
        inside = alive & np.all((c >= 0) & (c < res_a), axis=1)
        if not np.any(inside):
            continue
        cc = c[inside]
        centre = origin + (cc.astype(np.float64) + 0.5) * h
        w = weight((pos[inside] - centre) / h) * mass[inside]
        idx = (cc[:, 0], cc[:, 1], cc[:, 2])
        np.add.at(grid_m, idx, w)
        affine = vel[inside]
        if apic is not None:
            affine = affine + np.einsum("nij,nj->ni", np.asarray(apic)[inside], centre - pos[inside])
        np.add.at(grid_p, idx, affine * w[:, None])
    vel_g = np.zeros_like(grid_p)
    known = grid_m > 0.0
    vel_g[known] = grid_p[known] / grid_m[known, None]
    return {"mass": grid_m, "momentum": grid_p, "vel": vel_g}


def g2p(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, grid_vel: np.ndarray, grid_vel_prev: np.ndarray,
        origin: np.ndarray, h: float, flip_blend: float = 0.0, apic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate grid velocity back to particles.

    `flip_blend` is the FLIP blend: 0 is PIC, 1 is pure FLIP. APIC ignores the blend and returns
    the affine reconstruction; the matrix uses D = h^2 / 4 for the quadratic kernel (Jiang 2015).
    """
    origin = np.asarray(origin, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    res = tuple(int(v) for v in grid_vel.shape[:3])
    res_a = np.asarray(res, dtype=np.int32)
    base = np.floor((pos - origin) / h).astype(np.int32)
    alive = mass > 0.0
    pic = np.zeros_like(vel)
    delta = np.zeros_like(vel)
    matrices = np.zeros((pos.shape[0], 3, 3), dtype=np.float64)
    scale = 4.0 / (h * h)
    for dx, dy, dz in _scatter_offsets():
        c = base + np.array([dx, dy, dz], dtype=np.int32)
        inside = alive & np.all((c >= 0) & (c < res_a), axis=1)
        if not np.any(inside):
            continue
        cc = c[inside]
        centre = origin + (cc.astype(np.float64) + 0.5) * h
        w = weight((pos[inside] - centre) / h)
        idx = (cc[:, 0], cc[:, 1], cc[:, 2])
        gv = grid_vel[idx]
        pic[inside] += w[:, None] * gv
        delta[inside] += w[:, None] * (gv - grid_vel_prev[idx])
        if apic:
            arm = (centre - pos[inside]) * scale
            matrices[inside] += w[:, None, None] * gv[:, :, None] * arm[:, None, :]
    out = vel.copy()
    if apic:
        out[alive] = pic[alive]
    else:
        out[alive] = (1.0 - flip_blend) * pic[alive] + flip_blend * (vel[alive] + delta[alive])
    return out, matrices


def extrapolate(vel: np.ndarray, liquid: np.ndarray, solid: np.ndarray | None = None, bands: int = 3) -> np.ndarray:
    """Copy liquid velocity out into a narrow air band so air-owned faces match the free surface."""
    out = np.array(vel, dtype=np.float64, copy=True)
    known = np.array(liquid, dtype=bool, copy=True)
    if solid is not None:
        known = known & ~np.asarray(solid, dtype=bool)
    for _ in range(int(bands)):
        acc = np.zeros_like(out)
        w = np.zeros(known.shape, dtype=np.float64)
        for axis in range(3):
            for delta in (1, -1):
                known_nb = pressure._sample(known.astype(np.float64), axis, delta, fill=0.0) > 0.5
                for comp in range(3):
                    nb = pressure._sample(out[..., comp], axis, delta, fill=0.0)
                    acc[..., comp] += np.where(known_nb, nb, 0.0)
                w += known_nb
        fill = (~known) & (w > 0.0)
        if solid is not None:
            fill = fill & ~np.asarray(solid, dtype=bool)
        out[fill] = acc[fill] / w[fill, None]
        known = known | fill
    return out


def classify(mass: np.ndarray, threshold: float, solid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    solid_m = np.zeros(mass.shape, dtype=bool) if solid is None else np.asarray(solid, dtype=bool)
    liquid = (mass >= threshold) & ~solid_m
    air = ~liquid & ~solid_m
    return liquid, air, solid_m


def grid_step(vel: np.ndarray, mass: np.ndarray, h: float, dt: float, gravity: np.ndarray, threshold: float,
              solid: np.ndarray | None = None, obstacle_vel: np.ndarray | None = None, cycles: int = 2,
              smooth: int = 4, coarse: int = 24) -> dict:
    """Gravity, free-surface projection, solid velocity. Returns the projected grid and the masks."""
    liquid, air, solid_m = classify(mass, threshold, solid)
    # FLIP's saved grid velocity is this P2G field, before gravity. The particle update is
    # v += (v_projected - v_p2g), which already contains gravity and the pressure impulse.
    # Saving the post-gravity field instead drops gravity and keeps the cancelling pressure.
    p2g_vel = np.array(vel, dtype=np.float64, copy=True)
    out = np.array(vel, dtype=np.float64, copy=True)
    g = np.asarray(gravity, dtype=np.float64)
    out[liquid] = out[liquid] + g * dt
    if obstacle_vel is not None:
        out[solid_m] = obstacle_vel[solid_m]
    else:
        out[solid_m] = 0.0
    before = out.copy()
    out = extrapolate(out, liquid, solid_m, bands=2)
    div = pressure.divergence(out, h, solid_m, obstacle_vel, air)
    solved = pressure.solve(div, h, solid_m, cycles=cycles, smooth=smooth, coarse_iters=coarse, air=air)
    projected = pressure.project(out, solved, h, solid_m, obstacle_vel, air)
    after = pressure.divergence(projected, h, solid_m, obstacle_vel, air)
    return {
        "vel_p2g": p2g_vel, "vel_before": before, "vel": projected, "pressure": solved,
        "liquid": liquid, "air": air, "solid": solid_m,
        "div_before": div, "div_after": after,
        "div_rms": pressure.rms(after, solid_m, air),
        "div_max": float(np.max(np.abs(after[liquid]))) if np.any(liquid) else 0.0,
    }


def seed_box(origin, size, box_min, box_max, spacing: float, velocity=(0.0, 0.0, 0.0), density: float = 1000.0,
             jitter: float = 0.0, seed: int = 0) -> dict:
    """Lattice of equal-mass particles inside `box`, clipped to the domain inset by half a spacing."""
    origin = np.asarray(origin, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    lo = np.maximum(np.asarray(box_min, dtype=np.float64), origin + 0.5 * spacing)
    hi = np.minimum(np.asarray(box_max, dtype=np.float64), origin + size - 0.5 * spacing)
    axes = [np.arange(lo[a] + 0.5 * spacing, hi[a], spacing) for a in range(3)]
    if any(len(axis) == 0 for axis in axes):
        raise ValueError("seed box is empty")
    xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    pos = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        pos = pos + (rng.random(pos.shape) - 0.5) * 2.0 * jitter * spacing
        pos = np.clip(pos, origin + 0.5 * spacing, origin + size - 0.5 * spacing)
    n = pos.shape[0]
    vel = np.tile(np.asarray(velocity, dtype=np.float64), (n, 1))
    # One particle represents a spacing^3 cube of fluid, so the lattice volume is the liquid volume.
    mass = np.full(n, density * spacing ** 3, dtype=np.float64)
    return {"pos": pos, "vel": vel, "mass": mass, "spacing": float(spacing), "density": float(density)}


def collide_domain(pos, vel, origin, size, radius: float):
    origin = np.asarray(origin, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    lo = origin + radius
    hi = origin + size - radius
    for axis in range(3):
        below = pos[:, axis] < lo[axis]
        above = pos[:, axis] > hi[axis]
        pos[below, axis] = lo[axis]
        pos[above, axis] = hi[axis]
        vel[below, axis] = np.maximum(vel[below, axis], 0.0)
        vel[above, axis] = np.minimum(vel[above, axis], 0.0)
    return pos, vel


def collide_sdf(pos, vel, sdf_fn, radius: float):
    """Push particles out of a solid. `sdf_fn(points) -> (sdf, gradient, velocity)`."""
    if sdf_fn is None:
        return pos, vel
    sdf, grad, obs = sdf_fn(pos)
    inside = sdf < radius
    if not np.any(inside):
        return pos, vel
    n = grad[inside]
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-8)
    pos[inside] = pos[inside] + n * (radius - sdf[inside])[:, None]
    # Inelastic: kill the inward radial speed and take the obstacle's tangential velocity.
    radial = np.sum(vel[inside] * n, axis=1)
    inward = radial < 0.0
    vel[inside] = vel[inside] - (inward * radial)[:, None] * n
    vel[inside] = 0.5 * vel[inside] + 0.5 * obs[inside]
    return pos, vel


def liquid_volume(mass: np.ndarray, density: float) -> float:
    alive = mass > 0.0
    if not np.any(alive) or density <= 0.0:
        return 0.0
    return float(mass[alive].sum() / density)


def step_particles(state: dict, origin, size, h: float, res, dt: float, gravity=(0.0, -9.81, 0.0),
                   flip: float = 0.95, threshold_frac: float = 0.2, solid: np.ndarray | None = None,
                   obstacle_vel: np.ndarray | None = None, sdf_fn=None, project: bool = True,
                   apic: bool = False, cycles: int = 2, smooth: int = 4, coarse: int = 24) -> dict:
    """One FLIP step. State is pos, vel, mass, and optional `C` (N, 3, 3)."""
    origin = np.asarray(origin, dtype=np.float64)
    res_t = tuple(int(v) for v in res)
    h = float(h)
    transferred = p2g(state["pos"], state["vel"], state["mass"], origin, h, res_t,
                      apic=state.get("C") if apic else None)
    density = float(state.get("density", 1000.0))
    # A full cell holds about density * h^3. threshold_frac of that marks the surface.
    threshold = threshold_frac * density * (h ** 3)
    if project:
        grid = grid_step(transferred["vel"], transferred["mass"], h, dt, np.asarray(gravity, dtype=np.float64),
                         threshold, solid, obstacle_vel, cycles=cycles, smooth=smooth, coarse=coarse)
        grid_vel = grid["vel"]
    else:
        grid_vel = transferred["vel"]
        grid = {"vel": grid_vel, "vel_p2g": transferred["vel"], "vel_before": transferred["vel"],
                "div_rms": 0.0, "div_max": 0.0, "liquid": transferred["mass"] >= threshold,
                "div_after": np.zeros(res_t)}
    new_vel, matrices = g2p(state["pos"], state["vel"], state["mass"], grid_vel, grid["vel_p2g"],
                            origin, h, flip_blend=0.0 if apic else flip, apic=apic)
    new_pos = state["pos"] + new_vel * dt
    radius = 0.5 * float(state["spacing"])
    new_pos, new_vel = collide_domain(new_pos, new_vel, origin, size, radius)
    new_pos, new_vel = collide_sdf(new_pos, new_vel, sdf_fn, radius)
    out = {
        "pos": new_pos, "vel": new_vel, "mass": state["mass"].copy(), "spacing": state["spacing"],
        "density": density, "grid": grid, "transfer": transferred,
    }
    if apic:
        out["C"] = matrices
    return out


def column_heights(pos: np.ndarray, mass: np.ndarray, origin, size, bins: int = 8) -> np.ndarray:
    """Max particle y in each x/z bin. A flat free surface has a small standard deviation."""
    alive = mass > 0.0
    p = pos[alive]
    if p.size == 0:
        return np.zeros(0)
    origin = np.asarray(origin, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    ix = np.clip(((p[:, 0] - origin[0]) / size[0] * bins).astype(int), 0, bins - 1)
    iz = np.clip(((p[:, 2] - origin[2]) / size[2] * bins).astype(int), 0, bins - 1)
    heights = np.full(bins * bins, np.nan)
    flat = ix + bins * iz
    for b in range(bins * bins):
        sel = flat == b
        if np.any(sel):
            heights[b] = float(p[sel, 1].max())
    return heights[np.isfinite(heights)]


def front_x(pos: np.ndarray, mass: np.ndarray, y_max: float) -> float:
    """Rightmost particle on the floor band, the dam-break front."""
    alive = (mass > 0.0) & (pos[:, 1] <= y_max)
    if not np.any(alive):
        alive = mass > 0.0
    if not np.any(alive):
        return 0.0
    return float(pos[alive, 0].max())
