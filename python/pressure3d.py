"""Reference pressure projection for the 3D collocated solver.

The discrete divergence and the pressure gradient are adjoints of the 7-point Laplacian, so a
projection can actually drive the measured divergence down. Central differences over two cells
compose to a wider stencil than the Jacobi operator; that mismatch is why a correct-looking
multigrid was only removing about a tenth of the divergence.

    div_x[c] = (flux(c) - flux(c - e_x)) / h
    flux(c)  = v_x[c]                         when c+e_x is a fluid cell
             = obstacle velocity             when that neighbour is solid
             = 0                             when it is outside the domain
    grad_x[c] = (p[c+e_x] - p[c]) / h        when c+e_x is fluid, else 0

Walls and solids are Neumann: the missing neighbour is omitted from the Laplacian, which is the
same mirror the Jacobi iteration uses. The Poisson problem is singular (constants are invisible
to the gradient), so the right-hand side mean over fluid cells is removed before the solve.
"""

from __future__ import annotations

import numpy as np

def _sample(a, axis, delta, fill=0.0):
    """Value of the neighbour `delta` steps along `axis` (positive delta is +axis). Out of range is `fill`."""
    out = np.full_like(a, fill, dtype=np.float64)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if delta > 0:
        src[axis] = slice(delta, None)
        dst[axis] = slice(0, -delta)
    elif delta < 0:
        src[axis] = slice(0, delta)
        dst[axis] = slice(-delta, None)
    else:
        return np.asarray(a, dtype=np.float64).copy()
    out[tuple(dst)] = np.asarray(a, dtype=np.float64)[tuple(src)]
    return out


def fluid_mask(solid: np.ndarray | None, shape) -> np.ndarray:
    if solid is None:
        return np.ones(shape, dtype=bool)
    return ~np.asarray(solid, dtype=bool)


def _neighbour_fluid(fluid: np.ndarray, axis: int, delta: int) -> np.ndarray:
    return _sample(fluid.astype(np.float64), axis, delta, fill=0.0) > 0.5


def cell_kinds(shape, solid: np.ndarray | None = None, air: np.ndarray | None = None):
    """Liquid / air / solid masks. `air is None` is the smoke path: every non-solid cell is fluid."""
    if solid is None:
        solid_m = np.zeros(shape, dtype=bool)
    else:
        solid_m = np.asarray(solid, dtype=bool)
    if air is None:
        return ~solid_m, None, solid_m
    air_m = np.asarray(air, dtype=bool) & ~solid_m
    return ~solid_m & ~air_m, air_m, solid_m


def divergence(vel: np.ndarray, h: float = 1.0, solid: np.ndarray | None = None,
               obstacle_vel: np.ndarray | None = None, air: np.ndarray | None = None) -> np.ndarray:
    """RHS of the projection: discrete divergence of velocity (1/s when h is a world size).

    With `air` set, the unknown lives on liquid cells only. A face between liquid and air carries
    the velocity of the cell that owns that face (the MAC +face), and an air cell that owns a face
    into liquid contributes the velocity stored there. Callers extrapolate liquid velocity into that
    one-cell air band first, so a uniform field has no divergence at the free surface. Solids and
    closed domain faces are unchanged.
    """
    if air is None:
        fluid = fluid_mask(solid, vel.shape[:3])
        div = np.zeros(vel.shape[:3], dtype=np.float64)
        for axis in range(3):
            bound = None if obstacle_vel is None else obstacle_vel[..., axis]
            flux = _face_flux(vel[..., axis], fluid, bound, axis)
            div += (flux - _sample(flux, axis, -1, fill=0.0)) / h
        return np.where(fluid, div, 0.0)
    liquid, air_m, _solid_m = cell_kinds(vel.shape[:3], solid, air)
    div = np.zeros(vel.shape[:3], dtype=np.float64)
    for axis in range(3):
        bound = None if obstacle_vel is None else obstacle_vel[..., axis]
        flux = _face_flux_liquid(vel[..., axis], liquid, air_m, bound, axis)
        div += (flux - _sample(flux, axis, -1, fill=0.0)) / h
    return np.where(liquid, div, 0.0)


def _face_flux(comp, fluid, bound, axis: int) -> np.ndarray:
    """Flux through the face on the +side of each cell.

    Both cells fluid: the owner's velocity component. A solid on either side prescribes the
    obstacle velocity (Neumann pressure does not change that flux). Outside the domain the
    flux is zero, a closed wall.
    """
    plus_in = _sample(np.ones(fluid.shape, dtype=np.float64), axis, 1, fill=0.0) > 0.5
    plus_fluid = _neighbour_fluid(fluid, axis, 1)
    if bound is None:
        bound = np.zeros(fluid.shape, dtype=np.float64)
    bound_nb = _sample(bound, axis, 1, fill=0.0)
    both_fluid = fluid & plus_fluid
    # Neighbour solid (in range) -> obstacle velocity there. Owner solid and neighbour fluid ->
    # obstacle velocity here. Anything else that is not two fluid cells is a wall (0) when the
    # neighbour is out of range.
    solid_flux = np.where(plus_fluid, bound, bound_nb)
    return np.where(both_fluid, comp, np.where(plus_in, solid_flux, 0.0))


def _face_flux_liquid(comp, liquid, air, bound, axis: int) -> np.ndarray:
    """+face flux when air is a Dirichlet free surface and solid is Neumann.

    Liquid owner, liquid or air neighbour: this cell's component (the face it owns).
    Air owner, liquid neighbour: the air cell's stored component. Extrapolation writes the
    neighbouring liquid velocity there before the solve, and the projection updates that same
    face, so the flux stays the adjoint of the gradient below.
    A solid on either side of an in-range face uses the obstacle velocity. Outside the domain
    the flux is 0 (closed wall).
    """
    plus_in = _sample(np.ones(liquid.shape, dtype=np.float64), axis, 1, fill=0.0) > 0.5
    plus_liquid = _neighbour_fluid(liquid, axis, 1)
    plus_air = _neighbour_fluid(air, axis, 1)
    plus_solid = plus_in & ~plus_liquid & ~plus_air
    if bound is None:
        bound = np.zeros(liquid.shape, dtype=np.float64)
    bound_nb = _sample(bound, axis, 1, fill=0.0)
    owner_solid = ~liquid & ~air
    flux = np.zeros(liquid.shape, dtype=np.float64)
    flux = np.where(liquid & (plus_liquid | plus_air), comp, flux)
    flux = np.where(liquid & plus_solid, bound_nb, flux)
    flux = np.where(owner_solid & plus_in, bound, flux)
    flux = np.where(air & plus_liquid, comp, flux)
    flux = np.where(air & plus_solid, bound_nb, flux)
    return flux


def gradient(pressure: np.ndarray, h: float = 1.0, solid: np.ndarray | None = None,
             air: np.ndarray | None = None) -> np.ndarray:
    """Forward difference of pressure, zero on a face that touches a solid or the domain wall.

    Air on the far side of a face is Dirichlet (pressure 0). An air cell that owns a face into
    liquid carries (p_liquid - 0) / h so the projection updates that face velocity.
    """
    if air is None:
        fluid = fluid_mask(solid, pressure.shape)
        grad = np.zeros(pressure.shape + (3,), dtype=np.float64)
        for axis in range(3):
            plus = _sample(pressure, axis, 1, fill=0.0)
            plus_fluid = _neighbour_fluid(fluid, axis, 1)
            grad[..., axis] = np.where(fluid & plus_fluid, (plus - pressure) / h, 0.0)
        return grad
    liquid, air_m, _solid_m = cell_kinds(pressure.shape, solid, air)
    grad = np.zeros(pressure.shape + (3,), dtype=np.float64)
    for axis in range(3):
        plus = _sample(pressure, axis, 1, fill=0.0)
        plus_liquid = _neighbour_fluid(liquid, axis, 1)
        plus_air = _neighbour_fluid(air_m, axis, 1)
        g = np.zeros(pressure.shape, dtype=np.float64)
        g = np.where(liquid & plus_liquid, (plus - pressure) / h, g)
        g = np.where(liquid & plus_air, (0.0 - pressure) / h, g)
        g = np.where(air_m & plus_liquid, plus / h, g)
        grad[..., axis] = g
    return grad


def project(vel: np.ndarray, pressure: np.ndarray, h: float = 1.0, solid: np.ndarray | None = None,
            obstacle_vel: np.ndarray | None = None, air: np.ndarray | None = None) -> np.ndarray:
    out = vel - gradient(pressure, h, solid, air)
    if solid is not None:
        if obstacle_vel is None:
            obstacle_vel = np.zeros_like(vel)
        out = np.where(np.asarray(solid, dtype=bool)[..., None], obstacle_vel, out)
    return out


def laplacian(pressure: np.ndarray, h: float = 1.0, solid: np.ndarray | None = None,
              air: np.ndarray | None = None) -> np.ndarray:
    """div(grad p). Solids contribute nothing; a solid or outside neighbour is Neumann.

    An air neighbour is Dirichlet. The operator is exactly div(grad p), so the Jacobi stencil
    below has to match it (see `jacobi`).
    """
    return divergence(gradient(pressure, h, solid, air), h, solid, air=air)


def residual(pressure, rhs, h=1.0, solid=None, air=None) -> np.ndarray:
    if air is None:
        fluid = fluid_mask(solid, rhs.shape)
    else:
        fluid = cell_kinds(rhs.shape, solid, air)[0]
    r = np.where(fluid, rhs - laplacian(pressure, h, solid, air), 0.0)
    return r


def remove_mean(rhs, solid=None) -> np.ndarray:
    fluid = fluid_mask(solid, rhs.shape)
    if not np.any(fluid):
        return rhs
    mean = float(rhs[fluid].mean())
    out = rhs.copy()
    out[fluid] -= mean
    return out


def jacobi(pressure, rhs, h=1.0, solid=None, iterations=1, omega=2.0 / 3.0, air=None) -> np.ndarray:
    """Weighted Jacobi. omega 1 is plain Jacobi; 2/3 is the usual multigrid smoother.

    Without `air`, a missing neighbour mirrors pressure (Neumann), which is the smoke operator.
    With `air`, an air neighbour contributes 0 (Dirichlet free surface) and a solid or closed
    wall still mirrors. The diagonal stays 6, matching div(grad p) on liquid cells.
    """
    if air is None:
        fluid = fluid_mask(solid, rhs.shape)
        air_m = None
    else:
        fluid, air_m, _solid_m = cell_kinds(rhs.shape, solid, air)
    p = np.array(pressure, dtype=np.float64, copy=True)
    for _ in range(int(iterations)):
        acc = np.zeros_like(p)
        diag = np.zeros_like(p)
        for axis in range(3):
            for delta in (1, -1):
                nb = _sample(p, axis, delta, fill=0.0)
                nb_fluid = _neighbour_fluid(fluid, axis, delta)
                if air_m is None:
                    # Neumann: a missing neighbour mirrors p, which adds p into the 6-term sum.
                    acc += np.where(nb_fluid, nb, p)
                else:
                    nb_air = _neighbour_fluid(air_m, axis, delta)
                    acc += np.where(nb_fluid, nb, np.where(nb_air, 0.0, p))
                diag += 1.0
        # diag is 6 everywhere. Fixed point matches (sum_pAt - 6p) / h^2 = rhs.
        guess = (acc - (h * h) * rhs) / np.maximum(diag, 1.0)
        updated = (1.0 - omega) * p + omega * guess
        p = np.where(fluid, updated, 0.0)
    return p


def _weights_1d():
    # Cell-centred full weighting, offsets -1,0,1,2 relative to 2*I. Sums to 1.
    return {(-1): 0.125, 0: 0.375, 1: 0.375, 2: 0.125}


def restrict(fine: np.ndarray, solid_fine: np.ndarray | None = None,
             air_fine: np.ndarray | None = None, renorm: bool = True) -> np.ndarray:
    """Full-weighting restriction onto a grid of half the resolution (rounded down, at least 2).

    Air cells are not unknowns. They are skipped the same way solids are, so the coarse
    right-hand side only sees the liquid residual. Renormalising that partial stencil
    amplifies the free-surface residual (the missing air samples are treated as if they
    carried the same residual) and the V-cycle diverges once the grid is finer than 16.
    The liquid path therefore keeps the un-normalised weights, which still sum to 1 in
    the interior of the liquid.
    """
    if air_fine is not None:
        inactive = np.asarray(air_fine, dtype=bool)
        if solid_fine is not None:
            inactive = inactive | np.asarray(solid_fine, dtype=bool)
        return restrict(fine, inactive, renorm=False)
    nf = fine.shape
    nc = tuple(max(2, n // 2) for n in nf)
    out = np.zeros(nc, dtype=np.float64)
    wsum = np.zeros(nc, dtype=np.float64)
    fluid = fluid_mask(solid_fine, nf)
    w1 = _weights_1d()
    # Only the exact half-resolution interior uses the 4-tap stencil. A coarse grid that did not
    # halve falls back to the average of the fine cells it covers.
    exact = all(nc[a] * 2 == nf[a] for a in range(3))
    if exact:
        for dz, wz in w1.items():
            for dy, wy in w1.items():
                for dx, wx in w1.items():
                    w = wx * wy * wz
                    xs = np.arange(nc[0]) * 2 + dx
                    ys = np.arange(nc[1]) * 2 + dy
                    zs = np.arange(nc[2]) * 2 + dz
                    x_ok = (xs >= 0) & (xs < nf[0])
                    y_ok = (ys >= 0) & (ys < nf[1])
                    z_ok = (zs >= 0) & (zs < nf[2])
                    xs_c = np.clip(xs, 0, nf[0] - 1)
                    ys_c = np.clip(ys, 0, nf[1] - 1)
                    zs_c = np.clip(zs, 0, nf[2] - 1)
                    sample = fine[np.ix_(xs_c, ys_c, zs_c)]
                    fmask = fluid[np.ix_(xs_c, ys_c, zs_c)]
                    valid = x_ok[:, None, None] & y_ok[None, :, None] & z_ok[None, None, :] & fmask
                    out += w * np.where(valid, sample, 0.0)
                    wsum += w * valid
        if renorm:
            return np.where(wsum > 0, out / np.maximum(wsum, 1e-12), 0.0)
        return out
    # Box average, used when the coarse size is clamped.
    for z in range(nc[2]):
        for y in range(nc[1]):
            for x in range(nc[0]):
                x0, x1 = x * nf[0] // nc[0], max((x + 1) * nf[0] // nc[0], x * nf[0] // nc[0] + 1)
                y0, y1 = y * nf[1] // nc[1], max((y + 1) * nf[1] // nc[1], y * nf[1] // nc[1] + 1)
                z0, z1 = z * nf[2] // nc[2], max((z + 1) * nf[2] // nc[2], z * nf[2] // nc[2] + 1)
                block = fine[x0:x1, y0:y1, z0:z1]
                mask = fluid[x0:x1, y0:y1, z0:z1]
                if np.any(mask):
                    out[x, y, z] = float(block[mask].mean())
    return out


def restrict_solid(solid: np.ndarray | None, shape) -> np.ndarray | None:
    if solid is None:
        return None
    coarse = restrict(solid.astype(np.float64))
    return coarse >= 0.5


def _any_block(mask: np.ndarray) -> np.ndarray:
    """True on a coarse cell when any of the 2x2x2 fine children is true. Odd sizes drop the last fine cell."""
    mask = np.asarray(mask, dtype=bool)
    nf = mask.shape
    nc = tuple(max(2, n // 2) for n in nf)
    if all(nc[a] * 2 == nf[a] for a in range(3)):
        m = mask
        return (m[0::2, 0::2, 0::2] | m[1::2, 0::2, 0::2] | m[0::2, 1::2, 0::2] | m[1::2, 1::2, 0::2]
                | m[0::2, 0::2, 1::2] | m[1::2, 0::2, 1::2] | m[0::2, 1::2, 1::2] | m[1::2, 1::2, 1::2])
    out = np.zeros(nc, dtype=bool)
    for z in range(nc[2]):
        for y in range(nc[1]):
            for x in range(nc[0]):
                block = mask[x * 2:x * 2 + 2, y * 2:y * 2 + 2, z * 2:z * 2 + 2]
                out[x, y, z] = bool(np.any(block))
    return out


def coarsen_masks(solid, air, fine_shape):
    """Coarse solid (majority) and coarse air (neither majority-solid nor any-liquid child).

    A coarse cell that still contains liquid stays liquid, so the free surface does not disappear
    between levels. `air is None` keeps the smoke path (no air mask).
    """
    if air is None:
        return restrict_solid(solid, fine_shape), None
    solid_m = np.zeros(fine_shape, dtype=bool) if solid is None else np.asarray(solid, dtype=bool)
    air_m = np.asarray(air, dtype=bool) & ~solid_m
    liquid = ~solid_m & ~air_m
    solid_c = restrict_solid(solid_m, fine_shape)
    if solid_c is None:
        solid_c = np.zeros(_any_block(liquid).shape, dtype=bool)
    liquid_c = _any_block(liquid)
    if solid_c.shape != liquid_c.shape:
        solid_c = np.zeros(liquid_c.shape, dtype=bool)
    liquid_c = liquid_c & ~solid_c
    air_c = ~liquid_c & ~solid_c
    return solid_c, air_c


def prolong(coarse: np.ndarray, shape) -> np.ndarray:
    """Trilinear prolongation onto `shape`. Cell centres share the same [0,1] domain."""
    nf = shape
    nc = coarse.shape
    out = np.zeros(nf, dtype=np.float64)
    # Coarse centre I sits at (I+0.5)/nc, fine centre i at (i+0.5)/nf.
    xs = (np.arange(nf[0]) + 0.5) * (nc[0] / nf[0]) - 0.5
    ys = (np.arange(nf[1]) + 0.5) * (nc[1] / nf[1]) - 0.5
    zs = (np.arange(nf[2]) + 0.5) * (nc[2] / nf[2]) - 0.5
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    z0 = np.floor(zs).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    tx = xs - x0
    ty = ys - y0
    tz = zs - z0
    x0c = np.clip(x0, 0, nc[0] - 1)
    x1c = np.clip(x1, 0, nc[0] - 1)
    y0c = np.clip(y0, 0, nc[1] - 1)
    y1c = np.clip(y1, 0, nc[1] - 1)
    z0c = np.clip(z0, 0, nc[2] - 1)
    z1c = np.clip(z1, 0, nc[2] - 1)

    def at(ix, iy, iz):
        return coarse[np.ix_(ix, iy, iz)]

    c000 = at(x0c, y0c, z0c)
    c100 = at(x1c, y0c, z0c)
    c010 = at(x0c, y1c, z0c)
    c110 = at(x1c, y1c, z0c)
    c001 = at(x0c, y0c, z1c)
    c101 = at(x1c, y0c, z1c)
    c011 = at(x0c, y1c, z1c)
    c111 = at(x1c, y1c, z1c)
    tx = tx[:, None, None]
    ty = ty[None, :, None]
    tz = tz[None, None, :]
    c00 = c000 * (1 - tx) + c100 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz


def vcycle(pressure, rhs, h=1.0, solid=None, smooth=3, coarse_iters=40, levels_left=6, air=None) -> np.ndarray:
    """One geometric V-cycle. Coarse spacing is 2h and the operator is rediscretised.

    An air mask is coarsened with the liquid region kept wherever any fine child is liquid, and the
    same Dirichlet/Neumann stencil is rediscretised on that coarse mask.
    """
    if air is None:
        fluid = fluid_mask(solid, rhs.shape)
    else:
        fluid = cell_kinds(rhs.shape, solid, air)[0]
    n = min(rhs.shape)
    if n <= 4 or levels_left <= 1:
        return jacobi(pressure, rhs, h, solid, iterations=coarse_iters, omega=1.0, air=air)
    p = jacobi(pressure, rhs, h, solid, iterations=smooth, omega=2.0 / 3.0, air=air)
    res = residual(p, rhs, h, solid, air)
    res_c = restrict(res, solid, air)
    solid_c, air_c = coarsen_masks(solid, air, rhs.shape)
    # Rediscretised operator uses H = 2h, so the coarse unknown is the correction in pressure units
    # and the restricted residual is already the coarse right-hand side (full weighting preserves a constant).
    corr = vcycle(np.zeros(res_c.shape), res_c, h * 2.0, solid_c, smooth, coarse_iters, levels_left - 1, air_c)
    p = p + prolong(corr, p.shape)
    p = np.where(fluid, p, 0.0)
    p = jacobi(p, rhs, h, solid, iterations=smooth, omega=2.0 / 3.0, air=air)
    return p


def _liquid_dot(a, b, liquid) -> float:
    return float(np.sum(a[liquid] * b[liquid]))


def mgpcg(rhs, h=1.0, solid=None, air=None, iterations=8, smooth=1, coarse_iters=16) -> tuple[np.ndarray, list]:
    """Multigrid-preconditioned CG for the free-surface Poisson problem (McAdams et al. 2010).

    The system is A p = b with A = -div(grad) and b = -div(velocity), which is SPD on liquid
    cells once air is Dirichlet. One symmetric V-cycle (damped Jacobi, equal pre/post sweeps)
    preconditions each iteration. The returned history is ||r|| / ||r0|| after every iteration,
    including 1 at the start.
    """
    if air is None or not np.any(air):
        solved = solve(rhs, h, solid, cycles=max(2, iterations // 2), smooth=max(smooth, 2),
                       coarse_iters=coarse_iters, air=None)
        return solved, []
    liquid = cell_kinds(rhs.shape, solid, air)[0]

    def apply(p):
        return np.where(liquid, -laplacian(p, h, solid, air), 0.0)

    def precondition(residual_cg):
        # A z = r_cg  <=>  laplacian(z) = -r_cg.
        return np.where(
            liquid,
            vcycle(np.zeros_like(residual_cg), np.where(liquid, -residual_cg, 0.0),
                   h, solid, smooth, coarse_iters, 6, air),
            0.0,
        )

    b = np.where(liquid, -rhs, 0.0)
    x = np.zeros_like(rhs, dtype=np.float64)
    r = b.copy()
    z = precondition(r)
    direction = z.copy()
    rho = _liquid_dot(r, z, liquid)
    r0 = np.sqrt(max(_liquid_dot(r, r, liquid), 1e-30))
    history = [1.0]
    for _ in range(int(iterations)):
        Ad = apply(direction)
        denom = _liquid_dot(direction, Ad, liquid)
        if abs(denom) < 1e-18:
            break
        alpha = rho / denom
        x = np.where(liquid, x + alpha * direction, 0.0)
        r = np.where(liquid, r - alpha * Ad, 0.0)
        rel = np.sqrt(max(_liquid_dot(r, r, liquid), 0.0)) / r0
        history.append(float(rel))
        if rel < 1e-8:
            break
        z = precondition(r)
        rho2 = _liquid_dot(r, z, liquid)
        direction = z + (rho2 / rho) * direction
        rho = rho2
    return x, history


def solve(rhs, h=1.0, solid=None, cycles=2, smooth=3, coarse_iters=40, air=None) -> np.ndarray:
    # Pure Neumann (smoke, or a tank with no air) is singular, so the mean is removed. A free
    # surface is Dirichlet and the mean is part of the hydrostatic load, so it stays.
    if air is None or not np.any(air):
        rhs = remove_mean(rhs, solid)
    p = np.zeros_like(rhs, dtype=np.float64)
    for _ in range(int(cycles)):
        p = vcycle(p, rhs, h, solid, smooth=smooth, coarse_iters=coarse_iters, air=air)
    return p


def rms(a, solid=None, air=None) -> float:
    if air is None:
        fluid = fluid_mask(solid, a.shape)
    else:
        fluid = cell_kinds(a.shape, solid, air)[0]
    data = a[fluid]
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(data ** 2)))


def reduction(vel, h=1.0, solid=None, obstacle_vel=None, cycles=2, smooth=3, coarse_iters=40, air=None) -> dict:
    """Divergence RMS before and after one projection."""
    div0 = divergence(vel, h, solid, obstacle_vel, air)
    if air is None or not np.any(air):
        rhs = remove_mean(div0, solid)
        before = rms(rhs, solid)
    else:
        rhs = div0
        before = rms(rhs, solid, air)
    pressure = solve(rhs, h, solid, cycles=cycles, smooth=smooth, coarse_iters=coarse_iters, air=air)
    after_vel = project(vel, pressure, h, solid, obstacle_vel, air)
    div1 = divergence(after_vel, h, solid, obstacle_vel, air)
    if air is None or not np.any(air):
        after = rms(remove_mean(div1, solid), solid)
    else:
        after = rms(div1, solid, air)
    return {"before": before, "after": after, "factor": (before / after) if after > 0 else None}
