"""Voronoi fracture of a closed convex solid. Pure NumPy (no SciPy).

A box, a tessellated sphere, or any closed triangle mesh is clipped by the half-spaces of a seeded
Voronoi diagram. Each piece is the original solid intersected with one cell, so the pieces are
closed, disjoint, and their volumes sum to the solid that was cut. A non-convex mesh is replaced by
its convex hull first: a Voronoi cell is a convex clip, and the hull is the solid those clips
partition. Seeds are drawn from a NumPy Generator (PCG64), so the same seed and pattern repeat.

Interior faces are the cuts. Planes are outward, `n · x = d`, with the solid on the side `n · x <= d`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


def _as_f64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _norm(v) -> float:
    return float(np.linalg.norm(v))


def _unit(v) -> np.ndarray:
    n = _norm(v)
    if n <= _EPS:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _face_normal(verts: np.ndarray) -> np.ndarray:
    if len(verts) < 3:
        return np.zeros(3, dtype=np.float64)
    n = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    for i in range(2, len(verts) - 1):
        n = n + np.cross(verts[i] - verts[0], verts[i + 1] - verts[0])
    return _unit(n)


def _subexpressions(w0, w1, w2):
    temp0 = w0 + w1
    f1 = temp0 + w2
    temp1 = w0 * w0
    temp2 = temp1 + w1 * temp0
    f2 = temp2 + w2 * f1
    f3 = w0 * temp1 + w1 * temp2 + w2 * f2
    g0 = f2 + w0 * (f1 + w0)
    g1 = f2 + w1 * (f1 + w1)
    g2 = f2 + w2 * (f1 + w2)
    return f1, f2, f3, g0, g1, g2


def mass_properties(triangles: np.ndarray, density: float = 1.0):
    """Volume, centre of mass, and inertia about the centre of mass.

    `triangles` is (T, 3, 3) with outward winding. The integrals are the polyhedral mass properties
    from Eberly / Mirtich. A box of half-extents (hx, hy, hz) and mass m has
    Ixx = m (hy² + hz²) / 3, and the same with the axes cycled.
    """
    tris = _as_f64(triangles).reshape(-1, 3, 3)
    intg = np.zeros(10, dtype=np.float64)
    for tri in tris:
        x0, y0, z0 = tri[0]
        x1, y1, z1 = tri[1]
        x2, y2, z2 = tri[2]
        a1, b1, c1 = x1 - x0, y1 - y0, z1 - z0
        a2, b2, c2 = x2 - x0, y2 - y0, z2 - z0
        d0 = b1 * c2 - b2 * c1
        d1 = a2 * c1 - a1 * c2
        d2 = a1 * b2 - a2 * b1
        f1x, f2x, f3x, g0x, g1x, g2x = _subexpressions(x0, x1, x2)
        f1y, f2y, f3y, g0y, g1y, g2y = _subexpressions(y0, y1, y2)
        f1z, f2z, f3z, g0z, g1z, g2z = _subexpressions(z0, z1, z2)
        intg[0] += d0 * f1x
        intg[1] += d0 * f2x
        intg[2] += d1 * f2y
        intg[3] += d2 * f2z
        intg[4] += d0 * f3x
        intg[5] += d1 * f3y
        intg[6] += d2 * f3z
        intg[7] += d0 * (y0 * g0x + y1 * g1x + y2 * g2x)
        intg[8] += d1 * (z0 * g0y + z1 * g1y + z2 * g2y)
        intg[9] += d2 * (x0 * g0z + x1 * g1z + x2 * g2z)
    intg *= np.array([1 / 6, 1 / 24, 1 / 24, 1 / 24, 1 / 60, 1 / 60, 1 / 60, 1 / 120, 1 / 120, 1 / 120])
    volume = float(intg[0])
    if abs(volume) <= _EPS:
        return 0.0, np.zeros(3), np.zeros((3, 3)), 0.0
    com = intg[1:4] / volume
    density = float(density)
    mass = density * volume
    # intg[4..9] are the origin integrals of x², y², z², xy, yz, zx. Scale by density, then
    # the parallel-axis shift puts the tensor on the centre of mass. Off-diagonals are −∫xy.
    x, y, z = (float(v) for v in com)
    ixx = density * (intg[5] + intg[6]) - mass * (y * y + z * z)
    iyy = density * (intg[4] + intg[6]) - mass * (x * x + z * z)
    izz = density * (intg[4] + intg[5]) - mass * (x * x + y * y)
    ixy = density * (-intg[7]) + mass * x * y
    iyz = density * (-intg[8]) + mass * y * z
    izx = density * (-intg[9]) + mass * z * x
    inertia = np.array([[ixx, ixy, izx], [ixy, iyy, iyz], [izx, iyz, izz]], dtype=np.float64)
    return volume, com, inertia, mass


def _triangles_of(faces: list[dict]) -> np.ndarray:
    tris = []
    interior = []
    for face in faces:
        v = face["verts"]
        flag = bool(face.get("interior", False))
        for i in range(1, len(v) - 1):
            tris.append((v[0], v[i], v[i + 1]))
            interior.append(flag)
    if not tris:
        return np.zeros((0, 3, 3)), np.zeros((0,), dtype=bool)
    return np.asarray(tris, dtype=np.float64), np.asarray(interior, dtype=bool)


@dataclass
class Solid:
    """Closed convex polyhedron. Faces wind outward."""

    faces: list

    def copy(self) -> "Solid":
        return Solid([{"verts": f["verts"].copy(), "interior": bool(f.get("interior", False))} for f in self.faces])

    def vertices(self) -> np.ndarray:
        if not self.faces:
            return np.zeros((0, 3))
        return np.vstack([f["verts"] for f in self.faces])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        v = self.vertices()
        if len(v) == 0:
            z = np.zeros(3)
            return z, z
        return v.min(axis=0), v.max(axis=0)

    def planes(self) -> np.ndarray:
        """(F, 4) outward planes, n · x <= d inside."""
        out = []
        for face in self.faces:
            n = _face_normal(face["verts"])
            if _norm(n) <= _EPS:
                continue
            d = float(np.dot(n, face["verts"][0]))
            out.append((n[0], n[1], n[2], d))
        if not out:
            return np.zeros((0, 4))
        return np.asarray(out, dtype=np.float64)

    def contains(self, points, tol: float = 1e-8) -> np.ndarray:
        pts = _as_f64(points).reshape(-1, 3)
        planes = self.planes()
        if len(planes) == 0 or len(pts) == 0:
            return np.zeros(len(pts), dtype=bool)
        sd = pts @ planes[:, :3].T - planes[:, 3]
        return np.max(sd, axis=1) <= tol

    def sdf(self, points) -> np.ndarray:
        """Exact convex SDF (negative inside): max over outward planes."""
        pts = _as_f64(points).reshape(-1, 3)
        planes = self.planes()
        if len(planes) == 0 or len(pts) == 0:
            return np.full(len(pts), 1e4)
        return np.max(pts @ planes[:, :3].T - planes[:, 3], axis=1)

    def volume(self) -> float:
        tris, _ = _triangles_of(self.faces)
        if len(tris) == 0:
            return 0.0
        vol, _, _, _ = mass_properties(tris, 1.0)
        return float(vol)


def _orient_outward(faces: list[dict]) -> list[dict]:
    tris, _ = _triangles_of(faces)
    if len(tris) == 0:
        return faces
    vol, _, _, _ = mass_properties(tris, 1.0)
    if vol >= 0.0:
        return faces
    flipped = []
    for face in faces:
        flipped.append({"verts": face["verts"][::-1].copy(), "interior": bool(face.get("interior", False))})
    return flipped


def box_solid(center=(0.0, 0.0, 0.0), half=(0.5, 0.5, 0.5)) -> Solid:
    c = _as_f64(center)
    h = np.abs(_as_f64(half))
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
    p = c + signs * h
    # Quads, right-handed outward normals.
    index = (
        (1, 5, 6, 2),  # +x
        (0, 3, 7, 4),  # -x
        (3, 2, 6, 7),  # +y
        (0, 4, 5, 1),  # -y
        (4, 7, 6, 5),  # +z
        (0, 1, 2, 3),  # -z
    )
    faces = [{"verts": p[list(idx)], "interior": False} for idx in index]
    return Solid(_orient_outward(faces))


def _icosahedron():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    return verts, faces


def sphere_solid(center=(0.0, 0.0, 0.0), radius: float = 0.5, subdiv: int = 1) -> Solid:
    """Convex icosphere. Its volume is the solid that fracture conserves, not the analytic ball."""
    verts, faces = _icosahedron()
    for _ in range(int(subdiv)):
        cache = {}
        nxt = verts.tolist()
        new_faces = []

        def mid(i, j):
            key = (i, j) if i < j else (j, i)
            if key in cache:
                return cache[key]
            p = _unit(verts[i] + verts[j])
            cache[key] = len(nxt)
            nxt.append(p)
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        verts = np.asarray(nxt, dtype=np.float64)
        faces = new_faces
    c = _as_f64(center)
    r = float(radius)
    verts = c + verts * r
    built = [{"verts": verts[list(f)], "interior": False} for f in faces]
    return Solid(_orient_outward(built))


def _unique_points(points: np.ndarray, tol: float) -> np.ndarray:
    pts = _as_f64(points).reshape(-1, 3)
    if len(pts) == 0:
        return pts
    order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
    pts = pts[order]
    keep = [pts[0]]
    for p in pts[1:]:
        if _norm(p - keep[-1]) > tol:
            keep.append(p)
    return np.asarray(keep, dtype=np.float64)


def convex_hull(points, tol: float = 1e-8) -> Solid:
    """Incremental hull. Face winding is forced outward against the vertex centroid."""
    pts = _unique_points(points, tol)
    if len(pts) < 4:
        raise ValueError("convex hull needs at least 4 points")
    i0 = 0
    i1 = int(np.argmax(np.linalg.norm(pts - pts[i0], axis=1)))
    axis = _unit(pts[i1] - pts[i0])
    i2 = int(np.argmax(np.linalg.norm(np.cross(pts - pts[i0], axis), axis=1)))
    n0 = np.cross(pts[i1] - pts[i0], pts[i2] - pts[i0])
    n0 = _unit(n0)
    i3 = int(np.argmax(np.abs((pts - pts[i0]) @ n0)))
    if abs(float((pts[i3] - pts[i0]) @ n0)) <= tol:
        raise ValueError("points are coplanar")
    seeds = (i0, i1, i2, i3)
    faces = [(i0, i1, i2), (i0, i2, i3), (i0, i3, i1), (i1, i2, i3)]
    used = set(seeds)

    def orient(face_list, ids):
        center = pts[list(ids)].mean(axis=0)
        fixed = []
        for a, b, c in face_list:
            nrm = np.cross(pts[b] - pts[a], pts[c] - pts[a])
            if float(np.dot(nrm, pts[a] - center)) < 0.0:
                fixed.append((a, c, b))
            else:
                fixed.append((a, b, c))
        return fixed

    faces = orient(faces, used)
    for pi in range(len(pts)):
        if pi in used:
            continue
        p = pts[pi]
        visible = []
        for fi, (a, b, c) in enumerate(faces):
            nrm = np.cross(pts[b] - pts[a], pts[c] - pts[a])
            if float(np.dot(nrm, p - pts[a])) > tol * 10:
                visible.append(fi)
        if not visible:
            continue
        vis = set(visible)
        edges: dict = {}
        for fi in visible:
            a, b, c = faces[fi]
            for e in ((a, b), (b, c), (c, a)):
                edges.setdefault(tuple(sorted(e)), []).append(e)
        horizon = [dirs[0] for dirs in edges.values() if len(dirs) == 1]
        kept = [f for i, f in enumerate(faces) if i not in vis]
        kept.extend((a, b, pi) for a, b in horizon)
        used.add(pi)
        faces = orient(kept, used)
    built = [{"verts": pts[list(f)], "interior": False} for f in faces]
    return Solid(_orient_outward(built))


def solid_from_mesh(vertices, faces) -> Solid:
    """Convex mesh as given; a concave mesh becomes its hull (see the module note)."""
    verts = _as_f64(vertices).reshape(-1, 3)
    tris = []
    for face in faces:
        idx = list(face)
        if len(idx) < 3:
            continue
        for i in range(1, len(idx) - 1):
            tris.append(verts[[idx[0], idx[i], idx[i + 1]]])
    if len(tris) < 4:
        raise ValueError("mesh needs at least 4 triangles")
    built = [{"verts": t, "interior": False} for t in tris]
    solid = Solid(_orient_outward(built))
    # Convex when every vertex is inside every face plane.
    planes = solid.planes()
    sd = verts @ planes[:, :3].T - planes[:, 3]
    extent = _norm(solid.bounds()[1] - solid.bounds()[0])
    if np.max(sd) <= 1e-6 * max(extent, 1.0):
        return solid
    return convex_hull(verts)


def _clip_polygon(poly: np.ndarray, normal: np.ndarray, offset: float, eps: float) -> np.ndarray:
    if len(poly) == 0:
        return poly
    out = []
    for i, curr in enumerate(poly):
        prev = poly[i - 1]
        dp = float(np.dot(normal, prev))
        dc = float(np.dot(normal, curr))
        prev_in = dp <= offset + eps
        curr_in = dc <= offset + eps
        if curr_in:
            if not prev_in:
                denom = dp - dc
                t = 0.0 if abs(denom) <= _EPS else (dp - offset) / denom
                out.append(prev + t * (curr - prev))
            out.append(curr)
        elif prev_in:
            denom = dp - dc
            t = 0.0 if abs(denom) <= _EPS else (dp - offset) / denom
            out.append(prev + t * (curr - prev))
    if not out:
        return np.zeros((0, 3))
    return np.asarray(out, dtype=np.float64)


def _poly_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    n = np.zeros(3, dtype=np.float64)
    for i in range(len(poly)):
        n += np.cross(poly[i], poly[(i + 1) % len(poly)])
    return 0.5 * _norm(n)


def _cap_polygon(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Convex hull of the cut points, wound so the normal matches `normal` (outward)."""
    n = _unit(normal)
    if len(points) < 3 or _norm(n) <= _EPS:
        return np.zeros((0, 3))
    tangent = np.array([1.0, 0.0, 0.0]) if abs(float(n[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _unit(np.cross(tangent, n))
    y = np.cross(n, x)
    origin = points.mean(axis=0)
    uv = np.column_stack(((points - origin) @ x, (points - origin) @ y))
    order = np.lexsort((uv[:, 1], uv[:, 0]))
    uv = uv[order]

    def cross2(a, b) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def chain(seq, turn):
        hull = []
        for p in seq:
            while len(hull) >= 2 and cross2(hull[-1] - hull[-2], p - hull[-2]) * turn <= 1e-16:
                hull.pop()
            hull.append(p)
        return hull

    ring = chain(uv, 1.0)[:-1] + chain(uv[::-1], 1.0)[:-1]
    if len(ring) < 3:
        return np.zeros((0, 3))
    uv_h = np.asarray(ring, dtype=np.float64)
    return origin + uv_h[:, 0:1] * x + uv_h[:, 1:2] * y


def clip_solid(solid: Solid, normal, offset: float, interior: bool = True, eps: float | None = None) -> Solid:
    """Keep the half-space n · x <= offset."""
    n = _unit(_as_f64(normal))
    if _norm(n) <= _EPS:
        return solid.copy()
    lo, hi = solid.bounds()
    extent = max(_norm(hi - lo), 1e-6)
    eps = 1e-10 * extent if eps is None else float(eps)
    kept = []
    cap_pts = []
    # Wide enough to catch an intersection that float rounding nudged off the plane, and tight
    # enough that a vertex merely near the cut is not swallowed into the cap.
    on_eps = max(eps * 1e4, 1e-9 * extent)
    for face in solid.faces:
        poly = _clip_polygon(face["verts"], n, float(offset), eps)
        if len(poly) == 0:
            continue
        dots = poly @ n - float(offset)
        cap_pts.extend(poly[np.abs(dots) <= on_eps])
        if len(poly) < 3 or _poly_area(poly) <= eps * extent:
            continue
        kept.append({"verts": poly, "interior": bool(face.get("interior", False))})
    if cap_pts:
        cap = _unique_points(np.asarray(cap_pts, dtype=np.float64), on_eps)
        cap = cap - n * ((cap @ n) - float(offset))[:, None]
        ordered = _cap_polygon(cap, n)
        if len(ordered) >= 3 and _poly_area(ordered) > eps * extent:
            kept.append({"verts": ordered, "interior": bool(interior)})
    return Solid(kept)


def _fully_inside(solid: Solid, normal: np.ndarray, offset: float, eps: float) -> bool:
    v = solid.vertices()
    if len(v) == 0:
        return True
    return bool(np.all(v @ normal <= offset + eps))


@dataclass
class Piece:
    vertices: np.ndarray          # world, at the rest pose
    triangles: np.ndarray         # (T, 3, 3) world
    interior: np.ndarray          # (T,) True on cut faces
    volume: float
    com: np.ndarray
    inertia: np.ndarray           # about COM, world axes of the rest pose
    mass: float
    local_vertices: np.ndarray    # relative to COM
    local_planes: np.ndarray      # (F, 4) in the COM frame
    local_aabb_center: np.ndarray
    local_half: np.ndarray
    radius: float

    def sdf(self, points) -> np.ndarray:
        pts = _as_f64(points).reshape(-1, 3) - self.com
        planes = self.local_planes
        if len(planes) == 0:
            return np.full(len(pts), 1e4)
        return np.max(pts @ planes[:, :3].T - planes[:, 3], axis=1)


def _finish_piece(solid: Solid, density: float) -> Piece | None:
    tris, interior = _triangles_of(solid.faces)
    if len(tris) < 4:
        return None
    volume, com, inertia, mass = mass_properties(tris, density)
    if volume <= 1e-10 or mass <= 0.0:
        return None
    # Drop a near-singular inertia so the rigid solver can invert it.
    inertia = inertia + np.eye(3) * (1e-8 * mass)
    local_verts = []
    for face in solid.faces:
        local_verts.append(face["verts"] - com)
    stacked = np.vstack(local_verts)
    # Unique-ish local vertices for collision.
    extent = max(_norm(stacked.max(0) - stacked.min(0)), 1e-6)
    uniq = _unique_points(stacked, 1e-8 * extent)
    planes = []
    for face in solid.faces:
        n = _face_normal(face["verts"])
        if _norm(n) <= _EPS:
            continue
        d = float(np.dot(n, face["verts"][0] - com))
        planes.append((n[0], n[1], n[2], d))
    local_planes = np.asarray(planes, dtype=np.float64) if planes else np.zeros((0, 4))
    mn, mx = uniq.min(0), uniq.max(0)
    center = 0.5 * (mn + mx)
    half = np.maximum(0.5 * (mx - mn), 1e-5)
    radius = float(np.linalg.norm(uniq, axis=1).max()) if len(uniq) else 0.0
    return Piece(
        vertices=stacked, triangles=tris, interior=interior, volume=float(volume), com=com,
        inertia=inertia, mass=float(mass), local_vertices=uniq, local_planes=local_planes,
        local_aabb_center=center, local_half=half, radius=radius,
    )


def _seeds(solid: Solid, count: int, rng: np.random.Generator, pattern: str, impact, sigma: float) -> np.ndarray:
    lo, hi = solid.bounds()
    span = np.maximum(hi - lo, 1e-6)
    impact = solid.vertices().mean(axis=0) if impact is None else _as_f64(impact)
    sigma = float(sigma) if sigma and sigma > 0 else float(np.max(span) * 0.25)
    found = []
    tries = 0
    limit = max(count * 400, 2000)
    while len(found) < count and tries < limit:
        tries += 1
        if pattern == "clustered":
            p = rng.normal(loc=impact, scale=sigma, size=3)
        elif pattern == "radial":
            direction = rng.normal(size=3)
            direction = _unit(direction)
            radius = float(rng.random() ** (1.0 / 3.0)) * sigma
            p = impact + direction * radius
        else:
            p = lo + rng.random(3) * span
        if bool(solid.contains(p, tol=1e-8)[0]):
            found.append(p)
    if len(found) < count:
        # Fill from a deterministic jittered lattice so a tight cluster still returns `count` pieces
        # when the gaussian misses the solid. Lattice points are inside by rejection on the grid.
        grid = int(np.ceil(count ** (1.0 / 3.0))) + 2
        for iz in range(grid):
            for iy in range(grid):
                for ix in range(grid):
                    u = (np.array([ix, iy, iz], dtype=np.float64) + 0.5) / grid
                    p = lo + u * span
                    if bool(solid.contains(p, tol=-1e-4)[0]):
                        found.append(p)
                    if len(found) >= count:
                        break
                if len(found) >= count:
                    break
            if len(found) >= count:
                break
    if len(found) < count:
        raise RuntimeError(f"only placed {len(found)} of {count} seeds inside the solid")
    seeds = np.asarray(found[:count], dtype=np.float64)
    order = np.lexsort((seeds[:, 2], seeds[:, 1], seeds[:, 0]))
    return seeds[order]


def fracture(solid: Solid, count: int, seed: int = 0, pattern: str = "uniform",
             impact=None, sigma: float = 0.0, density: float = 1.0) -> list[Piece]:
    """Clip `solid` into `count` Voronoi pieces. `pattern` is uniform, clustered, or radial."""
    count = int(count)
    if count < 1:
        raise ValueError("count must be positive")
    if pattern not in ("uniform", "clustered", "radial"):
        raise ValueError(pattern)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    seeds = _seeds(solid, count, rng, pattern, impact, sigma)
    lo, hi = solid.bounds()
    extent = max(_norm(hi - lo), 1e-6)
    eps = 1e-10 * extent
    pieces: list[Piece] = []
    for i, origin in enumerate(seeds):
        piece = solid.copy()
        delta = seeds - origin
        dist = np.linalg.norm(delta, axis=1)
        order = np.argsort(dist, kind="stable")
        for j in order:
            if i == j:
                continue
            normal = seeds[j] - origin
            length = _norm(normal)
            if length <= eps:
                continue
            mid = 0.5 * (origin + seeds[j])
            offset = float(np.dot(_unit(normal), mid))
            if _fully_inside(piece, _unit(normal), offset, eps * 10):
                continue
            piece = clip_solid(piece, normal, offset, interior=True, eps=eps)
            if not piece.faces:
                break
        finished = _finish_piece(piece, density)
        if finished is not None:
            pieces.append(finished)
    return pieces


def neighboring_pairs(pieces: list[Piece], tol: float | None = None) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Adjacent pieces: the midpoint of the centres lies on both surfaces.

    Returns (i, j, point, normal from i toward j). Used as the rest pose of a glue constraint.
    """
    if len(pieces) < 2:
        return []
    coms = np.stack([p.com for p in pieces])
    radii = np.array([p.radius for p in pieces])
    extent = float(np.linalg.norm(coms.max(0) - coms.min(0))) + float(radii.max())
    # Centres of mass sit a little off the shared face, so the midpoint's distance to each
    # surface is a few percent of the solid, not zero.
    tol = 0.04 * max(extent, 1e-3) if tol is None else float(tol)
    pairs = []
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            if _norm(coms[j] - coms[i]) > radii[i] + radii[j] + tol:
                continue
            mid = 0.5 * (coms[i] + coms[j])
            if abs(float(pieces[i].sdf(mid)[0])) <= tol and abs(float(pieces[j].sdf(mid)[0])) <= tol:
                normal = _unit(coms[j] - coms[i])
                pairs.append((i, j, mid, normal))
    return pairs


def edge_counts(piece: Piece, tol: float | None = None) -> dict:
    """Undirected edge multiplicities after welding vertices. A closed mesh is all 2s."""
    tris = piece.triangles
    if len(tris) == 0:
        return {}
    extent = max(_norm(piece.local_half) * 2, 1e-6)
    tol = 1e-7 * extent if tol is None else float(tol)
    counts: dict = {}
    for tri in tris:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            qa = tuple(np.round(tri[a] / tol).astype(np.int64))
            qb = tuple(np.round(tri[b] / tol).astype(np.int64))
            if qa == qb:
                continue
            key = (qa, qb) if qa <= qb else (qb, qa)
            counts[key] = counts.get(key, 0) + 1
    return counts
