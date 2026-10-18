"""Deterministic rigid bodies: gravity, a ground plane, convex contacts, glue, sleep.

NumPy only, so the same step runs in unit tests and inside TouchDesigner. Bullet was not used;
see AGENTS.md. State that is cached is float32, because that is the EXR the baker writes: each
exported step ends in `quantize()` and the next step reads those values back.

Contact slop is the distance at which a resting vertex is already "on" the surface. The solver
corrects anything deeper. A stack tolerance wider than the slop is that residual, not a filter
on the samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fracture import Piece, box_solid, mass_properties

_EPS = 1e-12


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n <= _EPS:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    wq = np.array([omega[0], omega[1], omega[2], 0.0])
    return quat_normalize(q + 0.5 * float(dt) * quat_mul(wq, q))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= _EPS:
        return np.zeros(3)
    return v / n


@dataclass
class Body:
    pos: np.ndarray
    quat: np.ndarray
    lin: np.ndarray
    ang: np.ndarray
    mass: float
    inv_mass: float
    inertia: np.ndarray
    inv_inertia: np.ndarray
    vertices: np.ndarray
    planes: np.ndarray
    radius: float
    kind: str = "dynamic"          # dynamic, kinematic, static
    asleep: bool = False
    sleep_steps: int = 0
    aabb_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    aabb_max: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def movable(self) -> bool:
        return self.kind == "dynamic" and not self.asleep and self.inv_mass > 0.0


@dataclass
class Glue:
    a: int
    b: int
    local_a: np.ndarray
    local_b: np.ndarray
    normal: np.ndarray
    threshold: float
    broken: bool = False
    last_impulse: float = 0.0


@dataclass
class _Contact:
    a: int
    b: int          # -1 is the ground
    point: np.ndarray
    normal: np.ndarray
    penetration: float
    acc_n: float = 0.0
    acc_t: np.ndarray = field(default_factory=lambda: np.zeros(2))


class World:
    def __init__(self, gravity=(0.0, -9.81, 0.0), ground_y: float = 0.0, iterations: int = 24,
                 position_iters: int = 8, slop: float = 2e-4, friction: float = 0.8,
                 restitution: float = 0.0, sleep_lin: float = 0.05, sleep_ang: float = 0.08,
                 sleep_after: int = 20, mesh_contacts: bool = True):
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.ground_y = float(ground_y)
        self.iterations = int(iterations)
        self.position_iters = int(position_iters)
        self.slop = float(slop)
        self.friction = float(friction)
        self.restitution = float(restitution)
        self.sleep_lin = float(sleep_lin)
        self.sleep_ang = float(sleep_ang)
        self.sleep_after = int(sleep_after)
        # Mesh contacts are the accurate path (the unit tests). The live wall uses spheres:
        # a hundred convex pieces, each with its own planes, does not fit in a 30 fps frame.
        self.mesh_contacts = bool(mesh_contacts)
        self.bodies: list[Body] = []
        self.glues: list[Glue] = []

    # -- construction -----------------------------------------------------------------------------------------
    def add_box(self, center, half, mass: float = 1.0, velocity=(0, 0, 0), kind: str = "dynamic") -> int:
        solid = box_solid((0.0, 0.0, 0.0), half)
        tris = []
        for face in solid.faces:
            v = face["verts"]
            tris.append((v[0], v[1], v[2]))
            tris.append((v[0], v[2], v[3]))
        _, _, inertia, _ = mass_properties(np.asarray(tris), density=1.0)
        # mass_properties used density 1, so scale the geometric inertia up to `mass`.
        volume = solid.volume()
        inertia = inertia * (float(mass) / volume)
        verts = np.unique(np.round(solid.vertices(), 8), axis=0)
        return self._add(np.asarray(center, np.float64), float(mass), inertia, verts, solid.planes(),
                         float(np.linalg.norm(half)), kind, np.asarray(velocity, np.float64))

    def add_sphere(self, center, radius: float, mass: float = 1.0, velocity=(0, 0, 0),
                   kind: str = "dynamic") -> int:
        r = float(radius)
        inertia = np.eye(3) * (0.4 * float(mass) * r * r)
        return self._add(np.asarray(center, np.float64), float(mass), inertia, np.zeros((0, 3)),
                         np.zeros((0, 4)), r, kind, np.asarray(velocity, np.float64))

    def add_piece(self, piece: Piece, kind: str = "dynamic", velocity=(0, 0, 0)) -> int:
        return self._add(piece.com.copy(), float(piece.mass), piece.inertia.copy(),
                         piece.local_vertices.copy(), piece.local_planes.copy(), float(piece.radius),
                         kind, np.asarray(velocity, np.float64))

    def _add(self, pos, mass, inertia, vertices, planes, radius, kind, velocity) -> int:
        inertia = np.array(inertia, dtype=np.float64)
        inertia = inertia + np.eye(3) * (1e-8 * max(mass, 1e-6))
        inv_m = 0.0 if kind != "dynamic" or mass <= 0 else 1.0 / mass
        inv_i = np.zeros((3, 3)) if inv_m == 0 else np.linalg.inv(inertia)
        body = Body(
            pos=np.asarray(pos, np.float64).copy(),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            lin=np.asarray(velocity, np.float64).astype(np.float64).copy(),
            ang=np.zeros(3),
            mass=float(mass), inv_mass=float(inv_m), inertia=inertia, inv_inertia=inv_i,
            vertices=np.asarray(vertices, np.float64).reshape(-1, 3),
            planes=np.asarray(planes, np.float64).reshape(-1, 4),
            radius=float(radius), kind=kind,
        )
        self.bodies.append(body)
        self._refresh_aabb(body)
        return len(self.bodies) - 1

    def add_glue(self, a: int, b: int, point, normal, threshold: float) -> int:
        pa = self._world_to_local(self.bodies[a], np.asarray(point, np.float64))
        pb = self._world_to_local(self.bodies[b], np.asarray(point, np.float64))
        self.glues.append(Glue(a, b, pa, pb, _unit(np.asarray(normal, np.float64)), float(threshold)))
        return len(self.glues) - 1

    def glue_broken(self) -> int:
        return sum(1 for g in self.glues if g.broken)

    def break_glue(self, index: int | None = None) -> None:
        if index is None:
            for glue in self.glues:
                glue.broken = True
        else:
            self.glues[index].broken = True
        for body in self.bodies:
            body.asleep = False
            body.sleep_steps = 0

    def explosion(self, point, radius: float, strength: float) -> None:
        origin = np.asarray(point, np.float64)
        radius = float(radius)
        for body in self.bodies:
            if body.inv_mass <= 0.0:
                continue
            delta = body.pos - origin
            dist = float(np.linalg.norm(delta))
            if dist >= radius or dist <= 1e-8:
                continue
            body.asleep = False
            body.sleep_steps = 0
            body.lin = body.lin + _unit(delta) * (float(strength) * (1.0 - dist / radius) * body.inv_mass)

    # -- transforms -------------------------------------------------------------------------------------------
    def _world_to_local(self, body: Body, point: np.ndarray) -> np.ndarray:
        return quat_matrix(body.quat).T @ (point - body.pos)

    def _rotation(self, body: Body) -> np.ndarray:
        cached = getattr(body, "_R", None)
        return quat_matrix(body.quat) if cached is None else cached

    def _local_to_world(self, body: Body, point: np.ndarray) -> np.ndarray:
        return self._rotation(body) @ point + body.pos

    def _sample_points(self, body: Body) -> np.ndarray:
        """Vertices plus one point per face. Face centres catch a stack whose corners only graze the edges."""
        verts = self.world_vertices(body)
        if len(body.planes) == 0 or len(body.vertices) == 0:
            return verts
        r = self._rotation(body)
        extra = []
        for plane in body.planes:
            on = body.vertices[np.abs(body.vertices @ plane[:3] - plane[3]) <= 1e-5]
            if len(on):
                extra.append(r @ on.mean(axis=0) + body.pos)
        if not extra:
            return verts
        return np.vstack([verts, np.asarray(extra)])

    def world_vertices(self, body: Body) -> np.ndarray:
        if len(body.vertices) == 0:
            return body.vertices
        return body.vertices @ self._rotation(body).T + body.pos

    def world_planes(self, body: Body) -> np.ndarray:
        if len(body.planes) == 0:
            return body.planes
        n = body.planes[:, :3] @ self._rotation(body).T
        d = body.planes[:, 3] + n @ body.pos
        return np.column_stack([n, d])

    def _refresh_aabb(self, body: Body) -> None:
        if body.kind == "static":
            return
        if not self.mesh_contacts or len(body.vertices) == 0:
            body.aabb_min = body.pos - body.radius
            body.aabb_max = body.pos + body.radius
            return
        v = self.world_vertices(body)
        body.aabb_min = v.min(axis=0)
        body.aabb_max = v.max(axis=0)

    def _point_velocity(self, body: Body, point: np.ndarray) -> np.ndarray:
        return body.lin + np.cross(body.ang, point - body.pos)

    def _inv_inertia_world(self, body: Body) -> np.ndarray:
        cached = getattr(body, "_invIw", None)
        if cached is not None:
            return cached
        if body.inv_mass <= 0.0:
            return np.zeros((3, 3))
        r = self._rotation(body)
        return r @ body.inv_inertia @ r.T

    def _cache_frames(self) -> None:
        """Rotation does not change during the velocity iterations, so build it once per step."""
        for body in self.bodies:
            r = quat_matrix(body.quat)
            body._R = r
            if body.inv_mass > 0.0:
                body._invIw = r @ body.inv_inertia @ r.T
            else:
                body._invIw = np.zeros((3, 3))

    def _drop_frames(self) -> None:
        for body in self.bodies:
            body._R = None
            body._invIw = None

    def _apply_impulse(self, body: Body, impulse: np.ndarray, point: np.ndarray) -> None:
        if body.inv_mass <= 0.0 or body.asleep:
            return
        body.lin = body.lin + impulse * body.inv_mass
        r = point - body.pos
        body.ang = body.ang + self._inv_inertia_world(body) @ np.cross(r, impulse)

    def _effective(self, a: Body, b: Body | None, point: np.ndarray, direction: np.ndarray) -> float:
        kn = a.inv_mass
        if a.inv_mass > 0.0:
            ra = np.cross(point - a.pos, direction)
            kn += float(np.dot(ra, self._inv_inertia_world(a) @ ra))
        if b is not None and b.inv_mass > 0.0:
            kn += b.inv_mass
            rb = np.cross(point - b.pos, direction)
            kn += float(np.dot(rb, self._inv_inertia_world(b) @ rb))
        return max(kn, 1e-10)

    # -- contacts ---------------------------------------------------------------------------------------------
    def _cluster_ids(self) -> list[int]:
        parent = list(range(len(self.bodies)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for glue in self.glues:
            if glue.broken:
                continue
            a, b = find(glue.a), find(glue.b)
            if a != b:
                parent[b] = a
        return [find(i) for i in range(len(self.bodies))]

    def _glue_index(self) -> set[tuple[int, int]]:
        cached = getattr(self, "_glue_pairs", None)
        if cached is not None:
            return cached
        pairs = set()
        for glue in self.glues:
            if glue.broken:
                continue
            a, b = (glue.a, glue.b) if glue.a < glue.b else (glue.b, glue.a)
            pairs.add((a, b))
        self._glue_pairs = pairs
        return pairs

    def _glued(self, i: int, j: int) -> bool:
        if i > j:
            i, j = j, i
        return (i, j) in self._glue_index()

    def _sdf_planes(self, points: np.ndarray, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sd = points @ planes[:, :3].T - planes[:, 3]
        which = np.argmax(sd, axis=1)
        return sd[np.arange(len(points)), which], which

    def _pair_contacts(self, i: int, j: int, contacts: list[_Contact]) -> None:
        a, b = self.bodies[i], self.bodies[j]
        if a.kind == "static" and b.kind == "static":
            return
        if a.aabb_min[0] > b.aabb_max[0] or b.aabb_min[0] > a.aabb_max[0]:
            return
        if a.aabb_min[1] > b.aabb_max[1] or b.aabb_min[1] > a.aabb_max[1]:
            return
        if a.aabb_min[2] > b.aabb_max[2] or b.aabb_min[2] > a.aabb_max[2]:
            return
        # One support per pair. A handful of corner contacts with friction torque the stack over.
        found: list[_Contact] = []
        self._verts_into(i, j, found)
        self._verts_into(j, i, found)
        if found:
            a, b = self.bodies[i], self.bodies[j]
            axis = b.pos - a.pos
            axis_n = axis / max(float(np.linalg.norm(axis)), 1e-8)

            def rank(c: _Contact):
                rel = c.point - a.pos
                off = rel - axis_n * float(np.dot(rel, axis_n))
                # Deepest first; ties go to the point on the line between the centres.
                return (round(c.penetration, 6), -float(np.dot(off, off)))

            contacts.append(max(found, key=rank))

    def _verts_into(self, src: int, dst: int, contacts: list[_Contact]) -> None:
        a, b = self.bodies[src], self.bodies[dst]
        planes = b._wp if getattr(b, "_wp", None) is not None else self.world_planes(b)
        if len(a.vertices) == 0:
            if len(planes) == 0:
                delta = a.pos - b.pos
                dist = float(np.linalg.norm(delta))
                pen = a.radius + b.radius - dist
                if pen > self.slop and dist > 1e-8:
                    n = delta / dist
                    contacts.append(_Contact(src, dst, b.pos + n * b.radius, n, pen))
                return
            sd, which = self._sdf_planes(a.pos.reshape(1, 3), planes)
            pen = a.radius - float(sd[0])  # positive when the sphere overlaps the solid
            # sd is negative inside. Overlap when sd < radius.
            if float(sd[0]) < a.radius - self.slop:
                n = planes[int(which[0]), :3]
                contacts.append(_Contact(src, dst, a.pos - n * a.radius, n, a.radius - float(sd[0])))
            return
        verts = a._samples if getattr(a, "_samples", None) is not None else self._sample_points(a)
        if len(planes) == 0:
            # Destination is a sphere.
            delta = verts - b.pos
            dist = np.linalg.norm(delta, axis=1)
            for k in range(len(verts)):
                pen = b.radius - float(dist[k])
                if pen > self.slop and dist[k] > 1e-8:
                    n = -delta[k] / dist[k]
                    contacts.append(_Contact(src, dst, verts[k], n, pen))
            return
        sd_all = verts @ planes[:, :3].T - planes[:, 3]
        sd = sd_all.max(axis=1)
        deep = np.where(sd < self.slop)[0]
        if len(deep) > 8:
            deep = deep[np.argsort(sd[deep])[:8]]
        # A corner sits on several planes at once. The separating direction is the one toward
        # the other body's centre; a side plane chosen by a tie lets a stack fall through.
        sep = a.pos - b.pos
        sep_len = float(np.linalg.norm(sep))
        for k in deep:
            near = np.where(sd_all[k] >= float(sd[k]) - 1e-5)[0]
            align = planes[near, :3] @ sep
            pick = int(near[int(np.argmax(align))])
            n = planes[pick, :3]
            pen = max(float(-sd[k]), 0.0)
            # A corner that merely lies on a side face is not holding the other body up.
            if pen <= self.slop and float(np.dot(n, sep)) < 0.5 * max(sep_len, 1e-6):
                continue
            contacts.append(_Contact(src, dst, verts[int(k)], n, pen))

    def _ground_contacts(self, i: int, contacts: list[_Contact]) -> None:
        body = self.bodies[i]
        if body.kind == "static":
            return
        n = np.array([0.0, 1.0, 0.0])
        if len(body.vertices) == 0:
            pen = self.ground_y - (body.pos[1] - body.radius)
            if pen > 0.0 or (body.pos[1] - body.radius) <= self.ground_y + self.slop:
                point = np.array([body.pos[0], self.ground_y, body.pos[2]])
                contacts.append(_Contact(i, -1, point, n, max(pen, 0.0)))
            return
        verts = self.world_vertices(body)
        for v in verts:
            if v[1] <= self.ground_y + self.slop:
                contacts.append(_Contact(i, -1, v.copy(), n, max(self.ground_y - float(v[1]), 0.0)))

    def _detect_spheres(self) -> list[_Contact]:
        """Bounding-sphere broadphase. Same impulse solver, without per-piece plane transforms."""
        contacts: list[_Contact] = []
        up = np.array([0.0, 1.0, 0.0])
        for i, body in enumerate(self.bodies):
            body.aabb_min = body.pos - body.radius
            body.aabb_max = body.pos + body.radius
            body._samples = None
            body._wp = None
            if body.kind == "static":
                continue
            # The bounding sphere of a shard sticks out past the mesh, so a sphere-ground test
            # hits pieces that are still in the air and the glue then tears from the bottom up.
            quat = body.quat
            upright = abs(float(quat[3])) > 0.999 and abs(float(quat[0])) + abs(float(quat[1])) + abs(float(quat[2])) < 1e-3
            if len(body.vertices) == 0:
                bottom = float(body.pos[1] - body.radius)
            elif upright:
                if not hasattr(body, "_local_miny"):
                    body._local_miny = float(body.vertices[:, 1].min())
                bottom = float(body.pos[1] + body._local_miny)
            else:
                bottom = float((body.vertices @ quat_matrix(quat).T + body.pos)[:, 1].min())
            pen = self.ground_y - bottom
            if pen > 0.0 or bottom <= self.ground_y + self.slop:
                contacts.append(_Contact(
                    i, -1, np.array([body.pos[0], self.ground_y, body.pos[2]]), up, max(pen, 0.0),
                ))
        # Pieces of one glued cluster overlap as spheres even when their faces only touch.
        # Colliding them shatters the wall. Only separate clusters (and loose debris) contact.
        cluster = self._cluster_ids()
        reach = max((b.radius for b in self.bodies), default=0.5)
        cell = max(reach * 2.0, 0.15)
        bins: dict[tuple[int, int, int], list[int]] = {}
        for i, body in enumerate(self.bodies):
            key = tuple(int(v) for v in np.floor(body.pos / cell))
            bins.setdefault(key, []).append(i)
        # Cell size is two radii, so an overlapping pair can sit in neighbouring cells.
        seen = set()
        for key, group in bins.items():
            for i in group:
                for dz in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            other = bins.get((key[0] + dx, key[1] + dy, key[2] + dz))
                            if not other:
                                continue
                            for j in other:
                                if j <= i:
                                    continue
                                pair = (i, j)
                                if pair in seen or cluster[i] == cluster[j]:
                                    continue
                                seen.add(pair)
                                ba, bb = self.bodies[i], self.bodies[j]
                                if ba.kind == "static" and bb.kind == "static":
                                    continue
                                delta = bb.pos - ba.pos
                                dist = float(np.linalg.norm(delta))
                                # A shard's bounding sphere contains its neighbours, so a projectile
                                # would "hit" the whole wall. Against a mesh piece the test is the
                                # projectile radius only, the same test as the impact sweep.
                                if len(ba.vertices) == 0 and len(bb.vertices) > 0:
                                    limit = ba.radius
                                elif len(bb.vertices) == 0 and len(ba.vertices) > 0:
                                    limit = bb.radius
                                else:
                                    limit = ba.radius + bb.radius
                                if dist <= 1e-8 or dist + self.slop >= limit:
                                    continue
                                n = delta / dist
                                contacts.append(_Contact(i, j, ba.pos + n * ba.radius, n, limit - dist))
        return contacts

    def _detect(self) -> list[_Contact]:
        if not self.mesh_contacts:
            return self._detect_spheres()
        big = set()
        if self.glues:
            for members in self._clusters().values():
                if len(members) >= 2:
                    big.update(members)
        # A glued cluster is one rigid body. Only loose pieces, and the wall pieces near them, need a mesh.
        active = [i for i in range(len(self.bodies)) if i not in big]
        for i, body in enumerate(self.bodies):
            if i in big:
                body.aabb_min = body.pos - body.radius
                body.aabb_max = body.pos + body.radius
                body._samples = None
                body._wp = None
            else:
                body._samples = self._sample_points(body)
                body._wp = self.world_planes(body)
                if len(body._samples):
                    body.aabb_min = body._samples.min(axis=0)
                    body.aabb_max = body._samples.max(axis=0)
                else:
                    body.aabb_min = body.pos - body.radius
                    body.aabb_max = body.pos + body.radius
        for i in list(active):
            body = self.bodies[i]
            for j, other in enumerate(self.bodies):
                if j == i or j not in big:
                    continue
                if body.aabb_min[0] > other.aabb_max[0] or other.aabb_min[0] > body.aabb_max[0]:
                    continue
                if body.aabb_min[1] > other.aabb_max[1] or other.aabb_min[1] > body.aabb_max[1]:
                    continue
                if body.aabb_min[2] > other.aabb_max[2] or other.aabb_min[2] > body.aabb_max[2]:
                    continue
                if other._samples is None:
                    other._samples = self._sample_points(other)
                    other._wp = self.world_planes(other)
                    if len(other._samples):
                        other.aabb_min = other._samples.min(axis=0)
                        other.aabb_max = other._samples.max(axis=0)
                active.append(j)
        contacts: list[_Contact] = []
        seen_ground = set()
        for i in active:
            if i in seen_ground:
                continue
            seen_ground.add(i)
            self._ground_contacts(i, contacts)
        # Uniform grid. Cell size tracks the largest body so a pair shares a cell.
        reach = max((b.radius for b in self.bodies), default=0.5)
        cell = max(reach * 2.0, 0.15)
        bins: dict[tuple[int, int, int], list[int]] = {}
        for i, body in enumerate(self.bodies):
            lo = np.floor(body.aabb_min / cell).astype(int)
            hi = np.floor(body.aabb_max / cell).astype(int)
            # A blown-up body must not expand into millions of bins.
            if np.any(hi - lo > 6) or not np.isfinite(body.aabb_min).all():
                key = tuple(int(v) for v in np.floor(body.pos / cell))
                bins.setdefault(key, []).append(i)
                continue
            for z in range(int(lo[2]), int(hi[2]) + 1):
                for y in range(int(lo[1]), int(hi[1]) + 1):
                    for x in range(int(lo[0]), int(hi[0]) + 1):
                        bins.setdefault((x, y, z), []).append(i)
        seen = set()
        for group in bins.values():
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    i, j = group[a], group[b]
                    if i > j:
                        i, j = j, i
                    if (i, j) in seen or self._glued(i, j) or (i in big and j in big):
                        continue
                    # Debris does not mesh-collide with a still-glued cluster. The projectile is a
                    # sphere and still does, via the sweep above and this pair.
                    if (i in big) != (j in big):
                        loose = i if i not in big else j
                        if len(self.bodies[loose].vertices) > 0:
                            continue
                    seen.add((i, j))
                    self._pair_contacts(i, j, contacts)
        return contacts

    def _relative(self, c: _Contact) -> np.ndarray:
        a = self.bodies[c.a]
        va = self._point_velocity(a, c.point)
        if c.b < 0:
            return va
        return va - self._point_velocity(self.bodies[c.b], c.point)

    def _solve_contact(self, c: _Contact, bounce: bool, dt: float) -> None:
        a = self.bodies[c.a]
        b = self.bodies[c.b] if c.b >= 0 else None
        n = c.normal
        if a.asleep or (b is not None and b.asleep):
            rel = self._relative(c)
            vn = float(np.dot(rel, n))
            # A resting sleeper stays asleep. A real hit (inward speed, or a deep overlap) wakes it.
            if vn < -0.25 or c.penetration > self.slop * 8:
                for body in (a, b):
                    if body is not None and body.asleep:
                        body.asleep = False
                        body.sleep_steps = 0
            else:
                return
        rel = self._relative(c)
        vn = float(np.dot(rel, n))
        e = self.restitution if bounce and vn < -0.4 else 0.0
        # Positive along n separates a from b. A resting overlap has vn ≈ 0, so a small
        # Baumgarte term asks for a separating speed. Capped so the correction cannot
        # outrun the fall and pump energy.
        bias = min(0.4 * max(c.penetration - self.slop, 0.0) / max(dt, 1e-4), 0.5)
        kn = self._effective(a, b, c.point, n)
        j = -((1.0 + e) * vn - bias) / kn
        acc = c.acc_n
        c.acc_n = max(acc + j, 0.0)
        j = c.acc_n - acc
        if abs(j) > 0.0:
            impulse = n * j
            self._apply_impulse(a, impulse, c.point)
            if b is not None:
                self._apply_impulse(b, -impulse, c.point)
            if c.b >= 0:
                self._break_from_impulse(c.a, abs(j))
                self._break_from_impulse(c.b, abs(j))
        # Friction in the tangent plane. Two axes so a resting box does not creep.
        rel = self._relative(c)
        tangent = rel - n * float(np.dot(rel, n))
        if float(np.linalg.norm(tangent)) < 1e-10:
            helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            t1 = _unit(np.cross(n, helper))
        else:
            t1 = _unit(tangent)
        t2 = _unit(np.cross(n, t1))
        limit = self.friction * c.acc_n
        for axis, ti in ((0, t1), (1, t2)):
            rel = self._relative(c)
            vt = float(np.dot(rel, ti))
            kt = self._effective(a, b, c.point, ti)
            jt = -vt / kt
            old = float(c.acc_t[axis])
            c.acc_t[axis] = float(np.clip(old + jt, -limit, limit))
            applied = c.acc_t[axis] - old
            if abs(applied) > 0.0:
                # Friction at the contact point torques a flush stack from solver noise until it
                # walks. The tangential impulse is applied at each centre of mass; the normal
                # impulse stays at the contact, so an off-centre hit still tips.
                impulse = ti * applied
                self._apply_impulse(a, impulse, a.pos)
                if b is not None:
                    self._apply_impulse(b, -impulse, b.pos)

    def _solve_glue(self, glue: Glue, dt: float) -> None:
        if glue.broken:
            return
        a, b = self.bodies[glue.a], self.bodies[glue.b]
        pa = self._local_to_world(a, glue.local_a)
        pb = self._local_to_world(b, glue.local_b)
        # Split the correction across the glue normal and the two tangents so a crack is the
        # normal impulse, which is what the threshold is compared to.
        n = _unit(self._rotation(a) @ glue.normal)
        helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axes = (n, _unit(np.cross(n, helper)), _unit(np.cross(n, _unit(np.cross(n, helper)))))
        point = 0.5 * (pa + pb)
        beta = 0.2 / max(dt, 1e-4)
        normal_impulse = 0.0
        total = 0.0
        for axis_i, axis in enumerate(axes):
            if float(np.linalg.norm(axis)) <= _EPS:
                continue
            rel = self._point_velocity(a, pa) - self._point_velocity(b, pb)
            vn = float(np.dot(rel, axis))
            bias = beta * float(np.dot(pa - pb, axis))
            kn = self._effective(a, b, point, axis)
            j = -(vn + bias) / kn
            if axis_i == 0:
                normal_impulse += j
            total += abs(j)
            self._apply_impulse(a, axis * j, pa)
            self._apply_impulse(b, -axis * j, pb)
        glue.last_impulse = abs(normal_impulse)
        if abs(normal_impulse) > glue.threshold or total > glue.threshold * 4.0:
            glue.broken = True

    def _clusters(self) -> dict[int, list[int]]:
        parent = list(range(len(self.bodies)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for glue in self.glues:
            if glue.broken:
                continue
            ra, rb = find(glue.a), find(glue.b)
            if ra != rb:
                parent[rb] = ra
        groups: dict[int, list[int]] = {}
        for i in range(len(self.bodies)):
            groups.setdefault(find(i), []).append(i)
        return groups

    def _spread_cluster_velocity(self) -> None:
        if not self.glues:
            return
        for members in self._clusters().values():
            if len(members) < 2:
                continue
            if all(self.bodies[i].asleep or self.bodies[i].inv_mass <= 0.0 for i in members):
                continue
            vel = np.zeros(3)
            mass = 0.0
            for i in members:
                body = self.bodies[i]
                if body.inv_mass <= 0.0:
                    continue
                vel += body.lin * body.mass
                mass += body.mass
            if mass <= 0.0:
                continue
            vel /= mass
            lowest = 1e9
            for i in members:
                body = self.bodies[i]
                quat = body.quat
                upright = abs(float(quat[3])) > 0.999 and abs(float(quat[0])) + abs(float(quat[1])) + abs(float(quat[2])) < 1e-3
                if len(body.vertices) == 0:
                    lowest = min(lowest, float(body.pos[1] - body.radius))
                elif upright:
                    if not hasattr(body, "_local_miny"):
                        body._local_miny = float(body.vertices[:, 1].min())
                    lowest = min(lowest, float(body.pos[1] + body._local_miny))
                else:
                    lowest = min(lowest, float((body.vertices @ quat_matrix(quat).T + body.pos)[:, 1].min()))
            if lowest < self.ground_y:
                if vel[1] < 0.0:
                    vel[1] = 0.0
                lift = self.ground_y - lowest
            else:
                lift = 0.0
            for i in members:
                body = self.bodies[i]
                if body.inv_mass <= 0.0 or body.asleep:
                    continue
                body.lin = vel.copy()
                body.ang[:] = 0.0
                if lift:
                    body.pos[1] += lift

    def _impact_sweep(self, dt: float) -> None:
        """A fast projectile can step clear through a thin piece. Sweep its sphere against glued bodies."""
        if not self.glues:
            return
        glued = set()
        for a, b in self._glue_index():
            glued.add(a)
            glued.add(b)
        for i, body in enumerate(self.bodies):
            # Only analytic spheres (the projectile). Debris already has mesh contacts.
            if i in glued or body.kind == "static" or len(body.vertices) > 0:
                continue
            start = body.pos - body.lin * float(dt)
            end = body.pos
            hits = 0
            for j in sorted(glued):
                other = self.bodies[j]
                reach = body.radius + other.radius
                # Distance from the other centre to the segment travelled this step.
                seg = end - start
                denom = float(np.dot(seg, seg))
                t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(other.pos - start, seg) / denom, 0.0, 1.0))
                closest = start + seg * t
                # Bounding spheres of neighbouring shards overlap, so the reach stays near the
                # projectile itself. Otherwise the sweep frees the whole wall.
                if float(np.linalg.norm(other.pos - closest)) > body.radius:
                    continue
                if hits >= 1:
                    break
                hits += 1
                speed = float(np.linalg.norm(body.lin))
                self._break_incident(j, max(self.glues[0].threshold * 1.4, 1e-3))
                away = other.pos - closest
                away_n = _unit(away) if float(np.linalg.norm(away)) > 1e-8 else np.array([0.0, 1.0, 0.0])
                if other.inv_mass > 0.0:
                    other.lin = other.lin + away_n * (0.35 * speed)
                    other.asleep = False

    def _break_glues_from_pull(self) -> None:
        """A relative speed at the glue is an impulse of reduced_mass * speed. Above the threshold it breaks."""
        rotation = {}
        for glue in self.glues:
            if glue.broken:
                continue
            a, b = self.bodies[glue.a], self.bodies[glue.b]
            # A cluster shares one velocity, so most glues cannot be near the threshold.
            rel = a.lin - b.lin
            if float(np.dot(rel, rel)) < 1e-8:
                glue.last_impulse = 0.0
                continue
            cached = getattr(a, "_R", None)
            if cached is None:
                cached = rotation.get(glue.a)
                if cached is None:
                    cached = quat_matrix(a.quat)
                    rotation[glue.a] = cached
            n = _unit(cached @ glue.normal)
            rel = float(np.dot(a.lin - b.lin, n))
            reduced = a.mass if b.inv_mass <= 0 else (b.mass if a.inv_mass <= 0 else (a.mass * b.mass) / (a.mass + b.mass))
            impulse = abs(rel) * reduced
            glue.last_impulse = impulse
            if impulse > glue.threshold:
                glue.broken = True
                self._glue_pairs = None

    def _break_incident(self, index: int, impulse: float) -> None:
        """Break the glue constraints that touch `index`, without walking into the rest of the wall."""
        for glue in self.glues:
            if glue.broken:
                continue
            if glue.a != index and glue.b != index:
                continue
            if impulse > glue.threshold:
                glue.broken = True
                glue.last_impulse = impulse
                self._glue_pairs = None

    def _break_from_impulse(self, index: int, impulse: float) -> None:
        if impulse <= 0.0:
            return
        # Sphere contacts are a broadphase. Walking the impulse through every glue from each one
        # both shatters the wall and spends the whole frame in this loop. The pull test propagates
        # a crack as the freed piece actually moves away. Mesh contacts keep the walk.
        if not self.mesh_contacts:
            self._break_incident(index, impulse)
            return
        pending = [(index, impulse)]
        seen = set()
        while pending:
            body, remaining = pending.pop()
            if body in seen or remaining <= 0.0:
                continue
            seen.add(body)
            for glue in self.glues:
                if glue.broken:
                    continue
                if glue.a != body and glue.b != body:
                    continue
                if remaining > glue.threshold:
                    glue.broken = True
                    self._glue_pairs = None
                    glue.last_impulse = remaining
                    other = glue.b if glue.a == body else glue.a
                    pending.append((other, remaining * 0.5))

    def _positional(self, contacts: list[_Contact]) -> None:
        # One correction per pair. Applying every vertex of a face stacks the same push and
        # throws the body back out with its speed intact.
        best: dict[tuple, _Contact] = {}
        for c in contacts:
            if c.penetration <= self.slop:
                continue
            key = (c.a, c.b)
            if key not in best or c.penetration > best[key].penetration:
                best[key] = c
        for c in best.values():
            a = self.bodies[c.a]
            b = self.bodies[c.b] if c.b >= 0 else None
            corr = (c.penetration - self.slop) * 0.8
            wa = a.inv_mass if a.kind == "dynamic" else 0.0
            wb = 0.0 if b is None or b.kind != "dynamic" else b.inv_mass
            total = wa + wb
            if total <= 0.0:
                continue
            push = c.normal * corr
            if wa > 0.0:
                a.pos = a.pos + push * (wa / total)
                a.asleep = False
            if b is not None and wb > 0.0:
                b.pos = b.pos - push * (wb / total)
                b.asleep = False

    def _sleep(self) -> None:
        for body in self.bodies:
            if body.kind != "dynamic":
                continue
            if body.aabb_min[1] < self.ground_y - self.slop:
                body.asleep = False
                body.sleep_steps = 0
                continue
            slow = float(np.linalg.norm(body.lin)) < self.sleep_lin and float(np.linalg.norm(body.ang)) < self.sleep_ang
            # Gravity exceeds sleep_lin within a step, so a body that is actually falling does not
            # accumulate this counter. A stack sitting in the slop does, and freezes before solver
            # noise can walk it.
            if slow:
                body.sleep_steps += 1
            else:
                body.sleep_steps = 0
                body.asleep = False
            if body.sleep_steps >= self.sleep_after:
                body.asleep = True
                body.lin[:] = 0.0
                body.ang[:] = 0.0

    def step(self, dt: float, quantize: bool = False) -> None:
        dt = float(dt)
        for body in self.bodies:
            if body.kind == "dynamic" and not body.asleep:
                body.lin = body.lin + self.gravity * dt
        # Intact glue is a rigid cluster, not a per-iteration impulse. Hundreds of glue solves
        # were the whole step; averaging a cluster is one pass. Break before the average so a
        # body that was kicked this step is not pulled back into the cluster.
        self._break_glues_from_pull()
        self._spread_cluster_velocity()
        # Move first. A fast body can cross the ground inside one step; the contact pass then
        # sees that penetration and takes the inward speed off before the position is projected
        # back out. Solving only the contacts from the start of the step misses that impact.
        for body in self.bodies:
            if body.asleep or body.kind == "static":
                continue
            if body.kind == "kinematic" or body.inv_mass > 0.0:
                body.pos = body.pos + body.lin * dt
                body.quat = quat_integrate(body.quat, body.ang, dt)
        self._impact_sweep(dt)
        self._cache_frames()
        contacts = self._detect()
        for _ in range(self.iterations):
            for contact in contacts:
                self._solve_contact(contact, True, dt)
        self._drop_frames()
        for _ in range(self.position_iters):
            self._positional(self._detect())
        # Integration can sink a glued cluster by one step; the per-piece ground contacts are
        # skipped for that cluster, so lift it after the move.
        self._spread_cluster_velocity()
        self._sleep()
        if quantize:
            self.quantize()

    def quantize(self) -> None:
        """Snap the cached degrees of freedom to float32, the EXR representation."""
        for body in self.bodies:
            body.pos = np.asarray(np.asarray(body.pos, np.float32), np.float64)
            body.lin = np.asarray(np.asarray(body.lin, np.float32), np.float64)
            body.ang = np.asarray(np.asarray(body.ang, np.float32), np.float64)
            body.quat = quat_normalize(np.asarray(np.asarray(body.quat, np.float32), np.float64))

    # -- queries ----------------------------------------------------------------------------------------------
    def energy(self) -> float:
        total = 0.0
        for body in self.bodies:
            if body.inv_mass <= 0.0:
                continue
            total += -body.mass * float(np.dot(self.gravity, body.pos))
            total += 0.5 * body.mass * float(np.dot(body.lin, body.lin))
            r = quat_matrix(body.quat)
            inertia = r @ body.inertia @ r.T
            total += 0.5 * float(np.dot(body.ang, inertia @ body.ang))
        return total

    def min_vertex_y(self) -> float:
        lowest = 1e9
        for body in self.bodies:
            if len(body.vertices) == 0:
                lowest = min(lowest, float(body.pos[1] - body.radius))
            else:
                lowest = min(lowest, float(self.world_vertices(body)[:, 1].min()))
        return lowest

    def max_penetration(self) -> float:
        deepest = 0.0
        for body in self.bodies:
            if len(body.vertices) == 0:
                deepest = max(deepest, self.ground_y - float(body.pos[1] - body.radius))
            else:
                deepest = max(deepest, self.ground_y - float(self.world_vertices(body)[:, 1].min()))
        for c in self._detect():
            deepest = max(deepest, c.penetration)
        return float(max(deepest, 0.0))

    def snapshot(self) -> dict:
        return {
            "bodies": [(b.pos.copy(), b.quat.copy(), b.lin.copy(), b.ang.copy(), b.asleep, b.sleep_steps, b.kind)
                       for b in self.bodies],
            "glue": [g.broken for g in self.glues],
        }

    def restore(self, snap: dict) -> None:
        for body, state in zip(self.bodies, snap["bodies"]):
            body.pos, body.quat, body.lin, body.ang = (x.copy() for x in state[:4])
            body.asleep, body.sleep_steps, body.kind = state[4], state[5], state[6]
            body.inv_mass = 0.0 if body.kind != "dynamic" else (0.0 if body.mass <= 0 else 1.0 / body.mass)
        for glue, broken in zip(self.glues, snap["glue"]):
            glue.broken = broken
            glue.last_impulse = 0.0
            # A later bake restores this snapshot. Leaving the emit flag set skips dust on the next break.
            glue._emitted = False
        self._glue_pairs = None

    def _body_rows(self) -> int:
        return max(1, (len(self.bodies) + 127) // 128)

    def _write_glue_flags(self, image: np.ndarray) -> None:
        """Glue broken/emitted flags in the unused body rows of the position quadrant.

        Body i lives on row i // 128, so rows at and above `_body_rows` are empty until the texture is
        full (128×128 bodies). Each free texel holds four flags.
        """
        row0 = self._body_rows()
        cap = (128 - row0) * 256 * 4
        for i, glue in enumerate(self.glues):
            if i >= cap:
                break
            row = row0 + i // 1024
            col = (i // 4) % 256
            ch = i % 4
            flag = (1.0 if glue.broken else 0.0) + (2.0 if getattr(glue, "_emitted", False) else 0.0)
            image[row, col, ch] = flag

    def _read_glue_flags(self, image: np.ndarray) -> None:
        row0 = self._body_rows()
        cap = (128 - row0) * 256 * 4
        for i, glue in enumerate(self.glues):
            if i >= cap:
                break
            row = row0 + i // 1024
            col = (i // 4) % 256
            ch = i % 4
            flag = int(round(float(image[row, col, ch])))
            glue.broken = bool(flag & 1)
            glue._emitted = bool(flag & 2)
            glue.last_impulse = 0.0
        self._glue_pairs = None

    def pack_state(self) -> np.ndarray:
        """256² rgba32float. Four quadrants: position, quaternion, linear velocity, angular velocity.

        Linear-velocity w is the sleep counter (inverse mass is recomputed from kind). Glue flags sit in
        the unused rows of the position quadrant.
        """
        image = np.zeros((256, 256, 4), dtype=np.float32)
        kind_code = {"dynamic": 0.0, "kinematic": 1.0, "static": 2.0}
        for i, body in enumerate(self.bodies):
            x = i % 128
            y = i // 128
            if y >= 128:
                break
            image[y, x] = (body.pos[0], body.pos[1], body.pos[2], 1.0 if body.asleep else 0.0)
            image[y, x + 128] = body.quat
            image[y + 128, x] = (body.lin[0], body.lin[1], body.lin[2], float(body.sleep_steps))
            image[y + 128, x + 128] = (body.ang[0], body.ang[1], body.ang[2], kind_code.get(body.kind, 0.0))
        self._write_glue_flags(image)
        return image

    def load_state(self, image: np.ndarray) -> None:
        kind_of = {0.0: "dynamic", 1.0: "kinematic", 2.0: "static"}
        image = np.asarray(image, dtype=np.float32)
        for i, body in enumerate(self.bodies):
            x = i % 128
            y = i // 128
            pos = image[y, x]
            quat = image[y, x + 128]
            lin = image[y + 128, x]
            ang = image[y + 128, x + 128]
            body.pos = np.asarray(pos[:3], np.float64)
            body.asleep = float(pos[3]) > 0.5
            # Keep the stored quaternion. Renormalizing here drifts a ulp and the next step diverges.
            body.quat = np.asarray(quat, np.float64)
            body.lin = np.asarray(lin[:3], np.float64)
            body.ang = np.asarray(ang[:3], np.float64)
            code = float(np.round(ang[3]))
            body.kind = kind_of.get(code, body.kind)
            body.inv_mass = 0.0 if body.kind != "dynamic" else (0.0 if body.mass <= 0 else 1.0 / body.mass)
            body.sleep_steps = int(np.clip(np.round(float(lin[3])), 0, 10_000))
        self._read_glue_flags(image)
