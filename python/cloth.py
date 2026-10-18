"""XPBD cloth. The math here is the numpy reference; the GPU shaders follow the same steps.

A grid of vertices carries stretch and shear, Bergou isometric bending, and long-range
attachments to the pins. One substep is a predict, one graph-coloured Gauss-Seidel iteration,
then collisions. Macklin's small-steps scheme is many of those substeps per frame. Within a
colour the constraints share no vertex, so Jacobi and Gauss-Seidel are the same update and the
result does not depend on thread order.

Compliance is a material constant. The solver uses alpha / dt^2, which is what makes the rest
shape independent of the timestep once the iteration has settled.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

KINDS = ("struct_h", "struct_v", "shear_a", "shear_b", "bend_h", "bend_v")
# Must match shaders/cloth/solve.glsl and build_cloth.MAX_SUBSTEPS.
MAX_SUBSTEPS = 16
# Distance constraints solved each substep. Bending is isometric (Bergou), not these skip-edges.
SOLVE_KINDS = ("struct_h", "struct_v", "shear_a", "shear_b")
SELF_ITERS = 4


def spacing_of(size: float, count: int) -> float:
    return float(size) / float(max(int(count) - 1, 1))


def constraint_ends(kind: str, i: int, j: int, nx: int, ny: int):
    """Endpoints of the constraint owned by vertex (i, j), or None when it falls off the grid."""
    if kind == "struct_h":
        if i + 1 >= nx:
            return None
        return (i, j), (i + 1, j)
    if kind == "struct_v":
        if j + 1 >= ny:
            return None
        return (i, j), (i, j + 1)
    if kind == "shear_a":
        if i + 1 >= nx or j + 1 >= ny:
            return None
        return (i, j), (i + 1, j + 1)
    if kind == "shear_b":
        if i + 1 >= nx or j + 1 >= ny:
            return None
        return (i, j + 1), (i + 1, j)
    if kind == "bend_h":
        if i + 2 >= nx:
            return None
        return (i, j), (i + 2, j)
    if kind == "bend_v":
        if j + 2 >= ny:
            return None
        return (i, j), (i, j + 2)
    raise KeyError(kind)


def owner_color(kind: str, i: int, j: int) -> int:
    """Colour of the constraint owned by (i, j). Two colours per kind; see vertex_role."""
    if kind in ("struct_h", "shear_a", "shear_b"):
        return i & 1
    if kind == "struct_v":
        return j & 1
    if kind == "bend_h":
        return (i // 2) & 1
    if kind == "bend_v":
        return (j // 2) & 1
    raise KeyError(kind)


def vertex_role(kind: str, color: int, i: int, j: int, nx: int, ny: int):
    """(sign, owner_i, owner_j) when vertex (i, j) is an endpoint of this colour, else None.

    Sign +1 is endpoint A (the first vertex returned by constraint_ends). The GPU writes only
    that vertex's own correction, reading both endpoints from the previous pass.
    """
    if kind == "struct_h":
        if (i & 1) == color and i + 1 < nx:
            return (1, i, j)
        if i > 0 and ((i - 1) & 1) == color:
            return (-1, i - 1, j)
        return None
    if kind == "struct_v":
        if (j & 1) == color and j + 1 < ny:
            return (1, i, j)
        if j > 0 and ((j - 1) & 1) == color:
            return (-1, i, j - 1)
        return None
    if kind == "shear_a":
        if (i & 1) == color and i + 1 < nx and j + 1 < ny:
            return (1, i, j)
        if i > 0 and j > 0 and ((i - 1) & 1) == color:
            return (-1, i - 1, j - 1)
        return None
    if kind == "shear_b":
        if j > 0 and (i & 1) == color and i + 1 < nx:
            return (1, i, j - 1)
        if i > 0 and ((i - 1) & 1) == color and j + 1 < ny:
            return (-1, i - 1, j)
        return None
    if kind == "bend_h":
        if ((i // 2) & 1) == color and i + 2 < nx:
            return (1, i, j)
        if i >= 2 and (((i - 2) // 2) & 1) == color:
            return (-1, i - 2, j)
        return None
    if kind == "bend_v":
        if ((j // 2) & 1) == color and j + 2 < ny:
            return (1, i, j)
        if j >= 2 and (((j - 2) // 2) & 1) == color:
            return (-1, i, j - 2)
        return None
    raise KeyError(kind)


def build_constraints(nx: int, ny: int) -> list[dict]:
    items = []
    for kind in KINDS:
        for j in range(ny):
            for i in range(nx):
                ends = constraint_ends(kind, i, j, nx, ny)
                if ends is None:
                    continue
                items.append({
                    "kind": kind,
                    "i": i,
                    "j": j,
                    "color": owner_color(kind, i, j),
                    "a": ends[0],
                    "b": ends[1],
                })
    return items


def colors_are_independent(items: list[dict]) -> bool:
    """No colour of one kind shares a vertex. That is the Gauss-Seidel independence condition."""
    groups: dict[tuple, set] = {}
    for item in items:
        key = (item["kind"], item["color"])
        used = groups.setdefault(key, set())
        for vertex in (item["a"], item["b"]):
            if vertex in used:
                return False
            used.add(vertex)
    return True


def rest_lengths(dx: float, dy: float) -> dict[str, float]:
    diag = math.hypot(dx, dy)
    return {
        "struct_h": float(dx),
        "struct_v": float(dy),
        "shear_a": diag,
        "shear_b": diag,
        "bend_h": 2.0 * float(dx),
        "bend_v": 2.0 * float(dy),
    }


def grid_positions(nx: int, ny: int, origin, size, plane: str = "xz") -> np.ndarray:
    """Rest pose, shape (ny, nx, 3). `plane` is `xy` (vertical flag) or `xz` (horizontal drape)."""
    dx = spacing_of(size[0], nx)
    dy = spacing_of(size[1], ny)
    pos = np.zeros((ny, nx, 3), dtype=np.float64)
    ox, oy, oz = (float(v) for v in origin)
    for j in range(ny):
        for i in range(nx):
            if plane == "xy":
                pos[j, i] = (ox + i * dx, oy - j * dy, oz)
            else:
                pos[j, i] = (ox + i * dx, oy, oz + j * dy)
    return pos


def inverse_mass(nx: int, ny: int, pin: str, density: float, dx: float, dy: float) -> np.ndarray:
    mass = max(float(density) * float(dx) * float(dy), 1e-8)
    inv = np.full((ny, nx), 1.0 / mass, dtype=np.float64)
    if pin == "left":
        inv[:, 0] = 0.0
    elif pin == "corners":
        inv[0, 0] = 0.0
        inv[0, nx - 1] = 0.0
    elif pin == "top":
        inv[0, :] = 0.0
    elif pin != "none":
        raise ValueError(pin)
    return inv


def triangle_normal(pos: np.ndarray, a, b, c) -> np.ndarray:
    pa = pos[a[1], a[0]]
    pb = pos[b[1], b[0]]
    pc = pos[c[1], c[0]]
    n = np.cross(pb - pa, pc - pa)
    length = float(np.linalg.norm(n))
    if length < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return n / length


def bending_rest_angles(pos: np.ndarray) -> np.ndarray:
    """Angle between normals of every pair of triangles that share an edge.

    A flat grid is coplanar, so every angle is 0. A right-angle fold returns pi/2 on the
    edges that hinge. Vertex indices are (i, j) into `pos` shaped (ny, nx, 3).
    """
    ny, nx = pos.shape[:2]
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))
    edges: dict[tuple, list] = {}
    for face in faces:
        for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges.setdefault(tuple(sorted(edge)), []).append(face)
    angles = []
    for faces_on_edge in edges.values():
        if len(faces_on_edge) != 2:
            continue
        n0 = triangle_normal(pos, *faces_on_edge[0])
        n1 = triangle_normal(pos, *faces_on_edge[1])
        if float(np.linalg.norm(n0)) < 0.5 or float(np.linalg.norm(n1)) < 0.5:
            continue
        angles.append(math.acos(float(np.clip(np.dot(n0, n1), -1.0, 1.0))))
    return np.array(angles, dtype=np.float64)


def vertex_normal(pos: np.ndarray, i: int, j: int) -> np.ndarray:
    ny, nx = pos.shape[:2]

    def at(x, y):
        return pos[min(max(y, 0), ny - 1), min(max(x, 0), nx - 1)]

    px = at(i + 1, j) - at(i - 1, j)
    py = at(i, j + 1) - at(i, j - 1)
    n = np.cross(px, py)
    length = float(np.linalg.norm(n))
    if length < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return n / length


def aero_force(normal, v_cloth, v_air, area: float, rho: float, drag: float, cap: float) -> np.ndarray:
    """Pressure along the normal plus tangential drag. Both push the cloth downwind.

    `v_air - v_cloth` is the wind relative to the vertex. The normal term is
    rho * area * vn * |vn| * n with vn = dot(v_rel, n), so the force follows the wind
    whichever way the normal points. Tangential drag uses the same relative velocity.
    `cap` limits the relative speed the force sees, which keeps one bad sample finite.
    """
    v_rel = np.asarray(v_air, dtype=np.float64) - np.asarray(v_cloth, dtype=np.float64)
    speed = float(np.linalg.norm(v_rel))
    if cap > 0.0 and speed > cap:
        v_rel = v_rel * (cap / speed)
    n = np.asarray(normal, dtype=np.float64)
    vn = float(np.dot(v_rel, n))
    vt = v_rel - vn * n
    force = float(rho) * float(area) * vn * abs(vn) * n
    vt_len = float(np.linalg.norm(vt))
    force = force + float(drag) * float(rho) * float(area) * vt_len * vt
    return force


def _apply_color(pos, inv, bits, kind, color, rest, alpha, nx, ny):
    delta = np.zeros_like(pos)
    for j in range(ny):
        for i in range(nx):
            role = vertex_role(kind, color, i, j, nx, ny)
            if role is None:
                continue
            sign, i0, j0 = role
            ends = constraint_ends(kind, i0, j0, nx, ny)
            if ends is None:
                continue
            broken = int(bits[j0, i0])
            if kind == "struct_h" and broken & 1:
                continue
            if kind == "struct_v" and broken & 2:
                continue
            (ia, ja), (ib, jb) = ends
            p1 = pos[ja, ia]
            p2 = pos[jb, ib]
            diff = p1 - p2
            length = float(np.linalg.norm(diff))
            grad = diff / length if length > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=pos.dtype)
            c = length - rest
            w1 = float(inv[ja, ia])
            w2 = float(inv[jb, ib])
            denom = w1 + w2 + alpha
            if denom <= 0.0:
                continue
            dl = -c / denom
            wi = w1 if sign > 0 else w2
            delta[j, i] = delta[j, i] + (sign * wi * dl) * grad
    pos += delta


def _update_tear(pos, bits, lengths, threshold, nx, ny):
    if threshold <= 0.0:
        return bits
    out = bits.copy()
    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx:
                length = float(np.linalg.norm(pos[j, i] - pos[j, i + 1]))
                if length > lengths["struct_h"] * (1.0 + threshold):
                    out[j, i] = int(out[j, i]) | 1
            if j + 1 < ny:
                length = float(np.linalg.norm(pos[j, i] - pos[j + 1, i]))
                if length > lengths["struct_v"] * (1.0 + threshold):
                    out[j, i] = int(out[j, i]) | 2
    return out


def _iso_bend(pos, inv, alpha):
    """One Jacobi step of quadratic isometric bending on a regular grid.

    Bergou et al. 2006: the bending energy is quadratic in positions, so the constraint is
    linear. On a flat regular grid the stencil is the 5-point Laplacian (the cotan weights
    are equal). Only the centre vertex is moved, which is a Jacobi update and needs no
    colouring. High-frequency wrinkles die in a few iterations; the sheet can still drape.
    """
    ny, nx = inv.shape
    out = pos.copy()
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            wc = float(inv[j, i])
            if wc <= 0.0:
                continue
            c = (4.0 * pos[j, i] - pos[j, i - 1] - pos[j, i + 1] - pos[j - 1, i] - pos[j + 1, i])
            wn = float(inv[j, i - 1] + inv[j, i + 1] + inv[j - 1, i] + inv[j + 1, i])
            denom = 16.0 * wc + wn + alpha
            if denom <= 0.0:
                continue
            out[j, i] = pos[j, i] + (-4.0 * wc / denom) * c
    return out


def _project_anchor(pos, j, i, anchor, rest_len):
    if rest_len <= 1e-8:
        return
    delta = pos[j, i] - anchor
    dist = float(np.linalg.norm(delta))
    if dist > rest_len:
        pos[j, i] = anchor + delta * (rest_len / dist)


def _long_range(pos, rest, inv, dx, dy, pin):
    """Unilateral max-distance to the pins (Kim, Chentanez, Müller 2012).

    The limit is the rest geodesic along the grid. A vertex cannot end up farther from a
    pin than the cloth between them, which is what stops a hanging sheet from growing.
    """
    ny, nx = inv.shape
    for j in range(ny):
        for i in range(nx):
            if inv[j, i] <= 0.0:
                continue
            if pin == "top":
                _project_anchor(pos, j, i, rest[0, i], j * dy)
            elif pin == "left":
                _project_anchor(pos, j, i, rest[j, 0], i * dx)
            elif pin == "corners":
                _project_anchor(pos, j, i, rest[0, 0], i * dx + j * dy)
                _project_anchor(pos, j, i, rest[0, nx - 1], (nx - 1 - i) * dx + j * dy)


def xpbd_substep(pos, vel, inv, rest, bits, dt, compliance, gravity, damping, lengths,
                 air=0.0, drag=0.0, wind=None, wind_cap=12.0, tear=0.0, ground=None,
                 thickness=0.0, friction=0.3, pin="none"):
    """One substep. Arrays are updated in place and also returned. `wind` is a single vec3 or None.

    Ground, when set, is (y, friction is ignored here — `friction` is the argument). Pinned
    vertices stay on `rest`. Air density 0 skips the aerodynamic force.
    """
    pos = np.array(pos, dtype=np.float64, copy=True)
    vel = np.array(vel, dtype=np.float64, copy=True)
    inv = np.asarray(inv, dtype=np.float64)
    rest = np.asarray(rest, dtype=np.float64)
    bits = np.array(bits, copy=True)
    ny, nx = inv.shape
    dt = float(dt)
    damp = math.exp(-float(damping) * dt)
    g = np.asarray(gravity, dtype=np.float64) * dt
    wind_v = np.zeros(3, dtype=np.float64) if wind is None else np.asarray(wind, dtype=np.float64)
    dx = lengths["struct_h"]
    dy = lengths["struct_v"]
    area = dx * dy
    prev = pos.copy()
    for j in range(ny):
        for i in range(nx):
            if inv[j, i] <= 0.0:
                pos[j, i] = rest[j, i]
                continue
            v = vel[j, i] * damp + g
            if air > 0.0:
                n = vertex_normal(prev, i, j)
                force = aero_force(n, v, wind_v, area, air, drag, wind_cap)
                v = v + force * float(inv[j, i]) * dt
            pos[j, i] = prev[j, i] + v * dt
    # Break stretched edges before the projection, so a stiff constraint cannot pull a torn
    # edge back to rest in the same substep. The GPU does the same check inside the colour pass.
    bits = _update_tear(pos, bits, lengths, float(tear), nx, ny)
    alphas = {kind: float(compliance.get(kind, 0.0)) / (dt * dt) for kind in SOLVE_KINDS}
    for kind in SOLVE_KINDS:
        for color in (0, 1):
            _apply_color(pos, inv, bits, kind, color, lengths[kind], alphas[kind], nx, ny)
    for kind in ("struct_h", "struct_v"):
        for color in (0, 1):
            _apply_color(pos, inv, bits, kind, color, lengths[kind], alphas[kind], nx, ny)
    bend_alpha = float(compliance.get("bend_h", compliance.get("bend", 1e-3))) / (dt * dt)
    for _ in range(4):
        pos = _iso_bend(pos, inv, bend_alpha)
    for _ in range(2):
        _long_range(pos, rest, inv, dx, dy, pin)
    if ground is not None and thickness >= 0.0:
        y = float(ground) + float(thickness)
        for j in range(ny):
            for i in range(nx):
                if inv[j, i] <= 0.0 or pos[j, i, 1] >= y:
                    continue
                n = np.array([0.0, 1.0, 0.0])
                pos[j, i, 1] = y
                disp = pos[j, i] - prev[j, i]
                rel = disp
                normal = float(np.dot(rel, n))
                if normal < 0.0:
                    rel = rel - normal * n
                tang = rel - float(np.dot(rel, n)) * n
                rel = rel - tang * min(max(float(friction), 0.0), 1.0)
                pos[j, i] = prev[j, i] + rel
                pos[j, i, 1] = max(pos[j, i, 1], y)
    vel = (pos - prev) / dt
    return pos, vel, bits


def edge_strains(pos, lengths, nx, ny) -> np.ndarray:
    strains = []
    dx, dy = lengths["struct_h"], lengths["struct_v"]
    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx:
                strains.append(abs(float(np.linalg.norm(pos[j, i] - pos[j, i + 1])) - dx) / dx)
            if j + 1 < ny:
                strains.append(abs(float(np.linalg.norm(pos[j, i] - pos[j + 1, i])) - dy) / dy)
    return np.array(strains, dtype=np.float64)


def theoretical_strain(nx: int, ny: int, density: float, dx: float, dy: float, compliance: float, gravity: float) -> float:
    """Upper bound on stretch from the weight of one column of vertices hanging on one edge.

    extension = F / k with k = 1 / compliance, and strain = extension / rest length. A real
    cloth shares that load across many edges, so this is a bound, not the expected value.
    """
    mass = float(density) * float(dx) * float(dy)
    force = ny * mass * abs(float(gravity))
    extension = force * float(compliance)
    return extension / max(min(dx, dy), 1e-8)


# -- TouchDesigner extension -------------------------------------------------------------------------------
def _ensure_python_path(comp) -> None:
    path = str(Path(comp.parent.sims.par.Root.eval()) / "python")
    if path not in sys.path:
        sys.path.insert(0, path)


class Cloth:
    """GPU XPBD cloth. State is pos (xyz, inverse mass) and vel (xyz, tear bits) textures."""

    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        _ensure_python_path(ownerComp)
        self._resume = False

    def _par(self):
        return self.ownerComp.par

    def _point(self, name: str, top) -> None:
        node = self.ownerComp.op(name)
        node.par.top.expr = ""
        node.par.top = top.path if hasattr(top, "path") else str(top)

    def ApplyPreset(self, name: str) -> None:
        p = self._par()
        name = str(name)
        fluid = op("/sims/fluid3d")  # noqa: F821
        bus = op.simbus3d  # noqa: F821
        clock = op.simclock  # noqa: F821
        if name == "showcase":
            p.Resx, p.Resy = 256, 256
            p.Substeps, p.Selfiters = 8, 0
            p.Plane, p.Pin = "xz", "none"
            p.Originx, p.Originy, p.Originz = -0.55, 1.35, -0.55
            p.Sizex, p.Sizey = 1.1, 1.1
            p.Thickness, p.Hashres = 0.004, 80
            p.Air, p.Drag, p.Couple = 0.4, 0.2, False
            p.Collide, p.Ground = True, 0.0
            p.Stretch, p.Shear, p.Bend = 1e-5, 1e-5, 2e-4
            p.Damping, p.Tear = 0.4, 0.0
            p.Spherx, p.Sphery, p.Spherez, p.Spherer = 0.0, 0.42, 0.0, 0.28
            p.Sphervx, p.Sphervy, p.Sphervz = 0.0, 0.0, 0.0
            p.Cookrender = False
            p.Camaz, p.Camel, p.Camdist, p.Camfov = 0.55, 0.28, 2.8, 46.0
            p.Targetx, p.Targety, p.Targetz = 0.0, 0.75, 0.0
            self._wind_bus(False)
            self.PublishSphere(True)
            self.PublishForce(False)
            self.PublishJet(False)
        elif name == "hang":
            p.Resx, p.Resy = 32, 32
            p.Substeps, p.Selfiters = 8, 1
            p.Plane, p.Pin = "xz", "corners"
            p.Originx, p.Originy, p.Originz = -0.4, 1.55, -0.4
            p.Sizex, p.Sizey = 0.8, 0.8
            p.Thickness, p.Hashres = 0.004, 32
            p.Air, p.Drag, p.Couple = 0.0, 0.0, False
            p.Collide, p.Ground = False, 0.0
            p.Stretch, p.Shear, p.Bend = 1e-4, 1e-4, 1e-3
            p.Damping, p.Tear = 3.0, 0.0
            p.Cookrender = False
            self._wind_bus(False)
            self.PublishSphere(False)
            self.PublishForce(False)
            self.PublishJet(False)
        elif name == "drape":
            p.Resx, p.Resy = 32, 32
            p.Substeps, p.Selfiters = 8, 0
            p.Plane, p.Pin = "xz", "none"
            p.Originx, p.Originy, p.Originz = -0.32, 0.95, 0.38
            p.Sizex, p.Sizey = 0.64, 0.64
            p.Thickness, p.Hashres = 0.008, 32
            p.Air, p.Drag, p.Couple = 0.0, 0.0, False
            p.Collide, p.Ground = True, 0.0
            p.Stretch, p.Shear, p.Bend = 1e-5, 1e-5, 5e-4
            p.Damping, p.Tear = 1.0, 0.0
            p.Spherx, p.Sphery, p.Spherez, p.Spherer = 0.0, 0.36, 0.7, 0.22
            p.Sphervx, p.Sphervy, p.Sphervz = 0.0, 0.0, 0.0
            p.Cookrender = False
            p.Camaz, p.Camel, p.Camdist, p.Camfov = 0.4, 0.35, 1.8, 40.0
            p.Targetx, p.Targety, p.Targetz = 0.0, 0.55, 0.55
            self._wind_bus(False)
            self.PublishSphere(True)
            self.PublishForce(False)
            self.PublishJet(False)
        else:
            name = "realtime"
            p.Resx, p.Resy = 64, 64
            p.Substeps, p.Selfiters = 8, 1
            p.Plane, p.Pin = "xy", "top"
            p.Originx, p.Originy, p.Originz = 0.2, 1.65, 0.05
            p.Sizex, p.Sizey = 0.55, 0.45
            p.Thickness, p.Hashres = 0.006, 32
            p.Density = 0.04
            p.Air, p.Drag, p.Couple = 1.2, 2.0, False
            p.Collide, p.Ground = True, 0.0
            p.Stretch, p.Shear, p.Bend = 2e-5, 2e-5, 1e-3
            p.Damping, p.Tear = 0.25, 0.0
            p.Cookrender = True
            p.Renderw, p.Renderh = 480, 270
            bus.par.Resw, bus.par.Resh, bus.par.Resd = 48, 48, 48
            fluid.par.Solver = "jacobi"
            fluid.par.Jacobi = 6
            fluid.par.Raysteps = 6
            fluid.par.Shadowsteps = 2
            fluid.par.Renderw, fluid.par.Renderh = 480, 270
            fluid.par.Cookrender = True
            fluid.par.Maccormack = 0
            fluid.par.Windx, fluid.par.Windy, fluid.par.Windz = 0.15, 0.0, 1.6
            fluid.par.Winddrag = 0.8
            fluid.par.Camaz, fluid.par.Camel, fluid.par.Camdist = 0.4, 0.15, 1.8
            fluid.par.Targetx, fluid.par.Targety, fluid.par.Targetz = 0.45, 1.4, 0.15
            clock.par.Sims = "/sims/fluid3d /sims/cloth"
            clock.par.Substeps = 1
            clock.par.Fps = 60
            for node_name in ("render_out", "shadow"):
                node = fluid.op(node_name)
                if node is not None:
                    node.bypass = False
            self._wind_bus(True)
            self.PublishSphere(False)
            self.PublishForce(False)
            self.PublishJet(True)
            p.Camaz, p.Camel, p.Camdist, p.Camfov = 0.4, 0.15, 1.8, 36.0
            p.Targetx, p.Targety, p.Targetz = 0.45, 1.4, 0.15
        p.Preset = name
        self.RebuildMesh()
        self._sync_selects()

    def _wind_bus(self, use_bus: bool) -> None:
        wind = self.ownerComp.op("wind_src")
        wind.par.top.expr = ""
        if use_bus:
            wind.par.top = op.simbus3d.path + "/vel"  # noqa: F821
        else:
            wind.par.top = self.ownerComp.op("uniform_vel").path

    def SetUniformWind(self, x: float, y: float, z: float) -> None:
        """Sample a uniform velocity volume through the same input the bus velocity uses."""
        top = self.ownerComp.op("uniform_vel")
        for i in range(int(top.seq.vec.numBlocks)):
            if str(getattr(top.par, f"vec{i}name").eval()) != "uWind":
                continue
            for axis, value in zip("xyz", (x, y, z)):
                par = getattr(top.par, f"vec{i}value{axis}")
                par.expr = ""
                par.val = float(value)
        self._wind_bus(False)

    def PublishSphere(self, on: bool) -> None:
        bus = op.simbus3d  # noqa: F821
        top = self.ownerComp.op("sphere_obs")
        if on:
            bus.Publish("obstacles", top)
        else:
            bus.Unpublish(top, "obstacles")

    def PublishJet(self, on: bool) -> None:
        """A box of acceleration upwind of the flag. The smoke carries it; the cloth only sees the bus."""
        bus = op.simbus3d  # noqa: F821
        top = self.ownerComp.op("jet")
        if top is None:
            return
        if on:
            bus.Publish("forces", top)
        else:
            bus.Unpublish(top, "forces")

    def PublishForce(self, on: bool) -> None:
        bus = op.simbus3d  # noqa: F821
        top = self.ownerComp.op("cloth_force")
        if on and bool(self._par().Couple.eval()):
            bus.Publish("forces", top)
        else:
            bus.Unpublish(top, "forces")

    def RebuildMesh(self) -> None:
        mesh = self.ownerComp.op("cloth_geo/mesh")
        if mesh is None:
            return
        mesh.lock = False
        mesh.storage["gen"] = None
        mesh.cook(force=True)
        mesh.lock = True

    def FillMesh(self, sop) -> None:
        p = self._par()
        gen = (int(p.Resx.eval()), int(p.Resy.eval()))
        if sop.storage.get("gen") == gen and sop.numPoints > 0:
            return
        nx, ny = gen
        sop.clear()
        sop.pointAttribs.create("N")
        sop.pointAttribs.create("Cd")
        points = []
        for j in range(ny):
            for i in range(nx):
                pt = sop.appendPoint()
                pt.P = (float(i), float(j), 0.0)
                pt.N = (0.0, 0.0, 1.0)
                pt.Cd = (float(i), float(j), 0.0)
                points.append(pt)
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                b = a + 1
                c = a + nx
                d = c + 1
                poly = sop.appendPoly(3, closed=True, addPoints=False)
                poly[0].point = points[a]
                poly[1].point = points[b]
                poly[2].point = points[d]
                poly = sop.appendPoly(3, closed=True, addPoints=False)
                poly[0].point = points[a]
                poly[1].point = points[d]
                poly[2].point = points[c]
        sop.storage["gen"] = gen

    def _sync_selects(self) -> None:
        p = self._par()
        sub = min(max(int(p.Substeps.eval()), 1), MAX_SUBSTEPS)
        self._point("pos_solved", self.ownerComp.op(f"solve_{sub - 1}"))
        self._point("vel_solved", self.ownerComp.op(f"vel_{sub - 1}"))
        iters = min(max(int(p.Selfiters.eval()), 0), SELF_ITERS)
        if iters <= 0:
            self._point("pos_final", self.ownerComp.op("pos_solved"))
        else:
            self._point("pos_final", self.ownerComp.op(f"self_{iters - 1}"))

    def BeginStep(self, reset: bool) -> None:
        self._sync_selects()
        for field in ("pos", "vel"):
            src = self.ownerComp.op("src_" + field)
            src.lock = False
            if reset and self._resume:
                src.par.top.expr = ""
                src.par.top = self.ownerComp.op("resume_" + field).path
            elif reset:
                src.par.top.expr = ""
                src.par.top = self.ownerComp.op("init_" + field).path
            else:
                src.par.top.expr = ""
                src.par.top = self.ownerComp.op(field + "_out").path
            src.cook(force=True)
            # Freeze the frame-start sample. The output velocity is (end - this) / dt, and a
            # later cook must not retarget it at the position this step is about to write.
            src.lock = True
        if reset:
            self._resume = False

    def Step(self) -> None:
        # One cook of the last velocity pulls the unrolled substeps, both sorts, and self-collision.
        vel = self.ownerComp.op("vel_state")
        pos = self.ownerComp.op("pos_state")
        vel.lock = False
        pos.lock = False
        vel.cook(force=True)
        pos.cook(force=True)
        for field, state in (("pos", pos), ("vel", vel)):
            out = self.ownerComp.op(field + "_out")
            out.lock = False
            out.cook(force=True)
            out.lock = True
            state.lock = True
        if bool(self._par().Couple.eval()):
            self.ownerComp.op("aero").cook(force=True)
            self.ownerComp.op("bins_pack").cook(force=True)
            for name in ("aero_out", "bins_out", "keys_out"):
                node = self.ownerComp.op(name)
                node.lock = False
                node.cook(force=True)
                node.lock = True
            self.ownerComp.op("cloth_force").cook(force=True)
        if not bool(self._par().Cookrender.eval()):
            return
        self.ownerComp.op("cloth_render").cook(force=True)
        self.ownerComp.op("composite").cook(force=True)
        self.ownerComp.op("picture_out").cook(force=True)

    def StateTops(self) -> dict:
        return {"pos": self.ownerComp.op("pos_out"), "vel": self.ownerComp.op("vel_out")}

    def StateFormats(self) -> dict:
        return {"pos": "rgba32float", "vel": "rgba32float"}

    def ExportArray(self, field: str):
        if field not in ("pos", "vel"):
            return None
        image = self.ownerComp.op(field + "_out").numpyArray(delayed=False)
        if image is None:
            return None
        return np.ascontiguousarray(np.array(image, dtype=np.float32, copy=True))

    def _hold_resume(self, field: str, image) -> None:
        name = "resume_" + field
        top = self.ownerComp.op(name)
        if top is None:
            top = self.ownerComp.create(scriptTOP, name)  # noqa: F821
            top.par.format = "rgba32float"
        top.par.outputresolution = "custom"
        top.par.resolutionw = int(image.shape[1])
        top.par.resolutionh = int(image.shape[0])
        top.lock = True
        top.copyNumpyArray(np.ascontiguousarray(image, dtype=np.float32))

    def SetInit(self, paths: dict | None = None) -> None:
        self._resume = False
        if not paths:
            return
        root = Path(self.ownerComp.parent.sims.par.Root.eval())
        arrays = {}
        for field in ("pos", "vel"):
            if field not in paths:
                continue
            npy = Path(str(paths[field]))
            if not npy.is_absolute():
                npy = root / npy
            npy = npy.with_suffix(".npy")
            if not npy.is_file():
                continue
            arrays[field] = np.ascontiguousarray(np.load(npy), dtype=np.float32)
        if "pos" not in arrays or "vel" not in arrays:
            self._resume = False
            return
        for field, image in arrays.items():
            self._hold_resume(field, image)
        self._resume = True

    def Params(self) -> dict:
        p = self._par()
        clock = op.simclock  # noqa: F821
        bus = op.simbus3d  # noqa: F821
        shaders = sorted((d.name, d.text) for d in self.ownerComp.findChildren(type=textDAT, depth=1))  # noqa: F821
        digest = hashlib.sha256(repr(shaders).encode("utf-8")).hexdigest()[:16]
        nx, ny = int(p.Resx.eval()), int(p.Resy.eval())
        return {
            "sim": "cloth",
            "preset": str(p.Preset.eval()),
            "res": [nx, ny],
            "vertices": nx * ny,
            "substeps": int(p.Substeps.eval()),
            "iterations": 1,
            "self_iters": int(p.Selfiters.eval()),
            "plane": str(p.Plane.eval()),
            "pin": str(p.Pin.eval()),
            "origin": [float(p.Originx.eval()), float(p.Originy.eval()), float(p.Originz.eval())],
            "size": [float(p.Sizex.eval()), float(p.Sizey.eval())],
            "density": float(p.Density.eval()),
            "compliance": [float(p.Stretch.eval()), float(p.Shear.eval()), float(p.Bend.eval())],
            "damping": float(p.Damping.eval()),
            "thickness": float(p.Thickness.eval()),
            "friction": float(p.Friction.eval()),
            "ground": float(p.Ground.eval()),
            "collide": bool(p.Collide.eval()),
            "air": float(p.Air.eval()),
            "drag": float(p.Drag.eval()),
            "tear": float(p.Tear.eval()),
            "hash": int(p.Hashres.eval()),
            "couple": bool(p.Couple.eval()),
            "sphere": [float(p.Spherx.eval()), float(p.Sphery.eval()), float(p.Spherez.eval()), float(p.Spherer.eval())],
            "bus_res": list(bus.Res),
            "bounds": list(bus.Bounds),
            "fps": clock.par.Fps.eval(),
            "clock_substeps": int(clock.par.Substeps.eval()),
            "seed": int(clock.par.Seed.eval()),
            "shaders": digest,
        }

    def Positions(self):
        image = self.ownerComp.op("pos_out").numpyArray(delayed=False)
        return np.array(image, dtype=np.float32, copy=True)

    def Velocities(self):
        image = self.ownerComp.op("vel_out").numpyArray(delayed=False)
        return np.array(image, dtype=np.float32, copy=True)

    def Scene(self) -> dict:
        p = self._par()
        clock = op.simclock  # noqa: F821
        fluid = op("/sims/fluid3d")  # noqa: F821
        bypass = {}
        errors = []
        for name in ("cloth_render", "composite", "picture_out", "sort0", "sort1", "solve_0", "bins0", "cloth_force"):
            node = self.ownerComp.op(name)
            if node is None:
                continue
            bypass[name] = bool(node.bypass)
            err = node.errors()
            if err:
                errors.append(name + ": " + str(err).splitlines()[0][:160])
        fluid_bypass = {}
        for name in ("render_out", "shadow", "jacobi"):
            node = fluid.op(name)
            if node is None:
                continue
            fluid_bypass[name] = bool(node.bypass)
        nx, ny = int(p.Resx.eval()), int(p.Resy.eval())
        return {
            "sims": str(clock.par.Sims.eval()),
            "vertices": nx * ny,
            "res": [nx, ny],
            "substeps": int(p.Substeps.eval()),
            "iterations": 1,
            "colors": 12,
            "constraints": len(build_constraints(nx, ny)),
            "self_iters": int(p.Selfiters.eval()),
            "hash": int(p.Hashres.eval()),
            "thickness": float(p.Thickness.eval()),
            "collide": bool(p.Collide.eval()),
            "couple": bool(p.Couple.eval()),
            "air": float(p.Air.eval()),
            "cookrender": bool(p.Cookrender.eval()),
            "render": [int(p.Renderw.eval()), int(p.Renderh.eval())],
            "fluid_res": [int(op.simbus3d.par.Resw.eval()), int(op.simbus3d.par.Resh.eval()),  # noqa: F821
                          int(op.simbus3d.par.Resd.eval())],
            "fluid_solver": str(fluid.par.Solver.eval()),
            "fluid_jacobi": int(fluid.par.Jacobi.eval()),
            "fluid_rays": int(fluid.par.Raysteps.eval()),
            "fluid_cookrender": bool(fluid.par.Cookrender.eval()),
            "fluid_wind": [float(fluid.par.Windx.eval()), float(fluid.par.Windy.eval()), float(fluid.par.Windz.eval())],
            "bypass": bypass,
            "fluid_bypass": fluid_bypass,
            "errors": errors,
        }

    def StageTimes(self) -> dict:
        """Wall time of one forced cook per stage, including a single readback so the GPU finishes."""
        import time  # noqa: PLC0415

        names = ("solve_0", "sort0", "self_0", "vel_state", "cloth_render", "composite")
        out = {}
        for name in names:
            node = self.ownerComp.op(name)
            if node is None:
                continue
            t0 = time.perf_counter()
            node.cook(force=True)
            node.numpyArray(delayed=False)
            out[name] = (time.perf_counter() - t0) * 1000.0
        return out
