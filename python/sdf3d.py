"""Analytic signed distances and mesh helpers shared by the 3D obstacle publishers.

Pure Python so the sign convention (negative inside) and the atlas-independent geometry can be tested
without TouchDesigner. The GLSL publishers in shaders/fluid3d duplicate these formulas.
"""

from __future__ import annotations

import math


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _len(a):
    return math.sqrt(_dot(a, a))


def sd_sphere(p, center, radius: float) -> float:
    return _len(_sub(p, center)) - float(radius)


def sd_box(p, center, half) -> float:
    q = (abs(p[0] - center[0]) - half[0], abs(p[1] - center[1]) - half[1], abs(p[2] - center[2]) - half[2])
    outside = math.sqrt(max(q[0], 0.0) ** 2 + max(q[1], 0.0) ** 2 + max(q[2], 0.0) ** 2)
    inside = min(max(q[0], q[1], q[2]), 0.0)
    return outside + inside


def sd_capsule(p, a, b, radius: float) -> float:
    pa, ba = _sub(p, a), _sub(b, a)
    denom = _dot(ba, ba)
    h = 0.0 if denom <= 0.0 else min(1.0, max(0.0, _dot(pa, ba) / denom))
    return _len(_sub(pa, _mul(ba, h))) - float(radius)


def triangulate(points) -> list:
    """Fan-triangulate one polygon (3 or more points) into triangles of points."""
    pts = [tuple(p) for p in points]
    if len(pts) < 3:
        return []
    if len(pts) == 3:
        return [pts]
    return [[pts[0], pts[i], pts[i + 1]] for i in range(1, len(pts) - 1)]


def closest_on_triangle(p, a, b, c):
    """Closest point on triangle abc to p (Ericson)."""
    ab, ac, ap = _sub(b, a), _sub(c, a), _sub(p, a)
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = _sub(p, b)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return _add(a, _mul(ab, v))
    cp = _sub(p, c)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return _add(a, _mul(ac, w))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return _add(b, _mul(_sub(c, b), w))
    denom = va + vb + vc
    if denom == 0.0:
        return a
    v = vb / denom
    w = vc / denom
    return _add(a, _add(_mul(ab, v), _mul(ac, w)))


def triangle_normal(a, b, c):
    n = _cross(_sub(b, a), _sub(c, a))
    length = _len(n)
    if length <= 1e-12:
        return (0.0, 1.0, 0.0)
    return (n[0] / length, n[1] / length, n[2] / length)


def orient_outward(triangles) -> list:
    """Flip any triangle whose normal points toward the vertex centroid (convex-friendly outward)."""
    if not triangles:
        return []
    count = 0
    acc = [0.0, 0.0, 0.0]
    for a, b, c in triangles:
        for p in (a, b, c):
            acc[0] += p[0]
            acc[1] += p[1]
            acc[2] += p[2]
            count += 1
    centroid = (acc[0] / count, acc[1] / count, acc[2] / count)
    out = []
    for a, b, c in triangles:
        n = triangle_normal(a, b, c)
        mid = ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0)
        if _dot(n, _sub(mid, centroid)) < 0.0:
            out.append([a, c, b])
        else:
            out.append([a, b, c])
    return out


def mesh_sdf(p, triangles) -> float:
    """Exact closest-point signed distance. Negative inside when normals point outward."""
    best_d = 1e20
    best_s = 1.0
    for a, b, c in triangles:
        q = closest_on_triangle(p, a, b, c)
        d = _len(_sub(p, q))
        if d < best_d:
            best_d = d
            n = triangle_normal(a, b, c)
            best_s = -1.0 if _dot(_sub(p, q), n) < 0.0 else 1.0
    return best_s * best_d
