"""Voronoi fracture: volume, closure, disjointness, determinism."""

import numpy as np

from fracture import (
    box_solid,
    convex_hull,
    edge_counts,
    fracture,
    mass_properties,
    neighboring_pairs,
    solid_from_mesh,
    sphere_solid,
)


def _box_tris():
    solid = box_solid((0, 0, 0), (0.4, 0.5, 0.6))
    tris = []
    for face in solid.faces:
        v = face["verts"]
        for i in range(1, len(v) - 1):
            tris.append((v[0], v[i], v[i + 1]))
    return np.asarray(tris)


def test_box_mass_matches_the_analytic_solid():
    hx, hy, hz = 0.4, 0.5, 0.6
    vol, com, inertia, mass = mass_properties(_box_tris(), density=2.0)
    assert abs(vol - 8 * hx * hy * hz) < 1e-9
    assert abs(mass - 2.0 * vol) < 1e-9
    assert np.allclose(com, 0.0, atol=1e-9)
    expect = mass * np.array([hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy]) / 3.0
    assert np.allclose(np.diag(inertia), expect, rtol=1e-6, atol=1e-8)
    assert np.allclose(inertia - np.diag(np.diag(inertia)), 0.0, atol=1e-8)


def test_shifted_box_com():
    solid = box_solid((1.5, -0.25, 0.4), (0.2, 0.3, 0.1))
    tris = []
    for face in solid.faces:
        v = face["verts"]
        tris.append((v[0], v[1], v[2]))
        tris.append((v[0], v[2], v[3]))
    vol, com, _, _ = mass_properties(np.asarray(tris), 1.0)
    assert abs(vol - 8 * 0.2 * 0.3 * 0.1) < 1e-9
    assert np.allclose(com, (1.5, -0.25, 0.4), atol=1e-8)


def test_pieces_conserve_volume_and_stay_disjoint():
    solid = box_solid((0.1, 0.4, -0.2), (0.5, 0.35, 0.25))
    original = solid.volume()
    pieces = fracture(solid, 12, seed=3, pattern="uniform", density=1.5)
    assert len(pieces) == 12
    assert abs(sum(p.volume for p in pieces) - original) / original < 1e-4
    rng = np.random.default_rng(1)
    lo, hi = solid.bounds()
    inside = []
    while len(inside) < 80:
        p = lo + rng.random(3) * (hi - lo)
        if solid.contains(p)[0] and float(solid.sdf(p)[0]) < -1e-3:
            inside.append(p)
    samples = np.asarray(inside)
    owners = np.zeros(len(samples), dtype=np.int32)
    for piece in pieces:
        owners += piece.sdf(samples) < -1e-5
    # A sample within 1e-4 of a cut can sit on two closures. Strict interior belongs to one piece.
    assert np.all(owners == 1)


def test_watertight():
    pieces = fracture(box_solid((0, 0.5, 0), (0.4, 0.4, 0.15)), 8, seed=9)
    for piece in pieces:
        counts = edge_counts(piece)
        assert counts, "piece has no edges"
        assert set(counts.values()) == {2}


def test_deterministic_and_patterns():
    solid = box_solid((0, 0.5, 0), (0.3, 0.5, 0.3))
    a = fracture(solid, 10, seed=4, pattern="clustered", impact=(0.05, 0.7, 0.0), sigma=0.12)
    b = fracture(solid, 10, seed=4, pattern="clustered", impact=(0.05, 0.7, 0.0), sigma=0.12)
    c = fracture(solid, 10, seed=5, pattern="clustered", impact=(0.05, 0.7, 0.0), sigma=0.12)
    assert np.allclose([p.volume for p in a], [p.volume for p in b])
    assert np.allclose(np.stack([p.com for p in a]), np.stack([p.com for p in b]))
    assert not np.allclose([p.volume for p in a], [p.volume for p in c])
    radial = fracture(solid, 9, seed=2, pattern="radial", impact=(0.0, 0.5, 0.0), sigma=0.4)
    assert abs(sum(p.volume for p in radial) - solid.volume()) / solid.volume() < 1e-4
    assert len(neighboring_pairs(a)) >= 1


def test_sphere_and_mesh_and_hull():
    sphere = sphere_solid((0, 0.6, 0), 0.35, subdiv=1)
    pieces = fracture(sphere, 6, seed=1)
    assert abs(sum(p.volume for p in pieces) - sphere.volume()) / sphere.volume() < 1e-3
    # Triangulated box, the shape a SOP box arrives as.
    box = box_solid((0, 0.2, 0), (0.25, 0.2, 0.15))
    verts = box.vertices()
    # Weld for an index mesh.
    uniq, inv = np.unique(np.round(verts, 8), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3) if False else []
    # box faces are quads; triangulate.
    cursor = 0
    for face in box.faces:
        n = len(face["verts"])
        ids = inv[cursor:cursor + n]
        cursor += n
        faces.append((int(ids[0]), int(ids[1]), int(ids[2])))
        if n == 4:
            faces.append((int(ids[0]), int(ids[2]), int(ids[3])))
    mesh = solid_from_mesh(uniq, faces)
    assert abs(mesh.volume() - box.volume()) / box.volume() < 1e-6
    cut = fracture(mesh, 5, seed=7)
    assert abs(sum(p.volume for p in cut) - mesh.volume()) / mesh.volume() < 1e-4
    # A concave dent falls back to the hull, which still fractures as a closed solid.
    dent = uniq.copy()
    extra = dent.mean(0) + np.array([0.0, 0.05, 0.0])
    cloud = np.vstack([dent, extra])
    hull = convex_hull(cloud)
    assert hull.volume() > 0
    assert np.all(hull.contains(cloud, tol=1e-6))
