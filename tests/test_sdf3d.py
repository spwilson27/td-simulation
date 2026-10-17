import math

import sdf3d


def test_primitives_are_negative_inside():
    assert sdf3d.sd_sphere((0, 0, 0), (0, 0, 0), 1) == -1
    assert sdf3d.sd_sphere((2, 0, 0), (0, 0, 0), 1) == 1
    assert sdf3d.sd_box((0, 0, 0), (0, 0, 0), (1, 1, 1)) == -1
    assert abs(sdf3d.sd_box((2, 0, 0), (0, 0, 0), (1, 1, 1)) - 1) < 1e-9
    assert sdf3d.sd_capsule((0, 0, 0), (0, -1, 0), (0, 1, 0), 0.5) < 0
    assert sdf3d.sd_capsule((0, 0, 2), (0, -1, 0), (0, 1, 0), 0.5) > 0


def test_quad_and_outward_mesh_sign():
    # Unit cube, centroid at origin. Quads fan into triangles with outward normals.
    faces = [
        [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5)],
        [(-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)],
        [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, -0.5, -0.5)],
        [(-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)],
        [(-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (-0.5, -0.5, 0.5)],
        [(0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5)],
    ]
    tris = []
    for face in faces:
        tris.extend(sdf3d.triangulate(face))
    assert len(tris) == 12
    tris = sdf3d.orient_outward(tris)
    assert sdf3d.mesh_sdf((0, 0, 0), tris) < 0
    assert sdf3d.mesh_sdf((2, 0, 0), tris) > 0
    # A point just outside the +x face is about 0.25 away.
    assert abs(sdf3d.mesh_sdf((0.75, 0, 0), tris) - 0.25) < 1e-6
    # Closest point on the unit triangle to a point above its interior is the projection.
    q = sdf3d.closest_on_triangle((0.25, 0.25, 2), (0, 0, 0), (1, 0, 0), (0, 1, 0))
    assert math.dist(q, (0.25, 0.25, 0)) < 1e-6
