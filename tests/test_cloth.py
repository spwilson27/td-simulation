"""Numpy XPBD reference: colouring, rest angles, and one substep."""

import numpy as np

from cloth import (
    KINDS,
    aero_force,
    bending_rest_angles,
    build_constraints,
    colors_are_independent,
    constraint_ends,
    edge_strains,
    grid_positions,
    inverse_mass,
    owner_color,
    rest_lengths,
    theoretical_strain,
    vertex_role,
    xpbd_substep,
)


def test_colors_share_no_vertex():
    items = build_constraints(12, 9)
    assert colors_are_independent(items)
    assert len(items) > 100
    for item in items:
        assert item["color"] == owner_color(item["kind"], item["i"], item["j"])
        assert constraint_ends(item["kind"], item["i"], item["j"], 12, 9) == (item["a"], item["b"])


def test_every_endpoint_has_one_role():
    nx, ny = 8, 6
    for kind in KINDS:
        for color in (0, 1):
            seen = {}
            for j in range(ny):
                for i in range(nx):
                    role = vertex_role(kind, color, i, j, nx, ny)
                    if role is None:
                        continue
                    sign, i0, j0 = role
                    ends = constraint_ends(kind, i0, j0, nx, ny)
                    assert ends is not None
                    assert owner_color(kind, i0, j0) == color
                    vertex = ends[0] if sign > 0 else ends[1]
                    assert vertex == (i, j)
                    assert (i, j) not in seen
                    seen[(i, j)] = (i0, j0)
            owners = {}
            for j in range(ny):
                for i in range(nx):
                    if owner_color(kind, i, j) != color:
                        continue
                    if constraint_ends(kind, i, j, nx, ny) is None:
                        continue
                    owners[(i, j)] = constraint_ends(kind, i, j, nx, ny)
            assert len(seen) == 2 * len(owners)


def test_flat_rest_angle_is_zero_and_a_fold_is_a_right_angle():
    flat = grid_positions(3, 2, (0, 0, 0), (0.2, 0.1), "xz")
    angles = bending_rest_angles(flat)
    assert angles.size > 0
    assert float(angles.max()) < 1e-6
    folded = flat.copy()
    dx = float(flat[0, 1, 0] - flat[0, 0, 0])
    folded[:, 2, 0] = folded[:, 1, 0]
    folded[:, 2, 1] = folded[:, 1, 1] + dx
    folded_angles = bending_rest_angles(folded)
    assert abs(float(np.max(folded_angles)) - np.pi / 2) < 1e-6


def test_distance_constraint_pulls_toward_rest():
    pos = np.zeros((1, 2, 3))
    pos[0, 1] = (0.2, 0.0, 0.0)
    vel = np.zeros_like(pos)
    inv = np.ones((1, 2))
    rest = pos.copy()
    bits = np.zeros((1, 2), dtype=np.int32)
    lengths = rest_lengths(0.1, 0.1)
    comp = {kind: 0.0 for kind in KINDS}
    new, _, _ = xpbd_substep(pos, vel, inv, rest, bits, 1 / 60, comp, (0, 0, 0), 0.0, lengths, air=0.0)
    dist = float(np.linalg.norm(new[0, 0] - new[0, 1]))
    assert abs(dist - 0.1) < 1e-6


def test_wind_pushes_downwind():
    normal = np.array([0.0, 0.0, 1.0])
    force = aero_force(normal, np.zeros(3), np.array([0.0, 0.0, 3.0]), 0.01, 1.2, 0.5, 12.0)
    assert force[2] > 0.0
    flipped = aero_force(-normal, np.zeros(3), np.array([0.0, 0.0, 3.0]), 0.01, 1.2, 0.5, 12.0)
    assert flipped[2] > 0.0
    assert float(np.linalg.norm(aero_force(normal, np.zeros(3), np.zeros(3), 0.01, 1.2, 0.5, 12.0))) == 0.0


def test_hanging_cloth_settles_and_strain_tracks_compliance():
    nx = ny = 6
    dx = dy = 0.05
    pos = grid_positions(nx, ny, (-0.15, 1.0, -0.15), ((nx - 1) * dx, (ny - 1) * dy), "xz")
    vel = np.zeros_like(pos)
    inv = inverse_mass(nx, ny, "corners", 0.3, dx, dy)
    rest = pos.copy()
    bits = np.zeros((ny, nx), dtype=np.int32)
    lengths = rest_lengths(dx, dy)
    comp = {kind: 1e-4 if "bend" not in kind else 1e-3 for kind in KINDS}
    dt = 1 / 240
    for _ in range(400):
        pos, vel, bits = xpbd_substep(
            pos, vel, inv, rest, bits, dt, comp, (0, -9.81, 0), 6.0, lengths, air=0.0,
            ground=0.0, thickness=0.0, pin="corners")
    assert np.isfinite(pos).all()
    assert float(np.max(np.linalg.norm(vel, axis=2))) < 1.0
    # Long-range attachments plus stretch keep every edge inside 2%.
    assert float(edge_strains(pos, lengths, nx, ny).max()) < 0.02


def test_substep_count_reaches_a_similar_rest_shape():
    def settle(substeps):
        nx = ny = 5
        pos = grid_positions(nx, ny, (0, 1, 0), (0.4, 0.4), "xz")
        vel = np.zeros_like(pos)
        dx = 0.4 / 4
        inv = inverse_mass(nx, ny, "corners", 0.3, dx, dx)
        rest = pos.copy()
        bits = np.zeros((ny, nx), dtype=np.int32)
        lengths = rest_lengths(dx, dx)
        comp = {kind: 1e-4 if "bend" not in kind else 1e-3 for kind in KINDS}
        frames = 40
        dt = (1 / 60) / substeps
        for _ in range(frames * substeps):
            pos, vel, bits = xpbd_substep(
                pos, vel, inv, rest, bits, dt, comp, (0, -9.81, 0), 3.0, lengths, air=0.0, pin="corners")
        return pos

    a = settle(2)
    b = settle(4)
    assert float(np.max(np.abs(a - b))) < 0.05


def test_tear_breaks_one_edge_and_stays_broken():
    pos = np.zeros((1, 2, 3))
    pos[0, 1, 0] = 0.5
    vel = np.zeros_like(pos)
    inv = np.ones((1, 2))
    rest = pos.copy()
    rest[0, 1, 0] = 0.1
    bits = np.zeros((1, 2), dtype=np.int32)
    lengths = rest_lengths(0.1, 0.1)
    comp = {kind: 0.0 for kind in KINDS}
    _, _, bits = xpbd_substep(pos, vel, inv, rest, bits, 1 / 60, comp, (0, 0, 0), 0.0, lengths, tear=0.2)
    assert int(bits[0, 0]) & 1
    far = pos.copy()
    far[0, 1, 0] = 0.5
    new, _, bits2 = xpbd_substep(far, vel, inv, rest, bits, 1 / 60, comp, (0, 0, 0), 0.0, lengths, tear=0.2)
    assert abs(float(new[0, 1, 0]) - 0.5) < 1e-6
    assert int(bits2[0, 0]) & 1


def test_ground_keeps_vertices_above_the_plane():
    pos = np.zeros((2, 2, 3))
    pos[:, :, 1] = -0.2
    vel = np.zeros_like(pos)
    inv = np.ones((2, 2))
    rest = pos.copy()
    bits = np.zeros((2, 2), dtype=np.int32)
    lengths = rest_lengths(0.1, 0.1)
    comp = {kind: 1e-5 for kind in KINDS}
    new, _, _ = xpbd_substep(
        pos, vel, inv, rest, bits, 1 / 60, comp, (0, -9.81, 0), 0.0, lengths,
        ground=0.0, thickness=0.01, friction=0.5)
    assert float(new[:, :, 1].min()) >= 0.01 - 1e-6
