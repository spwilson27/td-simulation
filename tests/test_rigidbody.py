"""Drop, stack, energy, glue, and the float32 state round trip."""

import numpy as np

from fracture import box_solid, fracture, neighboring_pairs
from rigidbody import World, quat_matrix


def test_quaternion_rotates_x_toward_y():
    # 90 degrees about +z.
    s = np.sin(np.pi / 4)
    q = np.array([0.0, 0.0, s, s])
    assert np.allclose(quat_matrix(q) @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-8)


def test_box_drops_and_rests_without_gaining_energy():
    world = World(restitution=0.0, friction=0.8, slop=2e-4, iterations=30, position_iters=8)
    half = 0.1
    world.add_box((0.0, 1.0, 0.0), (half, half, half), mass=2.0)
    dt = 1.0 / 120.0
    start = world.energy()
    peak = start
    for _ in range(240):
        world.step(dt)
        peak = max(peak, world.energy())
        assert world.min_vertex_y() >= -1e-3
    body = world.bodies[0]
    assert abs(float(body.pos[1]) - half) < 2e-3
    assert float(np.linalg.norm(body.lin)) < 0.05
    assert peak <= start * 1.002 + 1e-4
    assert world.energy() <= start + 1e-3


def test_stack_stays_up():
    world = World(restitution=0.0, friction=0.9, slop=2e-4, iterations=32, position_iters=10, sleep_after=30)
    half = 0.1
    for i in range(3):
        world.add_box((0.0, half + i * 2 * half, 0.0), (half, half, half), mass=1.0)
    dt = 1.0 / 120.0
    for _ in range(360):
        world.step(dt)
        assert world.min_vertex_y() >= -1e-3
    for i, body in enumerate(world.bodies):
        expect = half + i * 2 * half
        assert abs(float(body.pos[1]) - expect) < 3e-3, (i, body.pos[1], expect)
        assert abs(float(body.pos[0])) < 2e-3 and abs(float(body.pos[2])) < 2e-3
    assert world.max_penetration() < 1.5e-3


def test_glue_breaks_and_state_roundtrip_is_exact():
    world = World(restitution=0.05, friction=0.5, iterations=16)
    a = world.add_box((0.0, 0.2, 0.0), (0.1, 0.1, 0.1), mass=1.0)
    b = world.add_box((0.22, 0.2, 0.0), (0.1, 0.1, 0.1), mass=1.0)
    world.add_glue(a, b, (0.11, 0.2, 0.0), (1.0, 0.0, 0.0), threshold=0.05)
    world.bodies[b].lin[0] = 3.0
    broke = False
    for _ in range(40):
        world.step(1.0 / 60.0)
        if world.glue_broken():
            broke = True
            break
    assert broke
    world.quantize()
    world.bodies[0].sleep_steps = 3
    world.glues[0]._emitted = True
    packed = world.pack_state()
    snap = world.snapshot()
    world.bodies[0].pos[1] = 5.0
    world.bodies[0].sleep_steps = 0
    world.glues[0].broken = False
    world.glues[0]._emitted = False
    world.load_state(packed)
    assert np.allclose(world.bodies[0].pos, snap["bodies"][0][0], atol=0)
    assert world.bodies[0].sleep_steps == 3
    assert world.glues[0].broken and world.glues[0]._emitted
    again = world.pack_state()
    assert packed.tobytes() == again.tobytes()


def test_sphere_contacts_rest_on_the_ground():
    # The live wall uses this broadphase. The ground test is the mesh bottom, not the
    # bounding sphere, so a box of half-extent 0.1 rests at y = 0.1.
    world = World(mesh_contacts=False, iterations=8, position_iters=4, restitution=0.0)
    world.add_box((0.0, 1.0, 0.0), (0.1, 0.1, 0.1), mass=1.0)
    for _ in range(180):
        world.step(1.0 / 60.0)
    body = world.bodies[0]
    assert abs(float(body.pos[1]) - 0.1) < 2e-3
    assert float(np.linalg.norm(body.lin)) < 0.05


def test_fracture_pieces_do_not_fall_through():
    solid = box_solid((0.0, 0.45, 0.0), (0.2, 0.45, 0.2))
    pieces = fracture(solid, 8, seed=1)
    # Ten position iterations clear a fast impact down to the solver slop. Four leave about 3mm.
    world = World(restitution=0.1, friction=0.6, iterations=12, position_iters=10)
    for piece in pieces:
        world.add_piece(piece)
    for i, j, point, normal in neighboring_pairs(pieces):
        world.add_glue(i, j, point, normal, threshold=0.4)
    world.explosion((0.0, 0.7, 0.0), radius=0.6, strength=1.5)
    for _ in range(90):
        world.step(1.0 / 60.0)
        assert world.min_vertex_y() >= -2e-3
    assert world.glue_broken() >= 1
