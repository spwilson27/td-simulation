import math

import pytest

from particles_math import box_point, collide_sphere, pcg, reaction_acceleration, sphere_point, step_particle, urand


def test_pcg_is_stable_and_in_range():
    a = pcg(1)
    assert a == pcg(1)
    assert 0 <= a < 2**32
    assert pcg(1) != pcg(2)
    assert 0.0 <= urand(3, 4, 5) < 1.0
    assert urand(3, 4, 5) == urand(3, 4, 5)
    assert urand(3, 4, 5) != urand(3, 4, 6)


def test_sphere_and_box_samples_stay_inside():
    for i in range(40):
        rnd = (urand(i, 1, 2), urand(i, 3, 4), urand(i, 5, 6))
        p = sphere_point(rnd, 0.4)
        assert math.sqrt(sum(c * c for c in p)) <= 0.4 + 1e-6
        b = box_point(rnd, (0.2, 0.3, 0.4))
        assert abs(b[0]) <= 0.2 + 1e-6
        assert abs(b[1]) <= 0.3 + 1e-6
        assert abs(b[2]) <= 0.4 + 1e-6


def test_constant_velocity_is_exact():
    velocity = lambda _p: (1.0, 0.25, 0.0)
    pos, vel = [-0.4, 0.6, 0.0], [1.0, 0.25, 0.0]
    dt = 1.0 / 60.0
    for _ in range(30):
        pos, vel = step_particle(pos, vel, dt, velocity, drag=0.0)
    assert vel == [1.0, 0.25, 0.0]
    assert pos[0] == pytest.approx(-0.4 + 30 * dt)
    assert pos[1] == pytest.approx(0.6 + 30 * dt * 0.25)
    assert pos[2] == pytest.approx(0.0)


def test_drag_relaxes_toward_the_field():
    velocity = lambda _p: (2.0, 0.0, 0.0)
    pos, vel = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    for _ in range(20):
        pos, vel = step_particle(pos, vel, 1.0 / 60.0, velocity, drag=8.0)
    assert vel[0] > 1.5
    assert abs(vel[1]) < 1e-6


def test_collision_ejects_a_particle_from_a_sphere():
    pos, vel = collide_sphere([0.0, 0.2, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0], radius=0.5,
                              restitution=0.5, friction=0.0, obstacle_velocity=(0.2, 0.0, 0.0))
    dist = math.sqrt(sum(c * c for c in pos))
    assert dist >= 0.5 - 1e-6
    assert vel[1] > 0.0  # was moving inward, bounced outward


def test_reaction_stays_small_for_light_embers():
    cell = 2.0 / 1024.0
    one = reaction_acceleration((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), drag=3.5, strength=1.0,
                                mass=1e-9, density=1.0, cell=cell, dimensions=2)
    # A few hundred embers in one cell still weigh less than the fluid there.
    assert abs(one[0]) * 400 < 3.5
    stronger = reaction_acceleration((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), drag=3.5, strength=10.0,
                                     mass=1e-9, density=1.0, cell=cell, dimensions=2)
    assert abs(stronger[0]) > abs(one[0]) * 9.0
