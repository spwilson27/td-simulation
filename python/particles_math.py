"""Pure helpers for the GPU particle system. The shaders in shaders/particles mirror these formulas.

pcg / urand match hash.glsl. step_particle matches the semi-implicit RK2 in step.glsl when temperature
and turbulence are constant (the shader evaluates them at each sample point).
"""

from __future__ import annotations

import math

_MASK = 0xFFFFFFFF


def pcg(v: int) -> int:
    v &= _MASK
    state = (v * 747796405 + 2891336453) & _MASK
    shift = (state >> 28) + 4
    word = (((state >> shift) ^ state) * 277803737) & _MASK
    return ((word >> 22) ^ word) & _MASK


def urand(x: int, y: int, salt: int) -> float:
    h = pcg((int(x) + pcg((int(y) + pcg(int(salt))) & _MASK)) & _MASK)
    return h / 4294967296.0


def sphere_point(rnd, radius: float) -> tuple[float, float, float]:
    """Uniform sample in a ball. rnd is three values in [0, 1). Matches spherePoint()."""
    z = rnd[1] * 2.0 - 1.0
    a = rnd[2] * math.tau
    rxy = math.sqrt(max(0.0, 1.0 - z * z))
    direction = (rxy * math.cos(a), rxy * math.sin(a), z)
    rad = float(radius) * max(rnd[0], 1e-4) ** (1.0 / 3.0)
    return tuple(direction[i] * rad for i in range(3))


def box_point(rnd, half_ext) -> tuple[float, float, float]:
    return tuple((rnd[i] * 2.0 - 1.0) * half_ext[i] for i in range(3))


def reaction_acceleration(vel_p, vel_f, drag: float, strength: float, mass: float, density: float,
                          cell: float, dimensions: int = 2):
    """Fluid acceleration from one particle's drag reaction.

    a = strength * mass * drag * (v_p - v_f) / (density * cell_volume). The same scaling
    multiplies a heat or smoke rate. `dimensions` is 2 or 3. Matches the deposit shaders.
    """
    volume = float(cell) ** int(dimensions)
    ratio = float(mass) / max(float(density) * volume, 1e-12)
    scale = float(strength) * ratio * float(drag)
    return tuple(scale * (float(vel_p[i]) - float(vel_f[i])) for i in range(3))


def step_particle(pos, vel, dt: float, velocity, drag: float = 0.0, gravity: float = 0.0,
                  buoyancy: float = 0.0, temperature: float = 0.0, turbulence=(0.0, 0.0, 0.0)):
    """One semi-implicit / RK2 step. `velocity(p)` is the field being followed.

    Drag is solved implicitly. Gravity, buoyancy and turbulence are sampled at the start and at the
    midpoint. With drag 0 and no extra acceleration this is exactly p + dt * v.
    """
    pos = [float(v) for v in pos]
    vel = [float(v) for v in vel]
    drag = float(drag)

    def extra(_p):
        return (float(turbulence[0]), -float(gravity) + float(buoyancy) * float(temperature) + float(turbulence[1]),
                float(turbulence[2]))

    def combine(v, h, p):
        vf = velocity(p)
        a = extra(p)
        return [(v[i] + h * (drag * vf[i] + a[i])) / (1.0 + h * drag) for i in range(3)]

    half = 0.5 * float(dt)
    v1 = combine(vel, half, pos)
    mid = [pos[i] + half * v1[i] for i in range(3)]
    v2 = combine(vel, float(dt), mid)
    p2 = [pos[i] + float(dt) * v2[i] for i in range(3)]
    return p2, v2


def collide_sphere(pos, vel, centre, radius: float, particle_radius: float = 0.0, restitution: float = 0.0,
                   friction: float = 0.0, obstacle_velocity=(0.0, 0.0, 0.0), iterations: int = 4):
    """Push a particle out of a sphere SDF and reflect its velocity. Mirrors collide() for an analytic sphere."""
    pos = [float(v) for v in pos]
    vel = [float(v) for v in vel]
    centre = [float(v) for v in centre]
    vobs = [float(v) for v in obstacle_velocity]
    for _ in range(int(iterations)):
        delta = [pos[i] - centre[i] for i in range(3)]
        dist = math.sqrt(sum(d * d for d in delta)) or 1e-8
        sd = dist - float(radius)
        if sd >= particle_radius:
            break
        n = [d / dist for d in delta]
        push = particle_radius - sd
        pos = [pos[i] + n[i] * push for i in range(3)]
        rel = [vel[i] - vobs[i] for i in range(3)]
        vn = sum(rel[i] * n[i] for i in range(3))
        if vn < 0.0:
            rel = [rel[i] - (1.0 + restitution) * vn * n[i] for i in range(3)]
            vn2 = sum(rel[i] * n[i] for i in range(3))
            vt = [rel[i] - vn2 * n[i] for i in range(3)]
            rel = [vn2 * n[i] + vt[i] * (1.0 - friction) for i in range(3)]
            vel = [vobs[i] + rel[i] for i in range(3)]
    return pos, vel
