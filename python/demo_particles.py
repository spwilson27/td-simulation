"""Presets for the particle demos. Pulled by the Apply pulse on /sims/demo_particles."""

from __future__ import annotations


def _particles():
    return op("/sims/particles")  # noqa: F821


class DemoParticles:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp

    def ApplyDemo(self, name: str | None = None) -> None:
        name = name or self.ownerComp.par.Demo.eval()
        if name == "sparks3d":
            self._sparks3d()
        else:
            self._embers2d()

    def _common(self, particles) -> None:
        p = particles.par
        p.Resx = 1024
        p.Resy = 1024
        p.Velmode = "bus"
        p.Drag = 3.5
        p.Gravity = 0.2
        p.Buoyancy = 0.55
        p.Turbulence = 0.4
        p.Turbfreq = 2.4
        p.Maxspeed = 12
        p.Radius = 0
        p.Restitution = 0.3
        p.Friction = 0.5
        p.Lifemin = 0.45
        p.Lifemax = 1.1
        p.Threshold = 0.08
        p.Couplingstrength = 1.0
        p.Particlemass = 1.0
        p.Density = 1.0
        p.Srcheat = 0.02
        p.Smokefrac = 0.25
        p.Cookrender = True
        p.Pointsize = 4
        p.Colorscale = 1.4
        p.Camaz = 0.7
        p.Camel = 0.38
        p.Camdist = 4.2
        p.Camfov = 38
        p.Targetx = 0
        p.Targety = 0.95
        p.Targetz = 0

    def _embers2d(self) -> None:
        particles = _particles()
        self._common(particles)
        p = particles.par
        p.Bus = "2d"
        p.Emitter = "fire"
        p.Emitx = 0
        p.Emity = 0.22
        p.Emitz = 0
        p.Emitradius = 0.14
        p.Emitvelx = 0.0
        p.Emitvely = 0.0
        p.Emitvelz = 0.0
        p.Emitvar = 0.12
        p.Rendermode = "composite"
        p.Renderw = 1024
        p.Renderh = 1024
        p.Pointsize = 5
        particles.Bind()
        fluid = op("/sims/fluid2d")  # noqa: F821
        fluid.allowCooking = True
        fluid.ApplyPreset("realtime")
        fluid.par.Vorticity = 2.5
        fluid.par.Cookrender = True
        op.simclock.par.Seed = 0  # noqa: F821
        op.simclock.par.Sims = "/sims/particles /sims/fluid2d"  # noqa: F821

    def _sparks3d(self) -> None:
        particles = _particles()
        self._common(particles)
        p = particles.par
        p.Bus = "3d"
        p.Emitter = "fire"
        p.Emitx = 0
        p.Emity = 0.2
        p.Emitz = 0
        p.Emitradius = 0.18
        p.Emitvelx = 0.0
        p.Emitvely = 0.0
        p.Emitvelz = 0.0
        p.Emitvar = 0.15
        p.Gravity = 0.35
        p.Rendermode = "composite"
        p.Renderw = 1280
        p.Renderh = 720
        p.Pointsize = 6
        particles.Bind()
        fluid = op("/sims/fluid3d")  # noqa: F821
        fluid.allowCooking = True
        fluid.par.Cookrender = True
        op.simclock.par.Sims = "/sims/particles /sims/fluid3d"  # noqa: F821
