"""Combined scene: one preset switches every sim onto a shared bus, camera and composite.

Realtime keeps every participating sim in the clock and draws them. Showcase is the same
arrangement at bake settings with a reduced cache. Resolutions are lower than the single-sim
presets so the five solvers fit in one frame; nothing is bypassed.
"""

from __future__ import annotations


CAM = (0.9, 0.42, 3.4, 42.0)  # azimuth, elevation, distance, fov
TARGET = (0.0, 0.9, 0.0)
LIGHT = (0.45, 0.85, 0.35)


class Scenes:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp

    def OnStep(self, clock_ext) -> None:
        # Step bypasses a render when Cookrender is off. Turn those back on so the one
        # composite cook is the picture, and the solvers are not raymarched twice.
        for path in ("/sims/liquid/render", "/sims/particles/beauty_render", "/sims/particles/beauty",
                     "/sims/cloth/cloth_render", "/sims/destruction/pieces_render", "/sims/fluid3d/render"):
            node = op(path)  # noqa: F821
            if node is not None:
                node.bypass = False
        picture = self.ownerComp.op("picture_out")
        if picture is not None:
            picture.cook(force=True)

    def Apply(self, name: str = "realtime") -> dict:
        name = str(name or "realtime")
        if name == "showcase":
            return self._combined(bake=True)
        return self._combined(bake=False)

    def _cam(self, comp) -> None:
        p = comp.par
        for par, value in zip(("Camaz", "Camel", "Camdist", "Camfov"), CAM):
            if hasattr(p, par):
                setattr(p, par, value)
        for par, value in zip(("Targetx", "Targety", "Targetz"), TARGET):
            if hasattr(p, par):
                setattr(p, par, value)
        for par, value in zip(("Lightx", "Lighty", "Lightz"), LIGHT):
            if hasattr(p, par):
                setattr(p, par, value)

    def _combined(self, bake: bool) -> dict:
        fluid = op("/sims/fluid3d")  # noqa: F821
        destruction = op("/sims/destruction")  # noqa: F821
        particles = op("/sims/particles")  # noqa: F821
        liquid = op("/sims/liquid")  # noqa: F821
        cloth = op("/sims/cloth")  # noqa: F821
        clock = op.simclock  # noqa: F821
        bus = op.simbus3d  # noqa: F821
        # Smoke world. Liquid's own preset would replace this grid; the tank is placed in it instead.
        bus.SetGrid(32 if not bake else 48, (-1.0, 0.0, -1.0, 1.0, 2.0, 1.0))
        for wall, closed in (("Wallleft", True), ("Wallright", True), ("Wallbottom", True),
                             ("Walltop", False), ("Wallfront", True), ("Wallback", True)):
            setattr(bus.par, wall, closed)
        bus.par.Spongewidth = 0.35
        fluid.par.Solver = "jacobi"
        fluid.par.Jacobi = 4 if not bake else 6
        fluid.par.Raysteps = 4 if not bake else 12
        fluid.par.Shadowsteps = 2
        fluid.par.Renderw = 480
        fluid.par.Renderh = 270
        fluid.par.Cookrender = False
        fluid.par.Detail = 0.0
        fluid.par.Maccormack = 0
        # A previous preset (the cloth showcase) leaves a wind on this solver. The combined
        # scene's plume is the fire emitter only.
        fluid.par.Windx = 0.0
        fluid.par.Windy = 0.0
        fluid.par.Windz = 0.0
        fluid.par.Winddrag = 0.0
        self._cam(fluid)
        demo = op("/sims/demo3d")  # noqa: F821
        if hasattr(demo.par, "Emitterx"):
            demo.par.Emitterx = -0.55
            demo.par.Emittery = 0.25
            demo.par.Emitterz = 0.0
            demo.par.Flicker = 0.0
        destruction.par.Seed = 1
        destruction.par.Pieces = 32 if not bake else 48
        if hasattr(destruction.par, "Debrisn"):
            destruction.par.Debrisn = 24 if not bake else 32
        destruction.par.Pattern = "clustered"
        destruction.par.Breakmode = "impact"
        destruction.Release()
        info = destruction.Arm()
        # Fluid drag readbacks the whole velocity volume on the CPU. The combined frame cannot
        # afford that sync; pieces still block the fluid and the liquid through the obstacle field.
        destruction.par.Fluiddrag = 0
        world = destruction.ext.Destruction.world
        if world is not None and not bake:
            # 32 pieces with two impulse iterations is most of the frame. One iteration
            # still breaks the wall; the bake keeps the fuller solve.
            world.iterations = 1
            world.position_iters = 1
        destruction.par.Cookrender = True
        destruction.par.Renderw = 480
        destruction.par.Renderh = 270
        # Arm switches the fluid to the 64³ destruction preset. Put the shared grid back.
        bus.SetGrid(32 if not bake else 48, (-1.0, 0.0, -1.0, 1.0, 2.0, 1.0))
        fluid.par.Solver = "jacobi"
        fluid.par.Jacobi = 4 if not bake else 6
        fluid.par.Raysteps = 4 if not bake else 12
        fluid.par.Renderw = 480
        fluid.par.Renderh = 270
        fluid.par.Cookrender = False
        # Arm, and any preset that ran in this process, can leave combustion and wind
        # off the fresh-build defaults. The plume, the debris it carries, and the banner
        # all follow those numbers, so both passes have to start from this list.
        for name, value in (
            ("Velocitydecay", 0.4), ("Smokedecay", 0.35), ("Fueldecay", 0.4),
            ("Burnrate", 5.0), ("Burnheat", 3.5), ("Burnsmoke", 0.5), ("Ignition", 0.12),
            ("Cooling", 0.9), ("Expansion", 0.08), ("Maxtemp", 6.0),
            ("Buoyancy", 1.6), ("Weight", 0.2), ("Vorticity", 1.2),
            ("Windx", 0.0), ("Windy", 0.0), ("Windz", 0.0), ("Winddrag", 0.0),
            ("Maxspeed", 6.0), ("Smokedensity", 4.5), ("Fireintensity", 3.5),
            ("Whitehot", 2.8), ("Detail", 0.0), ("Detailfreq", 10.0), ("Detailadvect", 0.35),
        ):
            setattr(fluid.par, name, value)
        particles.par.Resx = 128 if not bake else 256
        particles.par.Resy = 128 if not bake else 256
        particles.par.Renderw = 480
        particles.par.Renderh = 270
        particles.par.Pointsize = 6.0
        # Arm leaves a 2.5 s life. The showcase is 3 s, and the late still should still show debris.
        particles.par.Lifemin = 4.0
        particles.par.Lifemax = 8.0
        particles.par.Bus = "3d"
        particles.par.Emitter = "tracer"
        particles.par.Velmode = "bus"
        particles.par.Drag = 0.4
        particles.par.Gravity = 2.0
        particles.par.Buoyancy = 0.0
        particles.par.Turbulence = 0.2
        particles.par.Turbfreq = 3.0
        particles.par.Maxspeed = 30.0
        particles.par.Radius = 0.0
        particles.par.Restitution = 0.3
        particles.par.Friction = 0.4
        particles.par.Particlemass = 1.0
        particles.par.Density = 1.0
        particles.par.Smokefrac = 0.2
        particles.par.Threshold = 0.05
        particles.par.Emitvar = 0.3
        particles.par.Couplingstrength = 0
        # Heat splat is off too. The default 0.02 would deposit debris into the smoke through
        # an order-sensitive 3D splat, and the cloth then diverges at the break.
        particles.par.Srcheat = 0
        particles.par.Cookrender = False
        particles.par.Rendermode = "composite"
        self._cam(particles)
        # Tank in the lower corner, opposite the fire. extra_solids makes pieces displace it.
        liquid.par.Mode = "tank"
        liquid.par.Resx = 256
        liquid.par.Resy = 128
        liquid.par.Spacing = 0.04 if not bake else 0.028
        liquid.par.Pcgiters = 2 if not bake else 8
        liquid.par.Pcgsmooth = 1
        liquid.par.Pcgcoarse = 12
        liquid.par.Flip = 0.4
        liquid.par.Spherer = 0.0
        liquid.par.Boxenable = True
        liquid.par.Boxx0 = 0.15
        liquid.par.Boxy0 = 0.0
        liquid.par.Boxz0 = -0.55
        liquid.par.Boxx1 = 0.95
        liquid.par.Boxy1 = 0.35
        liquid.par.Boxz1 = 0.55
        # Phase 6's free-surface solve, on this shared grid. Pcgiters does not shorten the
        # unrolled chain; the bake and the realtime scene both run the eight iterations.
        liquid.par.Project = True
        liquid.par.Raysteps = 4 if not bake else 8
        liquid.par.Renderw = 480
        liquid.par.Renderh = 270
        liquid.par.Cookrender = False
        liquid.par.Gravityx = 0.0
        liquid.par.Gravityy = -9.81
        liquid.par.Gravityz = 0.0
        self._cam(liquid)
        extra = liquid.op("extra_solids")
        if extra is not None:
            extra.par.top = destruction.op("obstacles3d")
        liquid.Bind()
        # Bind publishes the free surface as a solid. A resumed step rebuilds that surface from
        # saved particles and the pressure solve does not match the continuous bake. Pieces still
        # move the liquid through its own obstacle input. The smoke does not treat the water as a wall.
        bus.Unpublish(liquid.op("phi_out").path)
        bus.Unpublish(liquid.op("sphere").path)
        # Bind drops the demo emitter so a liquid bake stays clock-free. Flicker is 0, so this
        # fire is a static gaussian and a resumed bake still matches.
        bus.Publish("sources", "/sims/demo3d/emitter_sources")
        bus.Publish("forces", "/sims/demo3d/emitter_forces")
        cloth.par.Resx = 32 if not bake else 48
        cloth.par.Resy = 24 if not bake else 32
        cloth.par.Substeps = 4 if not bake else 4
        cloth.par.Selfiters = 0
        cloth.par.Plane = "xy"
        cloth.par.Pin = "top"
        cloth.par.Originx = -0.15
        # Low enough that the hanging sheet sits in the plume. At the ceiling the
        # smoke has not arrived by the time the pieces have already settled.
        cloth.par.Originy = 0.95
        cloth.par.Originz = 0.15
        cloth.par.Sizex = 0.9
        cloth.par.Sizey = 0.7
        cloth.par.Air = 24.0
        cloth.par.Drag = 1.6
        cloth.par.Couple = False
        cloth.par.Cookrender = False
        cloth.par.Collide = True
        cloth.par.Ground = 0.0
        # Mechanical defaults. A cloth showcase leaves a thinner, stiffer sheet and a
        # collision sphere, and the banner deflection follows those.
        cloth.par.Density = 0.25
        cloth.par.Stretch = 2.0e-5
        cloth.par.Shear = 2.0e-5
        cloth.par.Bend = 5.0e-4
        cloth.par.Damping = 0.6
        cloth.par.Thickness = 0.008
        cloth.par.Friction = 0.4
        cloth.par.Tear = 0.35
        cloth.par.Hashres = 32
        cloth.par.Gravityx = 0.0
        cloth.par.Gravityy = -9.81
        cloth.par.Gravityz = 0.0
        cloth.par.Spherer = 0.0
        self._cam(cloth)
        cloth.ext.Cloth._wind_bus(True)
        cloth.PublishJet(False)
        cloth.PublishForce(False)
        cloth.PublishSphere(False)
        cloth.RebuildMesh()
        # A previous bake leaves SetInit pointed at the cache. The next reset would resume
        # that state instead of building this preset, and the banner then never hangs in the plume.
        for comp in (destruction, liquid, fluid, particles, cloth):
            comp.SetInit(None)
        clock.par.Substeps = 1
        clock.par.Fps = 60
        clock.par.Seed = 0
        clock.par.Mode = "realtime"
        clock.par.Running = True
        # Liquid steps before the smoke. On a reset the fluid used to read whatever surface the
        # previous run had left in phi, so two bakes of the same scene diverged in the pressure
        # solve. Stepping the liquid first writes this run's surface before that solve.
        clock.par.Sims = "/sims/destruction /sims/liquid /sims/fluid3d /sims/particles /sims/cloth"
        clock.AddListener(self.ownerComp)
        self.ownerComp.par.Preset = "showcase" if bake else "realtime"
        return {
            "preset": self.ownerComp.par.Preset.eval(),
            "sims": str(clock.par.Sims.eval()),
            "res": list(bus.Res),
            "pieces": int(destruction.par.Pieces.eval()),
            "particles": [int(particles.par.Resx.eval()), int(particles.par.Resy.eval())],
            "fracture": info,
        }
