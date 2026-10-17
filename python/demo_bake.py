"""Showcase bake: a 256³ fire rising around the demo obstacles, stored as a reduced cache.

Apply() points the clock and the baker at fluid3d, selects the multigrid bake preset (substeps, MacCormack,
vorticity and combustion stay on), and turns on keyframe + scalar-crop storage. It does not start the bake;
pulse the baker, or call Bake(), and pump Tick() from outside so one execute does not run for minutes.
"""

from __future__ import annotations


class DemoBake:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp

    def Apply(self) -> None:
        p = self.ownerComp.par
        fluid = op("/sims/fluid3d")  # noqa: F821
        demo = op("/sims/demo3d")  # noqa: F821
        baker = op("/sims/baker")  # noqa: F821
        clock = op.simclock  # noqa: F821
        fluid.ApplyPreset("bake")
        fluid.par.Maccormack = 0.65
        fluid.par.Vorticity = 1.4
        fluid.par.Buoyancy = 1.8
        fluid.par.Burnrate = 5.5
        fluid.par.Expansion = 0.1
        fluid.par.Detail = 0.0
        demo.par.Emitterradius = 0.32
        demo.par.Smoke = 0.7
        demo.par.Heat = 5.2
        demo.par.Fuel = 8.0
        demo.par.Lift = 2.6
        if hasattr(demo, "RebuildMesh"):
            demo.RebuildMesh()
        clock.par.Sims = "/sims/fluid3d"
        clock.par.Mode = "bake"
        baker.par.Sim = "/sims/fluid3d"
        baker.par.Groupsims = ""
        baker.par.Cachedir = str(p.Cachedir.eval())
        baker.par.Framestart = 1
        baker.par.Frameend = int(p.Frames.eval())
        baker.par.Warmup = 0
        baker.par.Resume = True
        baker.par.Asyncsave = False
        baker.par.Keyinterval = int(p.Keyinterval.eval())
        baker.par.Playbackfields = "scalars"
        baker.par.Crop = bool(p.Crop.eval())
        baker.par.Croppad = 2
        fluid.par.Cookrender = False
