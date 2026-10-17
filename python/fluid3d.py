"""3D smoke/fire solver extension. The network is built by build_fluid3d.py.

Same clock/baker contract as Fluid2D: BeginStep, Step, StateTops, StateFormats, SetInit, Params.

Persistent state is a pair of Cache TOPs per field, ping-ponged through a Select. A Feedback TOP does not
capture 3D textures on this GPU (it passes its input through), which is why this is not the 2D pattern.
Public *_out selects point at the cache written this step; an inactive Cache TOP does not recook the solver.
"""

from __future__ import annotations

import hashlib
import math

STATE = {"vel": "rgba16float", "scalars": "rgba16float", "pressure": "mono32float"}

PHYSICS_PARS = (
    "Velocitydecay", "Smokedecay", "Fueldecay", "Maccormack", "Burnrate", "Burnheat", "Burnsmoke", "Ignition",
    "Cooling", "Expansion", "Maxtemp", "Buoyancy", "Weight", "Vorticity", "Windx", "Windy", "Windz", "Winddrag",
    "Maxspeed", "Jacobi", "Smooth", "Coarse", "Coarsegain", "Vcycles",
)

PRESETS = {
    # Realtime stays on Jacobi so the 128³ preview holds 30 fps. Bake uses the V-cycle.
    "preview": {"res": 96, "substeps": 1, "smooth": 4, "coarse": 24, "vcycles": 1, "rays": 36, "shadow": 12,
                "solver": "jacobi", "jacobi": 12},
    "realtime": {"res": 128, "substeps": 1, "smooth": 4, "coarse": 32, "vcycles": 1, "rays": 48, "shadow": 16,
                 "solver": "jacobi", "jacobi": 16},
    "bake": {"res": 256, "substeps": 2, "smooth": 14, "coarse": 40, "vcycles": 4, "rays": 56, "shadow": 16,
             "solver": "multigrid", "jacobi": 24},
    "high": {"res": 384, "substeps": 1, "smooth": 14, "coarse": 40, "vcycles": 4, "rays": 48, "shadow": 12,
             "solver": "multigrid", "jacobi": 16},
}


class Fluid3D:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self._reset = False
        self._read_slot = 0

    def _layout(self):
        import cache  # noqa: PLC0415

        return cache.atlas_layout(list(op.simbus3d.Res))  # noqa: F821

    def AtlasCols(self):
        return self._layout()["cols"]

    def AtlasRows(self):
        return self._layout()["rows"]

    def AtlasWidth(self):
        return self._layout()["width"]

    def AtlasHeight(self):
        return self._layout()["height"]

    def _slot(self) -> int:
        return int(self.ownerComp.fetch("slot", 0) or 0)

    def BeginStep(self, reset: bool) -> None:
        self._reset = bool(reset)
        slot = 0 if reset else self._slot()
        self._read_slot = slot
        for field in STATE:
            sel = self.ownerComp.op("src_" + field)
            if reset:
                sel.par.top = self.ownerComp.op("init_" + field).path
            else:
                sel.par.top = self.ownerComp.op(f"hold_{field}_{slot}").path
        solver = str(self.ownerComp.par.Solver.eval())
        cycles = int(self.ownerComp.par.Vcycles.eval())
        use_mg = solver == "multigrid"
        self.ownerComp.op("pressure_sel").par.index = 0 if not use_mg else min(max(cycles, 1), 4)
        self.ownerComp.op("jacobi").bypass = use_mg
        for child in self.ownerComp.children:
            prefix = child.name[:2]
            if prefix == "a_":
                child.bypass = not use_mg
            elif prefix == "b_":
                child.bypass = not (use_mg and cycles >= 2)
            elif prefix == "c_":
                child.bypass = not (use_mg and cycles >= 3)
            elif prefix == "d_":
                child.bypass = not (use_mg and cycles >= 4)

    def Step(self) -> None:
        for field in STATE:
            self.ownerComp.op(field + "_state").cook()
        write = 0 if self._reset else 1 - self._read_slot
        for field in STATE:
            hold = self.ownerComp.op(f"hold_{field}_{write}")
            hold.par.replace = True
            hold.par.replaceindex = 0
            hold.par.active = True
            hold.cook(force=True)
            hold.par.active = False
            self.ownerComp.op(field + "_out").par.top = hold.path
        self.ownerComp.store("slot", write)
        self._reset = False
        if self.ownerComp.par.Cookrender.eval():
            self.ownerComp.op("shadow").cook()
            self.ownerComp.op("render_out").cook()

    def StateTops(self) -> dict:
        return {field: self.ownerComp.op(field + "_atlas") for field in STATE}

    def StateFormats(self) -> dict:
        return dict(STATE)

    def SetInit(self, paths: dict | None = None) -> None:
        for field in STATE:
            load = self.ownerComp.op("load_" + field)
            choose = self.ownerComp.op("init_" + field)
            if paths:
                load.par.file = str(paths[field])
                load.cook(force=True)
                self.ownerComp.op("decode_" + field).cook(force=True)
                choose.par.index = 1
            else:
                choose.par.index = 0

    def Params(self) -> dict:
        p = self.ownerComp.par
        bus, clock = op.simbus3d, op.simclock  # noqa: F821
        shaders = sorted((d.name, d.text) for d in self.ownerComp.findChildren(type=textDAT, depth=1))  # noqa: F821
        digest = hashlib.sha256(repr(shaders).encode("utf-8")).hexdigest()[:16]
        return {
            "sim": "fluid3d",
            "solver": str(p.Solver.eval()),
            "physics": {name: getattr(p, name).eval() for name in PHYSICS_PARS},
            "res": list(bus.Res),
            "bounds": list(bus.Bounds),
            "walls": [bool(getattr(bus.par, w).eval()) for w in
                      ("Wallleft", "Wallright", "Wallbottom", "Walltop", "Wallfront", "Wallback")],
            "sponge": [bus.par.Spongewidth.eval(), bus.par.Spongerate.eval()],
            "fps": clock.par.Fps.eval(),
            "substeps": int(clock.par.Substeps.eval()),
            "seed": int(clock.par.Seed.eval()),
            "inputs": bus.PublisherSettings(),
            "shaders": digest,
        }

    def ApplyPreset(self, name: str) -> None:
        preset = PRESETS[name]
        op.simbus3d.SetGrid(preset["res"])  # noqa: F821
        op.simclock.par.Substeps = preset["substeps"]  # noqa: F821
        p = self.ownerComp.par
        p.Smooth = preset["smooth"]
        p.Coarse = preset["coarse"]
        p.Vcycles = preset["vcycles"]
        p.Raysteps = preset["rays"]
        p.Shadowsteps = preset["shadow"]
        # Gain 1 is the V-cycle. The old 0.05 gain was papering over a divergence/Laplacian mismatch.
        p.Coarsegain = 1.0
        p.Solver = preset["solver"]
        p.Jacobi = preset["jacobi"]
        p.Preset = name
        demo = op("/sims/demo3d")  # noqa: F821
        if demo is not None and hasattr(demo, "RebuildMesh"):
            demo.RebuildMesh()

    def EquivalentPasses(self) -> float:
        """Fine-grid Jacobi passes with about the same voxel work as the active multigrid cycle."""
        res = float(op.simbus3d.par.Resw.eval())  # noqa: F821
        cycles = 2 if str(self.ownerComp.par.Solver.eval()) == "jacobi" else int(self.ownerComp.par.Vcycles.eval())
        if str(self.ownerComp.par.Solver.eval()) == "jacobi":
            return float(self.ownerComp.par.Jacobi.eval())
        smooth = int(self.ownerComp.par.Smooth.eval())
        coarse = int(self.ownerComp.par.Coarse.eval())
        levels = 6
        work = 0.0
        for level in range(levels):
            weight = (max(4.0, math.floor(res / (2 ** level))) / res) ** 3
            iters = coarse if level == levels - 1 else smooth * 2  # pre + post
            work += iters * weight
        return work * max(cycles, 1)

    def Diagnostics(self) -> dict:
        """Post-step speed and divergence. Reads snapshots, so it does not recook the solver."""
        import numpy as np  # noqa: PLC0415

        vel = self.ownerComp.op("vel_out").numpyArray(delayed=False)
        speed = np.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2 + vel[..., 2] ** 2)
        div = self.ownerComp.op("div_check")
        div.cook(force=True)
        d = div.numpyArray(delayed=False)[..., 0]
        finite = bool(np.isfinite(vel).all() and np.isfinite(d).all())
        return {
            "max_speed": float(speed.max()) if speed.size else 0.0,
            "mean_speed": float(speed.mean()) if speed.size else 0.0,
            "div_rms": float(np.sqrt(np.mean(d ** 2))) if d.size else 0.0,
            "finite": finite,
        }

    def _rms(self, top) -> float:
        import numpy as np  # noqa: PLC0415

        top.cook(force=True)
        data = top.numpyArray(delayed=False)[..., 0]
        return float(np.sqrt(np.mean(np.square(data))))

    def PressureReport(self) -> dict:
        """Time multigrid against Jacobi at matched wall time and report residual / divergence reduction."""
        import time  # noqa: PLC0415

        p = self.ownerComp.par
        old = int(p.Jacobi.eval())
        jac = self.ownerComp.op("jacobi")
        mg = self.ownerComp.op("a_post0")
        was = bool(jac.bypass)
        jac.bypass = False
        mg.cook(force=True)
        p.Jacobi = 8
        jac.cook(force=True)
        t0 = time.perf_counter()
        mg.cook(force=True)
        mg_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        jac.cook(force=True)
        per = (time.perf_counter() - t0) / 8.0
        passes = max(1, int(round(mg_s / max(per, 1e-6))))
        p.Jacobi = passes
        t0 = time.perf_counter()
        jac.cook(force=True)
        jac_s = time.perf_counter() - t0
        pre = self._rms(self.ownerComp.op("residual_pre"))
        post = self._rms(self.ownerComp.op("residual_mg"))
        jpost = self._rms(self.ownerComp.op("residual_jac"))
        div_pre = self._rms(self.ownerComp.op("divergence"))
        div_post = self._rms(self.ownerComp.op("div_check"))
        p.Jacobi = old
        jac.bypass = was
        return {
            "mg_seconds": mg_s,
            "jacobi_seconds": jac_s,
            "jacobi_passes": passes,
            "equivalent_passes": self.EquivalentPasses(),
            "residual_before": pre,
            "residual_multigrid": post,
            "residual_jacobi": jpost,
            "residual_factor": (pre / post) if post > 0 else None,
            "divergence_before": div_pre,
            "divergence_after": div_post,
            "divergence_factor": (div_pre / div_post) if div_post > 0 else None,
            "multigrid_beats_jacobi": post < jpost,
        }

    def OccupiedBbox(self, pad: int = 0) -> list:
        """Inclusive voxel box of scalar texels that are not zero. Empty smoke is the full domain."""
        import numpy as np  # noqa: PLC0415

        import cache  # noqa: PLC0415

        sc = self.ownerComp.op("scalars_out").numpyArray(delayed=False)
        res = list(op.simbus3d.Res)  # noqa: F821
        if sc is None or sc.size == 0:
            return cache.occupied_bbox([], res, pad)
        active = np.any(sc != 0, axis=-1)
        zs, ys, xs = np.nonzero(active)
        if xs.size == 0:
            return cache.occupied_bbox([], res, pad)
        return cache.occupied_bbox(
            [(int(xs.min()), int(ys.min()), int(zs.min())), (int(xs.max()), int(ys.max()), int(zs.max()))],
            res, pad,
        )

    def CropAtlas(self, field: str, bbox: list):
        """Point the bypassed crop encoder at `bbox` and return it. The caller saves that TOP."""
        import cache  # noqa: PLC0415

        shape = cache.bbox_shape(bbox)
        layout = cache.atlas_layout(shape)
        top = self.ownerComp.op("crop_" + field)
        top.bypass = False
        top.par.resolutionw.expr = ""
        top.par.resolutionh.expr = ""
        top.par.resolutionw = int(layout["width"])
        top.par.resolutionh = int(layout["height"])
        top.par.vec0valuex = int(layout["cols"])
        top.par.vec0valuey = int(layout["rows"])
        top.par.vec0valuez = int(shape[0])
        top.par.vec0valuew = int(shape[1])
        top.par.vec1valuex = int(shape[2])
        top.par.vec2valuex = int(bbox[0])
        top.par.vec2valuey = int(bbox[1])
        top.par.vec2valuez = int(bbox[2])
        top.cook(force=True)
        return top
