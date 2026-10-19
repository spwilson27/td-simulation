"""GPU particle extension. State is two float textures (position+age, velocity+life) in Feedback TOPs.

The clock contract matches Fluid2D: BeginStep, Step, StateTops, StateFormats, SetInit, Params.
Public outputs are inactive Cache TOP snapshots, so a viewer or a numpy read does not advance the sim.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

STATE = {"pos": "rgba32float", "pvel": "rgba32float"}

PHYSICS = (
    "Drag", "Gravity", "Buoyancy", "Turbulence", "Turbfreq", "Maxspeed", "Radius", "Restitution", "Friction",
    "Couplingstrength", "Particlemass", "Density", "Srcheat", "Smokefrac", "Lifemin", "Lifemax", "Threshold",
)
EMIT = (
    "Emitx", "Emity", "Emitz", "Emitradius", "Boxhx", "Boxhy", "Boxhz", "Emitvelx", "Emitvely", "Emitvelz", "Emitvar",
    "Analyticx", "Analyticy", "Analyticz", "Analyticw", "Spawnx", "Spawny", "Spawnz", "Spawnvx", "Spawnvy", "Spawnvz",
)


def _ensure_python_path(comp) -> None:
    path = str(Path(comp.parent.sims.par.Root.eval()) / "python")
    if path not in sys.path:
        sys.path.insert(0, path)


class Particles:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        _ensure_python_path(ownerComp)
        import importlib  # noqa: PLC0415

        import cache  # noqa: PLC0415

        importlib.reload(cache)
        self.cache = cache
        self._sync_atlas()

    def _sync_atlas(self) -> None:
        res = op.simbus3d.Res  # noqa: F821
        layout = self.cache.atlas_layout(res)
        p = self.ownerComp.par
        p.Atlasw = int(layout["width"])
        p.Atlash = int(layout["height"])
        p.Atlascols = int(layout["cols"])
        p.Atlasrows = int(layout["rows"])

    def BeginStep(self, reset: bool) -> None:
        self._sync_atlas()
        p = self.ownerComp.par
        three = p.Bus.eval() == "3d"
        couple = float(p.Couplingstrength.eval()) != 0.0 or float(p.Srcheat.eval()) != 0.0
        show = bool(p.Cookrender.eval())
        for field in STATE:
            src = self.ownerComp.op("src_" + field)
            if reset and getattr(self, "_resume", False):
                src.par.top = self.ownerComp.op("resume_" + field).path
            elif reset:
                src.par.top = self.ownerComp.op("init_" + field).path
            else:
                src.par.top = self.ownerComp.op(field + "_out").path
            src.cook(force=True)
        if reset:
            self._resume = False
        # Bypassed branches do not pull the bus, so the fluid solver is not run once per particle TOP.
        for name, skip in (
            ("step_2d_pos", three), ("step_2d_vel", three), ("deposit_2d", three or not couple),
            ("step_3d_pos", not three), ("step_3d_vel", not three), ("deposit_3d", (not three) or not couple),
            ("splat_render", (not couple) or not three),
            ("splat_count", (not couple) or three), ("splat_scan", (not couple) or three),
            ("splat_base", (not couple) or three), ("splat_scatter", (not couple) or three),
            ("splat_sum", (not couple) or three), ("splat_part", (not couple) or three),
            ("splat_fence", (not couple) or three),
            ("splat_vol", not (three and couple)),
            ("forces_3d", not (three and couple)), ("sources_3d", not (three and couple)),
            ("beauty_render", not show), ("beauty", not show),
            ("emissive", not (show and p.Rendermode.eval() == "emissive")),
        ):
            node = self.ownerComp.op(name)
            if node is not None:
                node.bypass = skip

    def Step(self) -> None:
        # The previous image is a locked TOP, so this cook cannot recurse back into itself.
        # Locking the result is what keeps the next step from seeing a half-written texture.
        # Integrate both textures before either snapshot is replaced. The velocity shader reads
        # the position snapshot; updating it first would make a resumed step (which still reads
        # the loaded frame) disagree with a continuous one.
        for field in STATE:
            state = self.ownerComp.op(field + "_state")
            state.lock = False
            state.cook(force=True)
            state.lock = True
        for field in STATE:
            mem = self.ownerComp.op(field + "_out")
            mem.lock = False
            mem.cook(force=True)
            mem.lock = True
            # Stay locked. A later save or bus read must not cook this null again: its input
            # is the previous snapshot, so another cook would integrate a second time and the
            # file would no longer match the snapshot the next step reads.
        deposit = float(self.ownerComp.par.Couplingstrength.eval()) != 0.0 or float(self.ownerComp.par.Srcheat.eval()) != 0.0
        if deposit or self.ownerComp.par.Cookrender.eval():
            self.ownerComp.op("deposit").cook(force=True)
        if deposit:
            if self.ownerComp.par.Bus.eval() == "3d":
                self.ownerComp.op("splat_render").cook(force=True)
                self.ownerComp.op("forces_3d").cook(force=True)
                self.ownerComp.op("sources_3d").cook(force=True)
            else:
                # The fluid step already ran, against the frozen grid from the previous particle
                # step (or the grid the baker restored on resume). This step writes the next one.
                bus = op.simbus  # noqa: F821
                table = bus.op("publishers")
                for name in ("splat_count", "splat_scan", "splat_base", "splat_scatter", "splat_sum", "splat_part", "splat_fence"):
                    node = self.ownerComp.op(name)
                    if node is not None:
                        node.cook(force=True)
                # The fence TOP reads every block sum, so this one-pixel readback waits for
                # the whole grid. A full-resolution readback was the previous fence and it
                # does not fit a 1024² bus inside the frame.
                self.ownerComp.op("splat_fence").numpyArray(delayed=False)
                self.ownerComp.op("forces_2d").cook(force=True)
                self.ownerComp.op("sources_2d").cook(force=True)
                # Publish a locked copy. The fluid step runs first next frame and pulls the bus;
                # a live splat recooks from the particles and no longer matches the grid that was saved.
                for kind in ("forces", "sources"):
                    self._publish_frozen(bus, table, kind)
                bus.ext.FieldBus._refresh()
                bus.op("forces").cook(force=True)
                bus.op("sources").cook(force=True)
        if not self.ownerComp.par.Cookrender.eval():
            return
        self.ownerComp.op("render_out").cook(force=True)
        mode = self.ownerComp.par.Rendermode.eval()
        if mode == "composite":
            self.ownerComp.op("picture_out").cook(force=True)
        elif mode == "emissive":
            self.ownerComp.op("emissive_out").cook(force=True)

    def _publish_frozen(self, bus, table, kind: str) -> None:
        """Lock a copy of the splat and publish that, so the next fluid step cannot recook it."""
        live = self.ownerComp.op(kind + "_2d")
        if live is None:
            return
        import numpy as np  # noqa: PLC0415

        image = np.array(live.numpyArray(delayed=False), dtype=np.float32, copy=True)
        name = kind + "_pub"
        top = self.ownerComp.op(name)
        if top is not None and top.OPType != "scriptTOP":
            top.destroy()
            top = None
        if top is None:
            top = self.ownerComp.create(scriptTOP, name)  # noqa: F821
            top.nodeX = 400
            top.nodeY = 0 if kind == "forces" else -80
        top.par.outputresolution = "custom"
        top.par.resolutionw = int(image.shape[1])
        top.par.resolutionh = int(image.shape[0])
        top.par.format = "rgba32float"
        top.lock = True
        top.copyNumpyArray(np.ascontiguousarray(image))
        kept = False
        for row in reversed(range(1, table.numRows)):
            path = table[row, 1].val
            if table[row, 0].val != kind:
                continue
            if path == live.path or path.startswith(self.ownerComp.path) or path.endswith("/hold_" + kind):
                if not kept:
                    table[row, 1] = top.path
                    kept = True
                else:
                    table.deleteRow(row)
        if not kept:
            table.appendRow([kind, top.path])

    def StateTops(self) -> dict:
        return {field: self.ownerComp.op(field + "_out") for field in STATE}

    def StateFormats(self) -> dict:
        return dict(STATE)

    def ExportArray(self, field: str):
        """Float32 state. Movie File In does not return a 32-bit EXR of these textures, so the baker
        writes this array beside the EXR and SetInit prefers it."""
        if field not in STATE:
            return None
        top = self.ownerComp.op(field + "_out")
        if top is None:
            return None
        image = top.numpyArray(delayed=False)
        if image is None:
            return None
        import numpy as np  # noqa: PLC0415

        return np.array(image, dtype=np.float32, copy=True)

    def _hold_resume(self, field: str, image) -> None:
        name = "resume_" + field
        top = self.ownerComp.op(name)
        if top is None:
            top = self.ownerComp.create(scriptTOP, name)  # noqa: F821
            top.par.format = "rgba32float"
            top.nodeX = -200
            top.nodeY = 0 if field == "pos" else -120
        top.par.outputresolution = "custom"
        top.par.resolutionw = int(image.shape[1])
        top.par.resolutionh = int(image.shape[0])
        top.lock = True
        top.copyNumpyArray(image)

    def SetInit(self, paths: dict | None = None) -> None:
        self._resume = False
        if not paths:
            for field in STATE:
                self.ownerComp.op("init_" + field).par.index = 0
            return
        root = Path(self.ownerComp.parent.sims.par.Root.eval())
        arrays = {}
        for field in STATE:
            npy = Path(str(paths[field]))
            if not npy.is_absolute():
                npy = root / npy
            npy = npy.with_suffix(".npy")
            if not npy.is_file():
                arrays = None
                break
            import numpy as np  # noqa: PLC0415

            arrays[field] = np.ascontiguousarray(np.load(npy), dtype=np.float32)
        if arrays:
            for field, image in arrays.items():
                self._hold_resume(field, image)
            self._resume = True
            return
        for field in STATE:
            load = self.ownerComp.op("load_" + field)
            choose = self.ownerComp.op("init_" + field)
            state = self.ownerComp.op(field + "_state")
            load.par.file = str(paths[field])
            load.par.playmode = "locked"
            load.par.index.mode = ParMode.CONSTANT  # noqa: F821
            load.par.index = 0
            load.par.outputresolution = "custom"
            load.par.resolutionw = state.width
            load.par.resolutionh = state.height
            load.par.reloadpulse.pulse()
            load.cook(force=True)
            choose.par.index = 1

    def Params(self) -> dict:
        p = self.ownerComp.par
        bus = op.simbus3d if p.Bus.eval() == "3d" else op.simbus  # noqa: F821
        clock = op.simclock  # noqa: F821
        import cache  # noqa: PLC0415

        shaders = sorted((d.name, cache.text_dat_source(d)) for d in self.ownerComp.findChildren(type=textDAT, depth=1))  # noqa: F821
        digest = hashlib.sha256(repr(shaders).encode("utf-8")).hexdigest()[:16]
        res = list(bus.Res)
        return {
            "sim": "particles",
            "count": [int(p.Resx.eval()), int(p.Resy.eval())],
            "bus": p.Bus.eval(),
            "velmode": p.Velmode.eval(),
            "emitter": p.Emitter.eval(),
            "res": res,
            "bounds": list(bus.Bounds),
            "physics": {name: getattr(p, name).eval() for name in PHYSICS},
            "emit": {name: getattr(p, name).eval() for name in EMIT},
            "fps": clock.par.Fps.eval(),
            "substeps": int(clock.par.Substeps.eval()),
            "seed": int(clock.par.Seed.eval()),
            "publishers": {kind: sorted(bus.Publishers(kind)) for kind in ("sources", "forces", "obstacles")},
            "shaders": digest,
        }

    def Unbind(self) -> None:
        pairs = (
            (op.simbus, ("forces_2d", "sources_2d", "forces_pub", "sources_pub")),  # noqa: F821
            (op.simbus3d, ("forces_3d", "sources_3d")),  # noqa: F821
        )
        for bus, names in pairs:
            for name in names:
                top = self.ownerComp.op(name)
                if top is not None:
                    bus.Unpublish(top.path)

    def Bind(self) -> None:
        """Publish forces and sources on the bus selected by Bus. Replaces any previous particle publish."""
        self.Unbind()
        p = self.ownerComp.par
        if float(p.Couplingstrength.eval()) == 0.0 and float(p.Srcheat.eval()) == 0.0:
            return
        if p.Bus.eval() == "3d":
            bus = op.simbus3d  # noqa: F821
            bus.Publish("forces", self.ownerComp.op("forces_3d"))
            bus.Publish("sources", self.ownerComp.op("sources_3d"))
        else:
            bus = op.simbus  # noqa: F821
            bus.Publish("forces", self.ownerComp.op("forces_2d"))
            bus.Publish("sources", self.ownerComp.op("sources_2d"))
