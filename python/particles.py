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
            src.par.top = (self.ownerComp.op("init_" + field) if reset else self.ownerComp.op(field + "_out")).path
            src.cook(force=True)
        # Bypassed branches do not pull the bus, so the fluid solver is not run once per particle TOP.
        for name, skip in (
            ("step_2d_pos", three), ("step_2d_vel", three), ("deposit_2d", three or not couple),
            ("step_3d_pos", not three), ("step_3d_vel", not three), ("deposit_3d", (not three) or not couple),
            ("splat_render", (not couple) or not three), ("splat_sum", (not couple) or three),
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
                # A resumed fluid step samples the saved grids. Put the live publishers back
                # before this step writes the next ones.
                bus = op.simbus  # noqa: F821
                baker = op("/sims/baker")  # noqa: F821
                table = bus.op("publishers")
                swapped = False
                for kind in ("forces", "sources"):
                    live = self.ownerComp.op(kind + "_2d")
                    hold = baker.op("hold_" + kind) if baker is not None else None
                    if live is None or hold is None:
                        continue
                    for row in range(1, table.numRows):
                        if table[row, 0].val == kind and table[row, 1].val == hold.path:
                            table[row, 1] = live.path
                            swapped = True
                if swapped:
                    bus.ext.FieldBus._refresh()
                summed = self.ownerComp.op("splat_sum")
                for name in ("forces_2d", "sources_2d"):
                    node = self.ownerComp.op(name)
                    if node is not None:
                        node.lock = False
                summed.cook(force=True)
                # Readback finishes the compute before the fluid samples it. Without the
                # sync the fluid sometimes sees a partial grid and two bakes diverge.
                summed.numpyArray(delayed=False)
                self.ownerComp.op("forces_2d").cook(force=True)
                self.ownerComp.op("sources_2d").cook(force=True)
                # The bus merge is wired by a parameter, so it does not notice this cook on its own.
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

    def StateTops(self) -> dict:
        return {field: self.ownerComp.op(field + "_out") for field in STATE}

    def StateFormats(self) -> dict:
        return dict(STATE)

    def SetInit(self, paths: dict | None = None) -> None:
        for field in STATE:
            load = self.ownerComp.op("load_" + field)
            choose = self.ownerComp.op("init_" + field)
            if paths:
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
            else:
                choose.par.index = 0

    def Params(self) -> dict:
        p = self.ownerComp.par
        bus = op.simbus3d if p.Bus.eval() == "3d" else op.simbus  # noqa: F821
        clock = op.simclock  # noqa: F821
        shaders = sorted((d.name, d.text) for d in self.ownerComp.findChildren(type=textDAT, depth=1))  # noqa: F821
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
            "publishers": {kind: bus.Publishers(kind) for kind in ("sources", "forces", "obstacles")},
            "shaders": digest,
        }

    def Unbind(self) -> None:
        pairs = (
            (op.simbus, ("forces_2d", "sources_2d")),  # noqa: F821
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
