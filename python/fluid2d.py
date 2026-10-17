"""2D smoke/fire solver extension (the network itself is built by build_network.py).

Per step: advect velocity and scalars (RK2 semi-Lagrangian + MacCormack), combustion, body forces with vorticity
confinement, divergence, Jacobi pressure solve (Num Passes), projection. State lives in three Feedback TOPs; a
Switch in each loop selects the init state (zeros or a cached frame) on reset steps.
"""

from __future__ import annotations

import hashlib

STATE = {"vel": "rg32float", "scalars": "rgba16float", "pressure": "mono32float"}

PHYSICS_PARS = (
    "Velocitydecay", "Smokedecay", "Fueldecay", "Maccormack", "Burnrate", "Burnheat", "Burnsmoke", "Ignition", "Cooling",
    "Expansion", "Maxtemp", "Buoyancy", "Weight", "Vorticity", "Windx", "Windy", "Winddrag", "Maxspeed", "Jacobi",
)

PRESETS = {
    "realtime": {"width": 1024, "substeps": 1, "Jacobi": 30},
    "bake": {"width": 2048, "substeps": 4, "Jacobi": 80},
    "preview": {"width": 512, "substeps": 1, "Jacobi": 20},
}


class Fluid2D:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp

    # -- stepping (called by the clock) ---------------------------------------------------------------------
    def BeginStep(self, reset: bool) -> None:
        for field in STATE:
            self.ownerComp.op("loop_" + field).par.index = 1 if reset else 0

    def Step(self) -> None:
        # A Feedback TOP only captures its target on frames where it cooked itself. On reset steps the loop Switch
        # does not pull it, so cook it explicitly or the next step would replay the state from before the reset.
        # Unlock the public snapshots first. They are locked at the end of the step so a particle read cannot
        # recook the solver and move the frame that was just saved.
        for field in STATE:
            self.ownerComp.op(field + "_out").lock = False
            self.ownerComp.op(field + "_state").lock = False
        for field in STATE:
            self.ownerComp.op("fb_" + field).cook()
        for field in STATE:
            self.ownerComp.op(field + "_state").cook()
        # Snapshot the solved fields into the public *_out Cache TOPs, then set them inactive. A Cache TOP that is
        # inactive does not cook its input, so a later read of vel_out / scalars_out / pressure_out (the bus, a
        # viewer, the bake replay) cannot run the feedback loop again. That extra cook used to advance the live
        # state one step past the frame the baker had just saved. One TouchDesigner frame records a single cache
        # entry, which is what a timeline bake does. A script that ticks several steps has to call Snapshot().
        for field in STATE:
            # Replace Pulse copies the solved null immediately. An Active cook records only one
            # image per TouchDesigner frame and would leave the bus on the frame loaded at resume.
            held = self.ownerComp.op(field + "_out")
            held.lock = False
            held.par.replace = True
            held.par.replaceindex = 0
            held.par.replacespulse.pulse()
            held.par.replace = False
            held.lock = True
        if self.ownerComp.par.Cookrender.eval():
            self.ownerComp.op("render_out").cook()

    def Snapshot(self) -> None:
        """Store the latest solved frame. Needed when several Ticks run in one script.

        A size-1 Cache TOP records one image per TouchDesigner frame. The Replace Pulse
        copies the solved null immediately. Replace is turned off before the function
        returns so the pulse is not still armed for a later read.
        """
        for field in STATE:
            held = self.ownerComp.op(field + "_out")
            held.lock = False
            held.par.replace = True
            held.par.replaceindex = 0
            held.par.replacespulse.pulse()
            held.par.replace = False
            held.lock = True
        if self.ownerComp.par.Cookrender.eval():
            self.ownerComp.op("render_out").cook(force=True)

    # -- state for the baker ----------------------------------------------------------------------------------
    def StateTops(self) -> dict:
        return {field: self.ownerComp.op(field + "_out") for field in STATE}

    def StateFormats(self) -> dict:
        return dict(STATE)

    def SetInit(self, paths: dict | None = None) -> None:
        """Init state for the next reset step: zeros (paths=None) or EXR files {field: path} from a cache."""
        for field in STATE:
            load = self.ownerComp.op("load_" + field)
            choose = self.ownerComp.op("init_" + field)
            if paths:
                state = self.ownerComp.op(field + "_state")
                load.par.file = str(paths[field])
                load.par.playmode = "locked"
                load.par.index.mode = ParMode.CONSTANT  # noqa: F821
                load.par.index = 0
                # useinput ignores a file that has no wired input and shows a stale image.
                load.par.outputresolution = "custom"
                load.par.resolutionw = state.width
                load.par.resolutionh = state.height
                load.par.reloadpulse.pulse()
                load.cook(force=True)
                choose.par.index = 1
            else:
                choose.par.index = 0

    def Params(self) -> dict:
        """Everything that changes the result, for the cache parameter hash (shader sources and publisher settings)."""
        p = self.ownerComp.par
        bus, clock = op.simbus, op.simclock  # noqa: F821 (td globals)
        shaders = sorted((d.name, d.text) for d in self.ownerComp.findChildren(type=textDAT, depth=1))  # noqa: F821
        digest = hashlib.sha256(repr(shaders).encode("utf-8")).hexdigest()[:16]
        return {
            "sim": "fluid2d",
            "physics": {name: getattr(p, name).eval() for name in PHYSICS_PARS},
            "res": list(bus.ext.FieldBus.Res),
            "bounds": list(bus.ext.FieldBus.Bounds),
            "walls": [bool(getattr(bus.par, w).eval()) for w in ("Wallleft", "Wallright", "Wallbottom", "Walltop")],
            "sponge": [bus.par.Spongewidth.eval(), bus.par.Spongerate.eval()],
            "fps": clock.par.Fps.eval(),
            "substeps": int(clock.par.Substeps.eval()),
            "seed": int(clock.par.Seed.eval()),
            "inputs": bus.ext.FieldBus.PublisherSettings(),
            "shaders": digest,
        }

    def ApplyPreset(self, name: str) -> None:
        preset = PRESETS[name]
        op.simbus.ext.FieldBus.SetGrid(preset["width"])  # noqa: F821
        op.simclock.par.Substeps = preset["substeps"]  # noqa: F821
        self.ownerComp.par.Jacobi = preset["Jacobi"]
        self.ownerComp.par.Preset = name
