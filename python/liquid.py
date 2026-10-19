"""GPU FLIP liquid. The network is built by build_liquid.py.

Particles live in two float textures (position+mass, velocity). The grid is the 3D bus, so obstacles
sampled with texelFetch line up with the cells. P2G gathers bins that a stable radix sort ordered by
cell and then by particle index (the bitonic sort is the reference check, not the live path).
Pressure is MGPCG with air as Dirichlet. Public particle outputs are locked nulls; the phi the bus
publishes is an inactive cache, so a smoke step does not recook this solver.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

STATE = {"pos": "rgba32float", "vel": "rgba32float", "preview": "rgba8fixed", "pressure": "mono32float"}
DEMO_OBSTACLES = ("sphere", "box", "capsule", "mesh")


def _ensure_python_path(comp) -> None:
    path = str(Path(comp.parent.sims.par.Root.eval()) / "python")
    if path not in sys.path:
        sys.path.insert(0, path)


# Unrolled MGPCG iterations. Must match build_liquid.PCG_STEPS.
PCG_STEPS = 8


def sort_passes(count: int) -> int:
    count = int(count)
    if count < 2 or count & (count - 1):
        raise ValueError("particle count must be a power of two")
    logn = count.bit_length() - 1
    return logn * (logn + 1) // 2


class Liquid:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        _ensure_python_path(ownerComp)
        self._resume = False

    def _bus(self):
        return op.simbus3d  # noqa: F821

    def _sync_threshold(self) -> None:
        """Liquid if the gathered mass exceeds Threshfrac of one cell of fluid (density * h^3)."""
        bus = self._bus()
        p = self.ownerComp.par
        width = max(int(bus.par.Resw.eval()), 1)
        h = (float(bus.par.Boundsx1.eval()) - float(bus.par.Boundsx0.eval())) / width
        thresh = float(p.Threshfrac.eval()) * float(p.Density.eval()) * h * h * h
        top = self.ownerComp.op("classify")
        for i in range(int(top.seq.vec.numBlocks)):
            if str(getattr(top.par, f"vec{i}name").eval()) != "uThresh":
                continue
            par = getattr(top.par, f"vec{i}valuex")
            par.expr = ""
            par.val = thresh
            return

    def ApplyPreset(self, name: str) -> None:
        """Point the 3D bus at this liquid's grid. Restore() puts the smoke defaults back."""
        bus = self._bus()
        p = self.ownerComp.par
        name = str(name)
        if name == "bake":
            # Deep dam on 128³. spacing 0.00675 fills just over 2 million of the 2048×1024 slots.
            res, spacing, particles, render = 128, 0.00525, (2048, 1024), (640, 360)
            p.Pcgiters, p.Pcgsmooth, p.Pcgcoarse = 8, 1, 20
            p.Flip = 0.55
            p.Spherer = 0.14
            p.Spherx, p.Sphery, p.Spherez = 0.62, 0.22, 0.5
            p.Boxenable = True
            p.Boxx0, p.Boxy0, p.Boxz0 = 0.0, 0.0, 0.0
            p.Boxx1, p.Boxy1, p.Boxz1 = 0.35, 0.85, 1.0
        elif name == "tank":
            res, spacing, particles, render = 64, 1.0 / 64.0, (512, 512), (480, 270)
            p.Pcgiters, p.Pcgsmooth, p.Pcgcoarse = 8, 1, 20
            p.Mode = "tank"
            p.Flip = 0.35
            p.Spherer = 0.0
            p.Boxenable = True
            p.Boxx0, p.Boxy0, p.Boxz0 = 0.0, 0.0, 0.0
            p.Boxx1, p.Boxy1, p.Boxz1 = 1.0, 0.45, 1.0
        else:
            name = "realtime"
            # 64³ still misses a 30 ms frame after the solve stopped syncing every iteration
            # (about 26 steps/s). 48³ with this spacing holds 124k particles.
            res, spacing, particles, render = 48, 0.014, (512, 256), (480, 270)
            p.Pcgiters, p.Pcgsmooth, p.Pcgcoarse = PCG_STEPS, 1, 16
            p.Mode = "dam"
            p.Flip = 0.55
            p.Spherer = 0.10
            p.Spherx, p.Sphery, p.Spherez = 0.72, 0.20, 0.5
            p.Boxenable = True
            p.Boxx0, p.Boxy0, p.Boxz0 = 0.0, 0.0, 0.0
            p.Boxx1, p.Boxy1, p.Boxz1 = 0.50, 0.70, 1.0
        if name == "bake":
            p.Mode = "dam"
        bus.SetGrid(res, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
        for wall in ("Wallleft", "Wallright", "Wallbottom", "Walltop", "Wallfront", "Wallback"):
            setattr(bus.par, wall, True)
        bus.par.Spongewidth = 0.0
        p.Resx, p.Resy = particles
        p.Spacing = spacing
        p.Renderw, p.Renderh = render
        p.Raysteps = 16 if name == "realtime" else 24
        p.Preset = name
        p.Gravityx, p.Gravityy, p.Gravityz = 0.0, -9.81, 0.0
        p.Project = True
        p.Cookrender = True
        p.Sortpasses = sort_passes(int(p.Resx) * int(p.Resy))
        # The host project's cook rate is the user's; the realtime preset is sized to hold 33 fps.
        self.Bind()

    def Restore(self) -> None:
        """Back to the phase-2 bus: 128^3, the tall bounds, open top, demo obstacles, liquid unpublished."""
        self.Unbind()
        bus = self._bus()
        bus.SetGrid(128, (-1.0, 0.0, -1.0, 1.0, 2.0, 1.0))
        for wall, closed in (("Wallleft", True), ("Wallright", True), ("Wallbottom", True),
                             ("Walltop", False), ("Wallfront", True), ("Wallback", True)):
            setattr(bus.par, wall, closed)
        bus.par.Spongewidth = 0.35
        bus.par.Spongerate = 4.0
        for name in DEMO_OBSTACLES:
            bus.Publish("obstacles", "/sims/demo3d/" + name)
        bus.Publish("sources", "/sims/demo3d/emitter_sources")
        bus.Publish("forces", "/sims/demo3d/emitter_forces")
        p = self.ownerComp.par
        p.Cookrender = False
        p.Spherer = 0.0
        for node_name in ("render", "preview", "phi", "div_after"):
            node = self.ownerComp.op(node_name)
            if node is not None:
                node.bypass = False

    def Bind(self) -> None:
        bus = self._bus()
        for name in DEMO_OBSTACLES:
            bus.Unpublish("/sims/demo3d/" + name)
        bus.Unpublish("/sims/demo3d/emitter_sources")
        bus.Unpublish("/sims/demo3d/emitter_forces")
        bus.Publish("obstacles", self.ownerComp.op("sphere").path)
        bus.Publish("obstacles", self.ownerComp.op("phi_out").path)

    def Unbind(self) -> None:
        bus = self._bus()
        bus.Unpublish(self.ownerComp.op("sphere").path)
        bus.Unpublish(self.ownerComp.op("phi_out").path)

    def BeginStep(self, reset: bool) -> None:
        p = self.ownerComp.par
        count = int(p.Resx.eval()) * int(p.Resy.eval())
        p.Sortpasses = sort_passes(count)
        self._sync_threshold()
        show = bool(p.Cookrender.eval())
        render = self.ownerComp.op("render")
        preview = self.ownerComp.op("preview")
        if render is not None:
            render.bypass = not show
        # The preview slice is baked state, so it stays live even when the beauty pass is off.
        if preview is not None:
            preview.bypass = False
        for field in ("pos", "vel"):
            src = self.ownerComp.op("src_" + field)
            if reset and self._resume:
                src.par.top = self.ownerComp.op("resume_" + field).path
            elif reset:
                src.par.top = self.ownerComp.op("init_" + field).path
            else:
                src.par.top = self.ownerComp.op(field + "_out").path
            src.cook(force=True)
        if reset:
            resumed = self._resume
            self.ownerComp.store("resume_flag", [bool(resumed), self.ownerComp.op("resume_pressure") is not None])
            self._resume = False
            unpack = self.ownerComp.op("unpack_pressure")
            if resumed and unpack is not None:
                unpack.cook(force=True)
                self._arm_pressure(unpack)
            else:
                zero = self.ownerComp.op("zero_l0")
                if zero is not None:
                    self._arm_pressure(zero)

    def _snap(self, hold_name: str) -> None:
        hold = self.ownerComp.op(hold_name)
        hold.par.replace = True
        hold.par.replaceindex = 0
        hold.par.active = True
        hold.cook(force=True)
        hold.par.active = False
        hold.par.replace = False

    def _coef(self, node: str, a: float, b: float) -> None:
        top = self.ownerComp.op(node)
        for i in range(int(top.seq.vec.numBlocks)):
            if str(getattr(top.par, f"vec{i}name").eval()) != "uCoef":
                continue
            px = getattr(top.par, f"vec{i}valuex")
            py = getattr(top.par, f"vec{i}valuey")
            px.expr = ""
            py.expr = ""
            px.val = float(a)
            py.val = float(b)
            return

    def _point(self, select: str, target) -> None:
        sel = self.ownerComp.op(select)
        sel.par.top.expr = ""
        sel.par.top = target if isinstance(target, str) else target.path

    def _scalar(self, name: str):
        import numpy as np  # noqa: PLC0415

        data = np.asarray(self.ownerComp.op(name).numpyArray(delayed=False))
        if data.ndim == 4:
            data = data[..., 0]
        return data

    def _dot(self, a_name: str, b_name: str) -> float:
        """Sum of a*b over liquid cells. Reduced on the GPU so a 64³ dot is one voxel, not a readback."""
        import numpy as np  # noqa: PLC0415

        comp = self.ownerComp
        self._point("dot_a", comp.op(a_name))
        self._point("dot_b", comp.op(b_name))
        comp.op("dot_r3").cook(force=True)
        data = np.asarray(comp.op("dot_r3").numpyArray(delayed=False))
        if data.ndim == 4:
            data = data[..., 0]
        return float(data.sum())

    def SortAgrees(self) -> bool:
        """True when the six-digit radix chain matches a full bitonic sort of the same keys."""
        import numpy as np  # noqa: PLC0415

        comp = self.ownerComp
        comp.op("keys").cook(force=True)
        comp.op("sort").cook(force=True)
        tail = comp.op("radix_scatter_5")
        tail.cook(force=True)
        radix = np.asarray(tail.numpyArray(delayed=False))
        full = np.asarray(comp.op("sort").numpyArray(delayed=False))
        if radix.shape != full.shape:
            return False
        return bool(np.array_equal(radix, full))

    def _fill_lock(self, name: str, source) -> None:
        """Copy `source` into an inactive 3D cache and disconnect it so a later read cannot recook `source`."""
        hold = self.ownerComp.op(name)
        hold.inputConnectors[0].disconnect()
        source.outputConnectors[0].connect(hold.inputConnectors[0])
        self._snap(name)
        hold.inputConnectors[0].disconnect()

    def _arm_pressure(self, source) -> None:
        """Warm-start the next solve from `source` (zero, or the resumed pressure)."""
        self._fill_lock("p_a", source)
        self._point("p_warm", self.ownerComp.op("p_a"))
        self.ownerComp.store("p_read", "p_a")

    def _pcg(self) -> None:
        """Cook the unrolled MGPCG chain once.

        Alpha and beta stay in 1x1 textures. The only CPU read is the final residual ratio,
        after the solve, so the iteration loop does not stall on numpyArray.
        """
        comp = self.ownerComp
        if not bool(comp.par.Project.eval()):
            comp.store("pcg_history", [])
            comp.store("pcg_iters_used", 0)
            return
        iters = int(comp.par.Pcgiters.eval()) if hasattr(comp.par, "Pcgiters") else PCG_STEPS
        iters = min(PCG_STEPS, max(1, iters))
        # Cooking pcg_rel pulls all eight unrolled iterations. A shorter count cooks cgx{n-1}
        # and leaves the tail untouched, so the parameter actually shortens the frame.
        if iters >= PCG_STEPS:
            tail = comp.op("pcg_rel")
            solved = comp.op("pcg_x")
            if tail is None or solved is None:
                return
            tail.cook(force=True)
            import numpy as np  # noqa: PLC0415

            rel = float(np.reshape(np.asarray(tail.numpyArray(delayed=False)), -1)[0])
            history = [rel]
        else:
            solved = comp.op(f"cgx{iters - 1}")
            if solved is None:
                return
            solved.cook(force=True)
            history = []
        # Project reads the snapshot, not the live chain. Retargeting the warm start marks
        # the chain dirty; the gradient must not cook that second solve.
        hold = self._ensure_pressure_out()
        self._point("pcg_p", solved)
        if not hold.inputConnectors[0].connections:
            comp.op("pcg_p").outputConnectors[0].connect(hold.inputConnectors[0])
        self._snap("pressure_out")
        hold.inputConnectors[0].disconnect()
        self._point("p_use", hold)
        read = str(comp.fetch("p_read", "p_a") or "p_a")
        write = "p_b" if read == "p_a" else "p_a"
        self._fill_lock(write, solved)
        self._point("p_warm", comp.op(write))
        comp.store("p_read", write)
        comp.store("pcg_history", history)
        comp.store("pcg_iters_used", iters)

    def PressureHistory(self) -> list:
        stored = self.ownerComp.fetch("pcg_history", None)
        if stored:
            return list(stored)
        return list(getattr(self, "_pcg_history", []) or [])

    def Step(self) -> None:
        # The projection has to land before G2P. MGPCG writes pcg_p; the particle shaders read it.
        self._pcg()
        # Divergence belongs to the solve, so snap it before advection dirties the liquid mask.
        self.ownerComp.op("div_after").cook(force=True)
        self.ownerComp.op("classify").cook(force=True)
        self._snap("div_out")
        self._snap("type_out")
        # Position and velocity both read the previous snapshot. Cook the velocity first so the
        # position shader's input is that new velocity and not a second integration.
        for field, node in (("vel", "g2p"), ("pos", "advect")):
            state = self.ownerComp.op(field + "_state")
            state.lock = False
            self.ownerComp.op(node).cook(force=True)
            state.cook(force=True)
            state.lock = True
        for field in ("pos", "vel"):
            out = self.ownerComp.op(field + "_out")
            out.lock = False
            out.cook(force=True)
            out.lock = True
        self.ownerComp.op("phi").cook(force=True)
        # phi_out is a cache: only its first snap in a TouchDesigner frame sticks. A bake
        # snaps it in _publish_preview, from the particles the frame saves.
        if not self._baking():
            self._snap("phi_out")
        self._publish_preview()
        if self.ownerComp.par.Cookrender.eval():
            self.ownerComp.op("render").cook(force=True)

    def _baking(self) -> bool:
        baker = op("/sims/baker")  # noqa: F821
        return baker is not None and bool(getattr(getattr(baker, "ext", None), "Baker", None) and baker.ext.Baker.baking)

    def _publish_preview(self) -> None:
        """Write this step's mid-depth slice into the bake hold.

        The preview GLSL TOP can already have cooked this frame, during the reset, while phi
        was still the previous step. A later cook(force=True) does not run again, so the first
        resumed file repeated that frame. The bake reads phi, which this step just cooked.
        """
        src = self.ownerComp.op("preview")
        if not self._baking():
            if src is not None:
                src.cook(force=True)
            return
        import numpy as np  # noqa: PLC0415

        # The particle select still shows the step's input. pos_out is the state this frame saves.
        # On a resumed frame those two disagree, and a cook(force) will not refresh the select.
        src_pos = self.ownerComp.op("src_pos")
        current = str(src_pos.par.top.eval())
        src_pos.par.top = self.ownerComp.op("pos_out").path
        phi_top = self.ownerComp.op("phi")
        par = phi_top.par.resolutionw
        expr = par.expr
        width = int(par.eval())
        try:
            par.expr = ""
            par.val = width + 1
            phi_top.cook(force=True)
            par.expr = expr
            phi_top.cook(force=True)
            # A baked frame's picture is the surface of the particles it saves, so a still
            # rebuilt from that frame's cache draws the same phi.
            self._snap("phi_out")
            phi = np.asarray(phi_top.numpyArray(delayed=False))
        finally:
            par.expr = expr
            src_pos.par.top = current
        field = phi[..., 0] if phi.ndim == 4 else phi
        depth, height, width = field.shape
        sl = field[depth // 2]
        ys = np.minimum((np.arange(256) * height) // 256, height - 1)
        xs = np.minimum((np.arange(256) * width) // 256, width - 1)
        small = sl[ys][:, xs]
        # Same tint as shaders/liquid/preview.glsl, so the cached slice is a picture of this phi.
        bg = np.array([0.05, 0.06, 0.08], dtype=np.float32)
        col = np.broadcast_to(bg, small.shape + (3,)).copy()
        wet = small < 1.0
        band = np.clip(0.5 - small * 2.0, 0.0, 1.0)[..., None]
        shallow = (1.0 - band) * np.array([0.15, 0.35, 0.55]) + band * np.array([0.02, 0.12, 0.22])
        col[wet] = shallow[wet]
        inside = small < 0.0
        band = np.clip(-small * 8.0, 0.0, 1.0)[..., None]
        deep = (1.0 - band) * np.array([0.25, 0.55, 0.7]) + band * np.array([0.03, 0.15, 0.28])
        col[inside] = deep[inside]
        image = np.concatenate([col, np.ones(small.shape + (1,), dtype=np.float32)], axis=2)
        image = np.ascontiguousarray(image, dtype=np.float32)
        hold = self.ownerComp.op("preview_hold")
        if hold is None:
            hold = self.ownerComp.create(scriptTOP, "preview_hold")  # noqa: F821
            hold.par.format = "rgba8fixed"
        hold.par.outputresolution = "custom"
        hold.par.resolutionw = 256
        hold.par.resolutionh = 256
        hold.par.format = "rgba8fixed"
        hold.lock = False
        hold.lock = True
        hold.copyNumpyArray(image)

    def _ensure_pressure_out(self):
        comp = self.ownerComp
        hold = comp.op("pressure_out")
        if hold is not None:
            return hold
        hold = comp.create(cacheTOP, "pressure_out")  # noqa: F821
        hold.par.active = False
        hold.par.alwayscook = False
        hold.par.cachesize = 1
        hold.par.outputindex = 0
        hold.par.format = "mono32float"
        hold.par.outputresolution = "useinput"
        hold.par.inputfiltertype = "nearest"
        hold.par.filtertype = "nearest"
        comp.op("pcg_p").outputConnectors[0].connect(hold.inputConnectors[0])
        return hold

    def StateTops(self) -> dict:
        self._ensure_pressure_out()
        return {
            "pos": self.ownerComp.op("pos_out"),
            "vel": self.ownerComp.op("vel_out"),
            "preview": self.ownerComp.op("preview_hold") or self.ownerComp.op("preview_out"),
            "pressure": self.ownerComp.op("pressure_out"),
        }

    def StateFormats(self) -> dict:
        return dict(STATE)

    def ExportArray(self, field: str):
        if field not in ("pos", "vel", "pressure"):
            return None
        name = "pressure_out" if field == "pressure" else field + "_out"
        top = self.ownerComp.op(name)
        image = top.numpyArray(delayed=False)
        if image is None:
            return None
        import numpy as np  # noqa: PLC0415

        image = np.array(image, dtype=np.float32, copy=True)
        if field == "pressure":
            # Pack the volume into a 2D image. A script TOP round-trips that; a 3D script TOP
            # does not come back through texelFetch as the same field.
            vol = image[..., 0]
            depth, height, width = vol.shape
            return np.ascontiguousarray(vol.reshape(depth * height, width)[..., None])
        return image

    def _hold_resume(self, field: str, image) -> None:
        name = "resume_" + field
        top = self.ownerComp.op(name)
        if top is None:
            top = self.ownerComp.create(scriptTOP, name)  # noqa: F821
            top.par.format = "rgba32float"
        top.par.outputresolution = "custom"
        if image.ndim == 4:
            top.par.resolutionw = int(image.shape[2])
            top.par.resolutionh = int(image.shape[1])
        else:
            top.par.resolutionw = int(image.shape[1])
            top.par.resolutionh = int(image.shape[0])
        top.lock = True
        top.copyNumpyArray(image)

    def SetInit(self, paths: dict | None = None) -> None:
        self._resume = False
        if not paths:
            for field in ("pos", "vel"):
                self.ownerComp.op("init_" + field).par.index = 0
            return
        root = Path(self.ownerComp.parent.sims.par.Root.eval())
        arrays = {}
        for field in ("pos", "vel"):
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
            pressure = paths.get("pressure")
            if pressure:
                npy = Path(str(pressure))
                if not npy.is_absolute():
                    npy = root / npy
                npy = npy.with_suffix(".npy")
                if npy.is_file():
                    import numpy as np  # noqa: PLC0415

                    self._hold_resume("pressure", np.ascontiguousarray(np.load(npy), dtype=np.float32))
                    pack = self.ownerComp.op("pressure_pack")
                    if pack is not None:
                        self._point("pressure_pack", self.ownerComp.op("resume_pressure"))
                        self.ownerComp.op("unpack_pressure").cook(force=True)
            self._resume = True
            return
        for field in ("pos", "vel"):
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
        bus, clock = self._bus(), op.simclock  # noqa: F821
        import cache  # noqa: PLC0415

        shaders = sorted((d.name, cache.text_dat_source(d)) for d in self.ownerComp.findChildren(type=textDAT, depth=1))  # noqa: F821
        digest = hashlib.sha256(repr(shaders).encode("utf-8")).hexdigest()[:16]
        return {
            "sim": "liquid",
            "preset": str(p.Preset.eval()),
            "mode": str(p.Mode.eval()),
            "particles": [int(p.Resx.eval()), int(p.Resy.eval())],
            "spacing": float(p.Spacing.eval()),
            "flip": float(p.Flip.eval()),
            "project": bool(p.Project.eval()),
            "gravity": [float(p.Gravityx.eval()), float(p.Gravityy.eval()), float(p.Gravityz.eval())],
            "smooth": int(p.Smooth.eval()),
            "coarse": int(p.Coarse.eval()),
            "vcycles": int(p.Vcycles.eval()),
            "threshold": float(p.Threshfrac.eval()),
            "density": float(p.Density.eval()),
            "sphere": [float(p.Spherx.eval()), float(p.Sphery.eval()), float(p.Spherez.eval()), float(p.Spherer.eval())],
            "res": list(bus.Res),
            "bounds": list(bus.Bounds),
            "fps": clock.par.Fps.eval(),
            "substeps": int(clock.par.Substeps.eval()),
            "seed": int(clock.par.Seed.eval()),
            "shaders": digest,
            "active_particles": self.SeedCount(),
            "pcg_iters": int(p.Pcgiters.eval()),
        }

    def SeedCount(self) -> int:
        """Live particles the seed shader will write. Same lattice as shaders/liquid/seed.glsl."""
        p = self.ownerComp.par
        if str(p.Mode.eval()) == "transfer" or not bool(p.Boxenable.eval()):
            return 0
        bus = self._bus()
        size = float(bus.par.Boundsx1.eval()) - float(bus.par.Boundsx0.eval())
        box0 = (float(p.Boxx0.eval()), float(p.Boxy0.eval()), float(p.Boxz0.eval()))
        box1 = (float(p.Boxx1.eval()), float(p.Boxy1.eval()), float(p.Boxz1.eval()))
        spacing = max(float(p.Spacing.eval()), 1e-6)
        count = 1
        for a, b in zip(box0, box1):
            n = int((b - a) * size / spacing + 1e-4)
            count *= max(n, 1)
        slots = int(p.Resx.eval()) * int(p.Resy.eval())
        return int(min(count, slots))

    def Scene(self) -> dict:
        """What a realtime measurement has to record. Bypass flags are the nodes the frame actually cooks."""
        p = self.ownerComp.par
        names = (
            "keys", "sort", "bins", "p2g", "vel_p2g", "classify", "forces", "extrapolate", "divergence",
            "a_pre0", "b_pre0", "polish", "project", "g2p", "advect", "phi", "render", "preview", "sphere",
        )
        bypass = {}
        errors = []
        for name in names:
            node = self.ownerComp.op(name)
            if node is None:
                bypass[name] = None
                continue
            bypass[name] = bool(node.bypass)
            err = node.errors()
            if err:
                errors.append(name)
        alive = self._alive_count()
        return {
            "preset": str(p.Preset.eval()),
            "mode": str(p.Mode.eval()),
            "res": list(self._bus().Res),
            "bounds": list(self._bus().Bounds),
            "particles": [int(p.Resx.eval()), int(p.Resy.eval())],
            "active_particles": alive,
            "spacing": float(p.Spacing.eval()),
            "solver": "mgpcg",
            "pcg_iters": int(p.Pcgiters.eval()),
            "pcg_iters_used": int(self.ownerComp.fetch("pcg_iters_used", 0) or 0),
            "pressure_history": [round(float(x), 4) for x in self.PressureHistory()],
            "vcycles": int(p.Vcycles.eval()),
            "smooth": int(p.Smooth.eval()),
            "coarse": int(p.Coarse.eval()),
            "polish": int(p.Polish.eval()),
            "flip": float(p.Flip.eval()),
            "project": bool(p.Project.eval()),
            "gravity": [float(p.Gravityx.eval()), float(p.Gravityy.eval()), float(p.Gravityz.eval())],
            "render": [int(p.Renderw.eval()), int(p.Renderh.eval())],
            "raysteps": int(p.Raysteps.eval()),
            "cookrender": bool(p.Cookrender.eval()),
            # Operator paths stay relative so the report has no absolute path strings.
            "sims": str(op.simclock.par.Sims.eval()).lstrip("/"),  # noqa: F821
            "bypass": bypass,
            "errors": errors,
            "sphere_radius": float(p.Spherer.eval()),
        }

    def _alive_count(self) -> int:
        import numpy as np  # noqa: PLC0415

        pos = self.ownerComp.op("pos_out").numpyArray(delayed=False)
        if pos is None:
            return 0
        return int(np.count_nonzero(pos[..., 3] > 0.0))

    def Report(self) -> dict:
        """Particle, divergence and volume numbers from the snapshots. Does not cook the solver."""
        import numpy as np  # noqa: PLC0415

        pos = np.asarray(self.ownerComp.op("pos_out").numpyArray(delayed=False), dtype=np.float64)
        vel = np.asarray(self.ownerComp.op("vel_out").numpyArray(delayed=False), dtype=np.float64)
        # 2D numpyArray is (height, width, channels), row 0 is the top of the texture. The shader
        # indexed y from the bottom. Mass is in pos.a either way; we only reduce, we don't look up texels.
        alive = pos[..., 3] > 0.0
        speed = np.linalg.norm(vel[..., :3], axis=-1)
        mass = pos[..., 3]
        density = float(self.ownerComp.par.Density.eval())
        volume = float(mass[alive].sum() / density) if np.any(alive) and density > 0 else 0.0
        # The cache TOP keeps the first image of a TouchDesigner frame, so a script that ticks
        # several times reads the live divergence instead. It was cooked at the end of Step.
        div = np.asarray(self.ownerComp.op("div_out").numpyArray(delayed=False), dtype=np.float64)
        kind = np.asarray(self.ownerComp.op("type_out").numpyArray(delayed=False), dtype=np.float64)
        if div.ndim == 4:
            d = div[..., 0]
            kind_r = kind[..., 0] if kind.ndim == 4 else kind
        else:
            d = div
            kind_r = kind[..., 0] if kind.ndim == 4 else kind
        liquid = np.rint(kind_r) == 1
        if np.any(liquid):
            samples = d[liquid]
            div_rms = float(np.sqrt(np.mean(np.square(samples))))
            div_max = float(np.max(np.abs(samples)))
        else:
            div_rms = 0.0
            div_max = 0.0
        if np.any(alive):
            xs = pos[..., 0][alive]
            ys = pos[..., 1][alive]
            front = float(xs.max())
            mean_y = float(ys.mean())
            max_speed = float(speed[alive].max())
            mean_speed = float(speed[alive].mean())
        else:
            front = mean_y = max_speed = mean_speed = 0.0
        h = 0.0
        res = self._bus().Res
        bounds = self._bus().Bounds
        if res[0]:
            h = (bounds[3] - bounds[0]) / float(res[0])
        cell_volume = float(np.count_nonzero(liquid)) * (h ** 3)
        return {
            "active": int(np.count_nonzero(alive)),
            "volume": volume,
            "cell_volume": cell_volume,
            "max_speed": max_speed,
            "mean_speed": mean_speed,
            "mean_y": mean_y,
            "front_x": front,
            "div_rms": div_rms,
            "div_max": div_max,
            "liquid_cells": int(np.count_nonzero(liquid)),
            "finite": bool(np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(d).all()),
        }
