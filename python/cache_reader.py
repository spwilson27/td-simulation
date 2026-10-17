"""CacheReader extension: plays a baked cache back onto the bus.

Custom parameters: Cachedir, Frame (float; defaults to the sim clock frame), Interpolate (blend neighbouring frames
for slow motion), pulse Reload, read-only Status, Framestart, Frameend.

For each state field there are two Movie File In TOPs (frames a and b) blended by a Cross TOP into <field>_out, so
the reader exposes the same vel_out / scalars_out / pressure_out TOPs as a live sim and can be the bus field source
(op.simbus.ext.FieldBus.BindFields(reader)). sources_out republishes the cached scalars, scaled by Gain, as bus
sources so a cached simulation can feed a live one.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_python_path(comp) -> None:
    path = str(Path(comp.parent.sims.par.Root.eval()) / "python")
    if path not in sys.path:
        sys.path.insert(0, path)


class CacheReader:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        _ensure_python_path(ownerComp)
        import importlib  # noqa: PLC0415

        import cache  # noqa: PLC0415

        importlib.reload(cache)
        self.cache = cache
        self.manifest = None
        self.Reload()

    @property
    def CacheDir(self) -> Path:
        d = Path(self.ownerComp.par.Cachedir.eval())
        return d if d.is_absolute() else Path(self.ownerComp.parent.sims.par.Root.eval()) / d

    def Reload(self) -> None:
        self.manifest = self.cache.read_manifest(self.CacheDir)
        p = self.ownerComp.par
        if self.manifest is None:
            p.Status = f"no cache at {self.CacheDir}"
            return
        m = self.manifest
        last = self.cache.resume_frame(self.CacheDir, m) - 1
        p.Framestart, p.Frameend = m.frame_start, max(m.frame_start, last)
        state = "complete" if m.complete else f"partial ({last - m.frame_start + 1} of {m.frame_count} frames)"
        if len(m.res) >= 3:
            dims = f"{m.res[0]}x{m.res[1]}x{m.res[2]}"
        else:
            dims = f"{m.res[0]}x{m.res[1]}"
        p.Status = f"{m.sim} {dims} {state}"
        for field in m.fields:
            self._ensure_field(field)
            if len(m.res) >= 3:
                layout = (m.extra or {}).get("atlas") or self.cache.atlas_layout(m.res[:3])
                self._ensure_volume(field, layout, m.fields[field])

    def _blend(self):
        p = self.ownerComp.par
        return self.cache.blend_frames(p.Frame.eval(), int(p.Framestart.eval()), int(p.Frameend.eval()))

    def Path(self, field: str, which: int = 0) -> str:
        if self.manifest is None or field not in self.manifest.fields:
            return ""
        a, b, _ = self._blend()
        return str(self.cache.frame_path(self.CacheDir, field, b if which else a))

    def Weight(self) -> float:
        if self.manifest is None or not self.ownerComp.par.Interpolate.eval():
            return 0.0
        return self._blend()[2]

    def Output(self, field: str):
        """Decoded volume when the manifest is 3D, otherwise the 2D (or atlas) `{field}_out`."""
        if self.manifest is not None and len(self.manifest.res) >= 3:
            vol = self.ownerComp.op(field + "_vol")
            if vol is not None:
                return vol
        return self.ownerComp.op(field + "_out")

    def _ensure_field(self, field: str) -> None:
        if self.ownerComp.op(field + "_a") is not None:
            return
        comp = self.ownerComp
        movies = []
        for which in (0, 1):
            movie = comp.create(moviefileinTOP, f"{field}_{'ab'[which]}")
            movie.par.playmode = "locked"
            movie.par.file.expr = f"parent().ext.CacheReader.Path({field!r}, {which})"
            movies.append(movie)
        cross = comp.create(crossTOP, field + "_x")
        movies[0].outputConnectors[0].connect(cross.inputConnectors[0])
        movies[1].outputConnectors[0].connect(cross.inputConnectors[1])
        cross.par.cross.expr = "parent().ext.CacheReader.Weight()"
        out = comp.create(nullTOP, field + "_out")
        cross.outputConnectors[0].connect(out.inputConnectors[0])

    def _ensure_volume(self, field: str, layout: dict, fmt: str):
        comp = self.ownerComp
        name = field + "_vol"
        top = comp.op(name)
        if top is None:
            top = comp.create(glslTOP, name)
            top.par.mode = "compute"
            top.par.computedat = comp.op("atlas_decode_src")
            top.par.glslversion = "glsl460"
            top.par.type = "texture3d"
            top.par.depth = "custom"
            top.par.outputresolution = "custom"
            top.par.inputfiltertype = "nearest"
            top.par.filtertype = "nearest"
            top.par.outputaccess = "writeonly"
            cross = comp.op(field + "_x")
            cross.outputConnectors[0].connect(top.inputConnectors[0])
            top.seq.vec.numBlocks = 2
            top.par.vec0name = "uAtlas"
            top.par.vec1name = "uAtlas2"
        vol = layout["volume"]
        top.par.format = fmt
        top.par.resolutionw = int(vol[0])
        top.par.resolutionh = int(vol[1])
        top.par.customdepth = int(vol[2])
        top.par.vec0valuex = int(layout["cols"])
        top.par.vec0valuey = int(layout["rows"])
        top.par.vec0valuez = int(vol[0])
        top.par.vec0valuew = int(vol[1])
        top.par.vec1valuex = int(vol[2])
        return top

    def Load(self, frame: float) -> None:
        """Jump to a frame and load it now (Movie File In otherwise loads on its next cook)."""
        self.ownerComp.par.Frame = frame
        for field in self.manifest.fields if self.manifest else ():
            for name in (f"{field}_a", f"{field}_b", f"{field}_x", f"{field}_out", f"{field}_vol"):
                top = self.ownerComp.op(name)
                if top is not None:
                    top.cook(force=True)
