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
        member = self._member()
        if member:
            p.Status = f"group {member} {dims} {state}"
        else:
            p.Status = f"{m.sim} {dims} {state}"
        fields, info = self._active()
        res = info["res"] if info else m.res
        extra = (info or {}).get("extra") or m.extra or {}
        for field in fields:
            self._ensure_field(field)
            if len(res) >= 3:
                layout = extra.get("atlas") or self.cache.atlas_layout(res[:3])
                self._ensure_volume(field, layout, fields[field])
                cropping = self._crop_active() and field in self._crop_fields()
                vol = self.ownerComp.op(field + "_vol")
                if cropping:
                    self._ensure_crop(field, res, fields[field])
                    if vol is not None:
                        vol.bypass = True
                elif vol is not None:
                    vol.bypass = False
            if not (self._crop_active() and field in self._crop_fields()):
                for tag in "ab":
                    movie = self.ownerComp.op(f"{field}_{tag}")
                    if movie is not None and str(movie.par.outputresolution.eval()) == "custom":
                        movie.par.outputresolution = "native"

    def _blend(self):
        p = self.ownerComp.par
        return self.cache.blend_frames(p.Frame.eval(), int(p.Framestart.eval()), int(p.Frameend.eval()))

    def _member(self) -> str | None:
        m = self.manifest
        if m is None or not m.members:
            return None
        name = str(self.ownerComp.par.Member.eval()).strip()
        if name in m.members:
            return name
        return next(iter(m.members))

    def _active(self):
        m = self.manifest
        member = self._member()
        if member:
            return m.members[member]["fields"], m.members[member]
        return m.fields, None

    def Path(self, field: str, which: int = 0) -> str:
        if self.manifest is None:
            return ""
        fields, _info = self._active()
        if field not in fields:
            return ""
        a, b, _ = self._blend()
        frame = b if which else a
        frame = self.cache.resolve_frame(self.manifest, field, frame, self._member())
        return str(self.cache.frame_path(self.CacheDir, field, frame, self._member()))

    def Weight(self) -> float:
        if self.manifest is None or not self.ownerComp.par.Interpolate.eval():
            return 0.0
        return self._blend()[2]

    def Output(self, field: str):
        """Decoded volume when the manifest is 3D, otherwise the 2D (or atlas) `{field}_out`.

        A cropped field is expanded back to the full grid. That TOP is `{field}_play`.
        """
        if self._crop_active() and field in self._crop_fields():
            play = self.ownerComp.op(field + "_play")
            if play is not None:
                return play
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
        fields = self._active()[0] if self.manifest else {}
        crop_fields = self._crop_fields() if self._crop_active() else set()
        for field in fields:
            if field in crop_fields:
                self._load_cropped(field)
                continue
            for name in (f"{field}_a", f"{field}_b", f"{field}_x", f"{field}_out", f"{field}_vol"):
                top = self.ownerComp.op(name)
                if top is not None:
                    top.cook(force=True)

    def _crop_active(self) -> bool:
        storage = self.cache.storage_of(self.manifest, self._member()) if self.manifest else {}
        if not storage.get("crop"):
            return False
        res = self.manifest.res
        return len(res) >= 3

    def _crop_fields(self) -> set:
        storage = self.cache.storage_of(self.manifest, self._member()) if self.manifest else {}
        return set(storage.get("crop_fields") or [])

    def _active_res(self):
        _fields, info = self._active()
        return (info or {}).get("res") or self.manifest.res

    def _layout_for(self, field: str, frame: int) -> tuple[dict, list]:
        """Atlas layout and expand origin/crop size for one stored frame of `field`."""
        resolved = self.cache.resolve_frame(self.manifest, field, frame, self._member())
        res = [int(v) for v in self._active_res()[:3]]
        bbox = self.cache.frame_bbox(self.manifest, resolved)
        storage = self.cache.storage_of(self.manifest, self._member())
        cropped = bbox is not None and field in set(storage.get("crop_fields") or []) and not self.cache.is_keyframe(
            self.manifest, resolved, self._member())
        if cropped:
            shape = self.cache.bbox_shape(bbox)
            return self.cache.atlas_layout(shape), [int(bbox[0]), int(bbox[1]), int(bbox[2]), *shape]
        layout = (self.manifest.extra or {}).get("atlas") or self.cache.atlas_layout(res)
        return layout, [0, 0, 0, *res]

    def _ensure_crop(self, field: str, res, fmt: str) -> None:
        comp = self.ownerComp
        vol = [int(res[0]), int(res[1]), int(res[2])]
        play = comp.op(field + "_play")
        if play is not None:
            # The expand target is the full volume. A later cache can be a different grid;
            # leaving the first grid here made a 256³ replay compare a 128³ picture.
            play.par.format = fmt
            play.par.resolutionw = vol[0]
            play.par.resolutionh = vol[1]
            play.par.customdepth = vol[2]
            for tag in "ab":
                exp = comp.op(f"{field}_exp{tag}")
                if exp is None:
                    continue
                exp.par.format = fmt
                exp.par.resolutionw = vol[0]
                exp.par.resolutionh = vol[1]
                exp.par.customdepth = vol[2]
            return
        decoders = []
        for which, tag in ((0, "a"), (1, "b")):
            dec = comp.create(glslTOP, f"{field}_dec{tag}")
            dec.par.mode = "compute"
            dec.par.computedat = comp.op("atlas_decode_src")
            dec.par.glslversion = "glsl460"
            dec.par.type = "texture3d"
            dec.par.depth = "custom"
            dec.par.outputresolution = "custom"
            dec.par.format = fmt
            dec.par.inputfiltertype = "nearest"
            dec.par.filtertype = "nearest"
            dec.par.outputaccess = "writeonly"
            comp.op(f"{field}_{tag}").outputConnectors[0].connect(dec.inputConnectors[0])
            dec.seq.vec.numBlocks = 2
            dec.par.vec0name = "uAtlas"
            dec.par.vec1name = "uAtlas2"
            exp = comp.create(glslTOP, f"{field}_exp{tag}")
            exp.par.mode = "compute"
            exp.par.computedat = comp.op("atlas_expand_src")
            exp.par.glslversion = "glsl460"
            exp.par.type = "texture3d"
            exp.par.depth = "custom"
            exp.par.outputresolution = "custom"
            exp.par.format = fmt
            exp.par.resolutionw = vol[0]
            exp.par.resolutionh = vol[1]
            exp.par.customdepth = vol[2]
            exp.par.inputfiltertype = "nearest"
            exp.par.filtertype = "nearest"
            exp.par.outputaccess = "writeonly"
            dec.outputConnectors[0].connect(exp.inputConnectors[0])
            exp.seq.vec.numBlocks = 2
            exp.par.vec0name = "uOrigin"
            exp.par.vec1name = "uCrop"
            decoders.append((dec, exp))
        blend = comp.create(glslTOP, field + "_play")
        blend.par.mode = "compute"
        blend.par.computedat = comp.op("blend3d_src")
        blend.par.glslversion = "glsl460"
        blend.par.type = "texture3d"
        blend.par.depth = "custom"
        blend.par.outputresolution = "custom"
        blend.par.format = fmt
        blend.par.resolutionw = vol[0]
        blend.par.resolutionh = vol[1]
        blend.par.customdepth = vol[2]
        blend.par.inputfiltertype = "nearest"
        blend.par.filtertype = "nearest"
        blend.par.outputaccess = "writeonly"
        decoders[0][1].outputConnectors[0].connect(blend.inputConnectors[0])
        decoders[1][1].outputConnectors[0].connect(blend.inputConnectors[1])
        blend.seq.vec.numBlocks = 1
        blend.par.vec0name = "uWeight"

    def _load_cropped(self, field: str) -> None:
        a, b, weight = self._blend()
        if not self.ownerComp.par.Interpolate.eval():
            weight = 0.0
        for which, frame, tag in ((0, a, "a"), (1, b, "b")):
            layout, placed = self._layout_for(field, frame)
            movie = self.ownerComp.op(f"{field}_{tag}")
            movie.par.outputresolution = "custom"
            movie.par.resolutionw = int(layout["width"])
            movie.par.resolutionh = int(layout["height"])
            movie.par.index = 0
            movie.cook(force=True)
            if hasattr(movie.par, "reload"):
                movie.par.reload.pulse()
                movie.cook(force=True)
            dec = self.ownerComp.op(f"{field}_dec{tag}")
            dec.par.resolutionw = int(placed[3])
            dec.par.resolutionh = int(placed[4])
            dec.par.customdepth = int(placed[5])
            dec.par.vec0valuex = int(layout["cols"])
            dec.par.vec0valuey = int(layout["rows"])
            dec.par.vec0valuez = int(placed[3])
            dec.par.vec0valuew = int(placed[4])
            dec.par.vec1valuex = int(placed[5])
            dec.cook(force=True)
            exp = self.ownerComp.op(f"{field}_exp{tag}")
            exp.par.vec0valuex = int(placed[0])
            exp.par.vec0valuey = int(placed[1])
            exp.par.vec0valuez = int(placed[2])
            exp.par.vec1valuex = int(placed[3])
            exp.par.vec1valuey = int(placed[4])
            exp.par.vec1valuez = int(placed[5])
            exp.cook(force=True)
        play = self.ownerComp.op(field + "_play")
        play.par.vec0valuex = float(weight)
        play.cook(force=True)
