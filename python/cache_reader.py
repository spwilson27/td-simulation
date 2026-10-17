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
        p.Status = f"{m.sim} {m.res[0]}x{m.res[1]} {state}"

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

    def Load(self, frame: float) -> None:
        """Jump to a frame and load it now (Movie File In otherwise loads on its next cook)."""
        self.ownerComp.par.Frame = frame
        for field in self.manifest.fields if self.manifest else ():
            for name in (f"{field}_a", f"{field}_b", f"{field}_x", f"{field}_out"):
                self.ownerComp.op(name).cook(force=True)
