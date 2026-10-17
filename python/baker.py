"""Baker extension: steps a simulation in bake mode and writes its state fields to an EXR cache with a manifest.

Custom parameters: Sim (COMP with BeginStep/Step/StateTops/StateFormats/SetInit/Params), Cachedir (relative paths
resolve against the project Root), Framestart, Frameend, Warmup (unsaved frames simulated before Framestart),
Resume, Asyncsave, Pauseafter (stop the clock when done, leaving the last baked state live), pulses Bake / Cancel /
Clearcache, read-only Status, Progress, Eta.

While baking TouchDesigner's real-time flag is off, so each frame is fully computed however long it takes; the
previous value is restored when the bake finishes or is cancelled, along with the clock's Startframe, Mode, and
Running (Running is forced off when Pause After Bake is on). Every saved frame contains the full simulation
state, so an interrupted bake resumes from the last complete frame without re-simulating the earlier ones. A
longer Frameend with the same parameter hash extends the cache instead of deleting it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _ensure_python_path(comp) -> None:
    path = str(Path(comp.parent.sims.par.Root.eval()) / "python")
    if path not in sys.path:
        sys.path.insert(0, path)


class Baker:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        _ensure_python_path(ownerComp)
        import importlib  # noqa: PLC0415

        import cache  # noqa: PLC0415

        importlib.reload(cache)
        self.cache = cache
        self.manifest = None
        self.baking = False
        self._prev_realtime = None
        self._prev_startframe = None
        self._prev_mode = None
        self._prev_running = None
        self._t0 = 0.0
        self._frames_this_run = 0

    # -- helpers ----------------------------------------------------------------------------------------------
    @property
    def Sim(self):
        return op(self.ownerComp.par.Sim.eval())  # noqa: F821

    @property
    def CacheDir(self) -> Path:
        d = Path(self.ownerComp.par.Cachedir.eval())
        return d if d.is_absolute() else Path(self.ownerComp.parent.sims.par.Root.eval()) / d

    def _set_status(self, text: str) -> None:
        self.ownerComp.par.Status = text

    def NewManifest(self):
        sim, p = self.Sim, self.ownerComp.par
        bus, clock = op.simbus.ext.FieldBus, op.simclock  # noqa: F821
        return self.cache.Manifest(
            sim=sim.name, params=sim.Params(), fields=sim.StateFormats(), res=list(bus.Res), bounds=list(bus.Bounds),
            fps=clock.par.Fps.eval(), substeps=int(clock.par.Substeps.eval()),
            frame_start=int(p.Framestart.eval()), frame_end=int(p.Frameend.eval()),
            td_version=f"{app.version}.{app.build}",  # noqa: F821
        )

    def CacheStatus(self) -> str:
        p = self.ownerComp.par
        return self.cache.cache_status(self.CacheDir, self.Sim.Params(), int(p.Framestart.eval()), int(p.Frameend.eval()))

    # -- actions ----------------------------------------------------------------------------------------------
    def Bake(self) -> None:
        if self.baking:
            return
        p, clock = self.ownerComp.par, op.simclock  # noqa: F821
        sim, cache_dir = self.Sim, self.CacheDir
        manifest = self.NewManifest()
        plan = self.cache.directory_plan(cache_dir, manifest.params, manifest.frame_start, manifest.frame_end)
        if plan["action"] == "complete" and p.Resume.eval():
            self._set_status("complete (nothing to do)")
            return
        start = int(plan["frame_start"])
        if plan["action"] in ("stale", "missing") or not p.Resume.eval():
            self.cache.clear_cache(cache_dir, self.cache.read_manifest(cache_dir) or manifest)
            resume = start
        else:
            # Same identity. A longer Frameend keeps the files and continues; a shorter one stops at the request.
            manifest = self.cache.read_manifest(cache_dir)
            manifest.frame_end = int(plan["frame_end"])
            resume = int(plan["resume_frame"])
        manifest.complete = False
        self.cache.write_manifest(cache_dir, manifest)
        self.manifest = manifest

        warmup = int(p.Warmup.eval())
        self._prev_startframe = int(clock.par.Startframe.eval())
        self._prev_mode = str(clock.par.Mode.eval())
        self._prev_running = bool(clock.par.Running.eval())
        clock.par.Startframe = start - warmup
        if resume > start:
            sim.SetInit({f: self.cache.frame_path(cache_dir, f, resume - 1) for f in manifest.fields})
            clock.RequestReset(resume)
        else:
            sim.SetInit(None)
            clock.RequestReset(start - warmup)

        self._prev_realtime = project.realTime  # noqa: F821
        project.realTime = False  # noqa: F821
        clock.par.Mode = "bake"
        clock.par.Running = True
        clock.AddListener(self.ownerComp)
        self.baking = True
        self._t0 = time.perf_counter()
        self._frames_this_run = 0
        self._set_status(f"baking from frame {resume}")

    def OnStep(self, clock_ext) -> None:
        if not self.baking:
            return
        c, m = clock_ext.Clock, self.manifest
        if not c.frame_complete or c.frame < m.frame_start:
            return
        asynchronous = bool(self.ownerComp.par.Asyncsave.eval())
        for field, top in self.Sim.StateTops().items():
            path = self.cache.frame_path(self.CacheDir, field, c.frame)
            path.parent.mkdir(parents=True, exist_ok=True)
            top.save(str(path), asynchronous=asynchronous)
        m.frames_written = c.frame - m.frame_start + 1
        self._frames_this_run += 1
        done = c.frame >= m.frame_end
        if done or self._frames_this_run % 10 == 0:
            self.cache.write_manifest(self.CacheDir, m)
        elapsed = time.perf_counter() - self._t0
        remaining = m.frame_end - c.frame
        self.ownerComp.par.Progress = m.frames_written / m.frame_count
        self.ownerComp.par.Eta = elapsed / self._frames_this_run * remaining
        self._set_status(f"baking frame {c.frame} / {m.frame_end}")
        if done:
            self._finish(complete=True)

    def Cancel(self) -> None:
        if self.baking:
            self._finish(complete=False)

    def ClearCache(self) -> int:
        if self.baking:
            return 0
        n = self.cache.clear_cache(self.CacheDir, self.cache.read_manifest(self.CacheDir))
        self._set_status(f"cleared {n} files")
        return n

    def _finish(self, complete: bool) -> None:
        clock = op.simclock  # noqa: F821
        clock.RemoveListener(self.ownerComp)
        if self._prev_mode is not None:
            clock.par.Mode = self._prev_mode
        if self._prev_startframe is not None:
            clock.par.Startframe = self._prev_startframe
        if self.ownerComp.par.Pauseafter.eval():
            clock.par.Running = False
        elif self._prev_running is not None:
            clock.par.Running = self._prev_running
        if self._prev_realtime is not None:
            project.realTime = self._prev_realtime  # noqa: F821
        m = self.manifest
        m.complete = complete and self.cache.resume_frame(self.CacheDir, m) > m.frame_end
        self.cache.write_manifest(self.CacheDir, m)
        self.baking = False
        secs = time.perf_counter() - self._t0
        self._set_status(f"{'complete' if m.complete else 'cancelled'}: {self._frames_this_run} frames in {secs:.1f}s")
