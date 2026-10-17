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
        self._prev_clock_sims = None
        self._manual = False

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

    def _group_text(self) -> str:
        return str(self.ownerComp.par.Groupsims.eval()).strip()

    def _sims(self):
        text = self._group_text()
        if not text:
            return [self.Sim]
        return [op(path) for path in text.split() if op(path) is not None]  # noqa: F821

    def _member_record(self, sim) -> dict:
        bus = op.simbus.ext.FieldBus  # noqa: F821
        params = sim.Params()
        res = list(params.get("res", list(bus.Res)))
        bounds = list(params.get("bounds", list(bus.Bounds)))
        extra = {}
        if len(res) >= 3:
            extra["atlas"] = self.cache.atlas_layout(res[:3])
        storage = self.Storage()
        if storage:
            extra["storage"] = storage
        return {"params": params, "fields": sim.StateFormats(), "res": res, "bounds": bounds, "extra": extra}

    def Storage(self) -> dict:
        """Reduced-cache layout. Empty means every frame stores every state field, as in phases 1-3."""
        p = self.ownerComp.par
        out = {}
        interval = int(p.Keyinterval.eval())
        if interval > 1:
            text = str(p.Playbackfields.eval()).strip()
            out["keyframe_interval"] = interval
            out["playback_fields"] = text.split() if text else ["scalars"]
        if bool(p.Crop.eval()):
            out["crop"] = True
            out["crop_fields"] = ["scalars"]
            out["crop_pad"] = int(p.Croppad.eval())
        return out

    def NewManifest(self):
        p = self.ownerComp.par
        clock = op.simclock  # noqa: F821
        common = dict(
            fps=clock.par.Fps.eval(), substeps=int(clock.par.Substeps.eval()),
            frame_start=int(p.Framestart.eval()), frame_end=int(p.Frameend.eval()),
            td_version=f"{app.version}.{app.build}",  # noqa: F821
        )
        if not self._group_text():
            sim = self.Sim
            info = self._member_record(sim)
            return self.cache.Manifest(
                sim=sim.name, params=info["params"], fields=info["fields"], res=info["res"], bounds=info["bounds"],
                extra=info["extra"], **common,
            )
        sims = self._sims()
        members = {sim.name: self._member_record(sim) for sim in sims}
        first = members[sims[0].name]
        return self.cache.Manifest(
            sim="group", params={name: info["params"] for name, info in members.items()},
            fields={}, res=first["res"], bounds=first["bounds"], members=members, **common,
        )

    def CacheStatus(self) -> str:
        p = self.ownerComp.par
        params = self.NewManifest().params
        return self.cache.cache_status(
            self.CacheDir, params, int(p.Framestart.eval()), int(p.Frameend.eval()), storage=self.Storage(),
        )

    # -- actions ----------------------------------------------------------------------------------------------
    def Bake(self) -> None:
        if self.baking:
            return
        p, clock = self.ownerComp.par, op.simclock  # noqa: F821
        sim, cache_dir = self.Sim, self.CacheDir
        manifest = self.NewManifest()
        plan = self.cache.directory_plan(
            cache_dir, manifest.params, manifest.frame_start, manifest.frame_end, storage=self.Storage(),
        )
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
        self._prev_clock_sims = None
        if self._group_text():
            self._prev_clock_sims = str(clock.par.Sims.eval())
            clock.par.Sims = self._group_text()
        if resume > start:
            if manifest.members:
                for one in self._sims():
                    info = manifest.members[one.name]
                    one.SetInit({f: self.cache.frame_path(cache_dir, f, resume - 1, one.name) for f in info["fields"]})
                self._show_loaded_fluid()
                self._restore_coupling(resume - 1)
                clock.RequestReset(resume)
            else:
                origin = self.cache.resume_state_frame(manifest, resume)
                if origin is None:
                    sim.SetInit(None)
                    clock.RequestReset(start - warmup)
                else:
                    fields = self.cache.fields_for_frame(manifest, origin)
                    sim.SetInit({f: self.cache.frame_path(cache_dir, f, origin) for f in fields})
                    # The loaded keyframe is the state after `origin`. The reset tick simulates the next frame.
                    # When every frame is a keyframe, origin is resume - 1 and this is the phase 1 resume.
                    clock.RequestReset(origin + 1)
        else:
            for one in self._sims():
                one.SetInit(None)
            clock.RequestReset(start - warmup)

        self._prev_realtime = project.realTime  # noqa: F821
        project.realTime = False  # noqa: F821
        clock.par.Mode = "bake"
        clock.par.Running = True
        self.baking = True
        self._t0 = time.perf_counter()
        self._frames_this_run = 0
        self._manual = bool(self._group_text())
        if not self._manual:
            clock.AddListener(self.ownerComp)
        self._set_status(f"baking from frame {resume}")

    def PumpOnce(self) -> bool:
        """One group-bake step. Saving happens after Tick returns so the solved null is stable."""
        if not self.baking:
            return False
        op.simclock.Tick()  # noqa: F821
        self._save_frame()
        return self.baking

    def OnStep(self, clock_ext) -> None:
        if not self.baking or self._manual:
            return
        self._save_frame()

    def _show_loaded_fluid(self) -> None:
        """Put the resumed EXR on the bus snapshot so the next particle step reads that frame."""
        for one in self._sims():
            if not one.name.startswith("fluid"):
                continue
            for field in one.StateFormats():
                held = one.op(field + "_out")
                init = one.op("init_" + field)
                state = one.op(field + "_state")
                if held is None or init is None or state is None:
                    continue
                held.lock = False
                held.inputConnectors[0].disconnect()
                init.outputConnectors[0].connect(held.inputConnectors[0])
                held.par.replace = True
                held.par.replaceindex = 0
                held.par.replacespulse.pulse()
                held.par.replace = False
                held.lock = True
                held.inputConnectors[0].disconnect()
                state.outputConnectors[0].connect(held.inputConnectors[0])

    def _restore_coupling(self, frame: int) -> None:
        """Lock the force and source grids the next fluid step must see.

        A resumed tick steps the fluid before the particles rewrite those grids, so the
        images saved at the end of the previous frame have to be put back first.
        """
        particles = next((one for one in self._sims() if one.name == "particles"), None)
        if particles is None:
            return
        import numpy as np  # noqa: PLC0415

        holder = self.ownerComp.op("force_load")
        if holder is None:
            holder = self.ownerComp.create(moviefileinTOP, "force_load")
            holder.par.playmode = "locked"
            holder.par.outputresolution = "native"
        folder = Path(project.folder)  # noqa: F821
        bus = op.simbus  # noqa: F821
        table = bus.op("publishers")
        for kind in ("forces", "sources"):
            path = self.cache.frame_path(self.CacheDir, kind, frame, "particles")
            if not path.is_absolute():
                path = folder / path
            live = particles.op(kind + "_2d")
            if live is None or not path.exists():
                continue
            holder.par.file = path.as_posix()
            holder.cook(force=True)
            image = np.array(holder.numpyArray(delayed=False), dtype=np.float32, copy=True)
            hold = self.ownerComp.op("hold_" + kind)
            if hold is None:
                hold = self.ownerComp.create(scriptTOP, "hold_" + kind)
            hold.par.outputresolution = "custom"
            hold.par.resolutionw = int(image.shape[1])
            hold.par.resolutionh = int(image.shape[0])
            hold.par.format = "rgba32float"
            hold.lock = True
            hold.copyNumpyArray(image)
            for row in range(1, table.numRows):
                if table[row, 0].val == kind and table[row, 1].val == live.path:
                    table[row, 1] = hold.path
        op.simbus.ext.FieldBus._refresh()  # noqa: F821

    def _maybe_crop(self, field: str, top, frame: int):
        """Return a cropped atlas TOP for playback fields between keyframes, or `top` unchanged."""
        storage = (self.manifest.extra or {}).get("storage") or {}
        if not storage.get("crop") or field not in (storage.get("crop_fields") or []):
            return top
        if self.cache.is_keyframe(self.manifest, frame):
            return top
        sim = self.Sim
        if not hasattr(sim, "CropAtlas") or not hasattr(sim, "OccupiedBbox"):
            return top
        if getattr(self, "_bbox_frame", None) != int(frame):
            self._bbox_frame = int(frame)
            self._bbox_value = list(sim.OccupiedBbox(int(storage.get("crop_pad") or 0)))
            self.manifest.extra.setdefault("bboxes", {})[str(int(frame))] = self._bbox_value
        return sim.CropAtlas(field, self._bbox_value)

    def _save_frame(self) -> None:
        if not self.baking:
            return
        c, m = op.simclock.ext.SimClockExt.Clock, self.manifest  # noqa: F821
        # Only the last substep of an output frame is the state that gets saved.
        if not c.frame_complete:
            return
        asynchronous = bool(self.ownerComp.par.Asyncsave.eval())
        if m.members:
            targets = []
            for one in self._sims():
                for field, top in one.StateTops().items():
                    if one.name.startswith("fluid"):
                        state = one.op(field + "_state")
                        if state is not None:
                            top = state
                    targets.append((one.name, field, top))
        else:
            targets = [(None, field, top) for field, top in self.Sim.StateTops().items()]
        needed = None if m.members else set(self.cache.fields_for_frame(m, c.frame))
        for sim_name, field, top in targets:
            if needed is not None and field not in needed:
                continue
            if needed is not None:
                top = self._maybe_crop(field, top, c.frame)
            path = self.cache.frame_path(self.CacheDir, field, c.frame, sim_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            top.save(str(path), asynchronous=asynchronous)
            if sim_name and sim_name.startswith("fluid"):
                # save() cooks the solved null. Copy that image into the public snapshot
                # so a replay of the snapshot matches the file.
                held = op(f"/sims/{sim_name}/{field}_out")  # noqa: F821
                if held is not None:
                    held.lock = False
                    held.par.replace = True
                    held.par.replaceindex = 0
                    held.par.replacespulse.pulse()
                    held.par.replace = False
                    held.lock = True
        if m.members:
            particles = next((one for one in self._sims() if one.name == "particles"), None)
            if particles is not None:
                for field, node_name in (("forces", "forces_2d"), ("sources", "sources_2d")):
                    node = particles.op(node_name)
                    if node is None:
                        continue
                    path = self.cache.frame_path(self.CacheDir, field, c.frame, "particles")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    node.save(str(path), asynchronous=asynchronous)
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
        if self._prev_clock_sims is not None:
            clock.par.Sims = self._prev_clock_sims
        if self._prev_realtime is not None:
            project.realTime = self._prev_realtime  # noqa: F821
        m = self.manifest
        m.complete = complete and self.cache.resume_frame(self.CacheDir, m) > m.frame_end
        self.cache.write_manifest(self.CacheDir, m)
        self.baking = False
        secs = time.perf_counter() - self._t0
        self._set_status(f"{'complete' if m.complete else 'cancelled'}: {self._frames_this_run} frames in {secs:.1f}s")
