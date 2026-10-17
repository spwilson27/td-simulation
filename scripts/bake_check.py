"""End-to-end check of baking, resuming, staleness and cache playback in the scratch TD.

    python scripts/bake_check.py [--frames 40] [--preset realtime]

1. Full bake into cache/check_full.
2. Bake into cache/check_resume, cancel part way, bake again: it must resume, and every frame file must be
   byte-identical to the full bake (the saved state is complete, so resuming is exact).
3. Changing a physics parameter makes the cache stale; restoring it makes it complete again.
4. The CacheReader's render of the last frame matches the live render pixel for pixel (the baker pauses the clock
   on the last frame), and a half frame is the average of its neighbours.
Results go to reports/bake_check.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from td_scratch import ROOT, attach  # noqa: E402

CONFIG = r'''
import json
b = op('/sims/baker')
b.par.Sim = '/sims/fluid2d'
b.par.Groupsims = ''
op.simclock.par.Sims = '/sims/fluid2d'
op('/sims/fluid2d').ApplyPreset({preset!r})
b.par.Cachedir = {cache!r}
b.par.Framestart, b.par.Frameend, b.par.Resume, b.par.Warmup = 1, {frames}, True, 0
print('@@JSON@@' + json.dumps(b.ClearCache() if {clear} else 0))
'''
BAKE = "op('/sims/baker').Bake()\nprint('@@JSON@@' + __import__('json').dumps(op('/sims/baker').par.Status.eval()))"
CANCEL = "op('/sims/baker').Cancel()\nprint('@@JSON@@' + __import__('json').dumps(op('/sims/baker').par.Status.eval()))"
STATUS = r'''
import json
b = op('/sims/baker')
print('@@JSON@@' + json.dumps({'status': b.par.Status.eval(), 'baking': b.ext.Baker.baking, 'frame': op.simclock.par.Frame.eval(),
                               'cache': b.CacheStatus(), 'realtime': project.realTime}))
'''
STALE = r'''
import json
f, b = op('/sims/fluid2d'), op('/sims/baker')
before = b.CacheStatus()
old = f.par.Vorticity.eval()
f.par.Vorticity = old + 0.5
changed = b.CacheStatus()
f.par.Vorticity = old
print('@@JSON@@' + json.dumps({'before': before, 'after_change': changed, 'restored': b.CacheStatus()}))
'''
REPLAY = r'''
import json, numpy as np
r = op('/sims/cache_reader')
r.par.Cachedir = {cache!r}
r.Reload()
r.par.Frame.expr = ''
end = int(r.par.Frameend.eval())
r.par.Interpolate = False
r.Load(end)
live = op('/sims/fluid2d/render_out').numpyArray(delayed=False)
r.op('render').cook(force=True)
rd = r.op('render_out'); rd.cook(force=True)
cached = rd.numpyArray(delayed=False)
live_sc = op('/sims/fluid2d/scalars_out').numpyArray(delayed=False)
cached_sc = r.op('scalars_out').numpyArray(delayed=False)
a_sc = cached_sc.copy()
r.Load(end - 1)
b_sc = r.op('scalars_out').numpyArray(delayed=False).copy()
r.par.Interpolate = True
r.Load(end - 0.5)
mid = r.op('scalars_out').numpyArray(delayed=False)
project.folder and r.op('render_out').save(project.folder + '/reports/frame_cached.png')
op('/sims/fluid2d/render_out').save(project.folder + '/reports/frame_live_end.png')
print('@@JSON@@' + json.dumps({{'status': r.par.Status.eval(), 'frame': end, 'clock_frame': op.simclock.par.Frame.eval(),
    'render_max_abs_diff': float(np.abs(live - cached).max()), 'scalars_max_abs_diff': float(np.abs(live_sc - cached_sc).max()),
    'half_frame_max_err': float(np.abs(mid - 0.5 * (a_sc + b_sc)).max()), 'neighbour_frames_differ': float(np.abs(a_sc - b_sc).max()),
    'scalars_abs_max': float(max(np.abs(a_sc).max(), np.abs(b_sc).max()))}}))
r.par.Frame.expr = 'op.simclock.par.Frame'
'''


def wait_done(td, timeout=600, cancel_at=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = td.run_json(STATUS)
        if cancel_at is not None and s["baking"] and s["frame"] >= cancel_at:
            td.run_json(CANCEL)
            return td.run_json(STATUS), time.time() - t0
        if not s["baking"]:
            return s, time.time() - t0
        time.sleep(0.25)
    raise SystemExit("bake did not finish in time")


def files(cache: Path):
    return sorted(p for p in cache.rglob("*.exr"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--preset", default="realtime")
    args = ap.parse_args(argv)
    td = attach()
    report = {"frames": args.frames, "preset": args.preset}

    full, resume = "cache/check_full", "cache/check_resume"
    td.run_json(CONFIG.format(preset=args.preset, cache=full, frames=args.frames, clear=True))
    td.run_json(BAKE)
    s, secs = wait_done(td)
    report["full_bake"] = {**s, "seconds": round(secs, 2), "fps": round(args.frames / secs, 2)}
    print("full bake:", report["full_bake"])

    td.run_json(CONFIG.format(preset=args.preset, cache=resume, frames=args.frames, clear=True))
    td.run_json(BAKE)
    s, _ = wait_done(td, cancel_at=args.frames // 2)
    report["cancelled"] = s
    print("cancelled:", s)
    td.run_json(BAKE)
    report["resume_started"] = td.run_json(STATUS)["status"]
    s, _ = wait_done(td)
    report["resumed"] = s
    print("resumed:", s)

    a, b = files(ROOT / full), files(ROOT / resume)
    rel_a = [p.relative_to(ROOT / full) for p in a]
    rel_b = [p.relative_to(ROOT / resume) for p in b]
    mismatched = [str(r) for r in rel_a if r not in rel_b or (ROOT / full / r).read_bytes() != (ROOT / resume / r).read_bytes()]
    report["resume_identical"] = {"files": len(a), "same_names": rel_a == rel_b, "mismatched": mismatched[:10]}
    print("resume identical:", report["resume_identical"]["same_names"] and not mismatched, f"({len(a)} files)")

    report["stale"] = td.run_json(STALE)
    print("stale check:", report["stale"])
    report["replay"] = td.run_json(REPLAY.format(cache=full))
    print("replay:", report["replay"])
    sizes = sum(p.stat().st_size for p in a)
    report["cache_bytes"] = sizes
    (ROOT / "reports" / "bake_check.json").write_text(json.dumps(report, indent=2) + "\n")
    # Scalars are rgba16float: allow one half-precision ulp at the largest magnitude (2^-10 relative), floor 1e-3.
    blend_tol = max(1e-3, report["replay"].get("scalars_abs_max", 0.0) * 2.0 ** -10)
    ok = (report["full_bake"]["cache"] == "complete" and not mismatched and report["resume_identical"]["same_names"]
          and report["stale"]["after_change"] == "stale" and report["stale"]["restored"] == "complete"
          and report["replay"]["render_max_abs_diff"] == 0.0 and report["replay"]["half_frame_max_err"] < blend_tol)
    print("ALL OK" if ok else "CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
