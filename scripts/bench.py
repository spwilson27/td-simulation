"""Frame-rate and GPU cook-time benchmark of /sims/fluid2d in the scratch TouchDesigner.

    python scripts/bench.py [--seconds 3]

For preview (512), realtime (1024), and realtime settings at 1536 and 2048:

  uncapped   project.realTime = False, frames advanced over ~3 s of wall time, then realTime restored
  gpu        last gpuCookTime of each fluid2d GLSL TOP (milliseconds, as TouchDesigner reports it)
  realtime   project.realTime = True at cookRate 60, same wall-time window

Other TouchDesigner processes on this machine share the GPU; the numbers include that load.
Writes reports/bench_fluid2d.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from td_scratch import ROOT, attach  # noqa: E402

RUNS = (
    {"name": "preview", "preset": "preview", "width": 0},
    {"name": "realtime", "preset": "realtime", "width": 0},
    {"name": "1536", "preset": "realtime", "width": 1536},
    {"name": "2048", "preset": "realtime", "width": 2048},
)

SETUP = r'''
import json
f = op('/sims/fluid2d')
f.ApplyPreset({preset!r})
if {width}:
    op.simbus.ext.FieldBus.SetGrid({width})
clock = op.simclock
clock.par.Mode = 'realtime'
clock.par.Running = True
clock.RequestReset()
print('@@JSON@@' + json.dumps({{'res': list(op.simbus.ext.FieldBus.Res), 'substeps': int(clock.par.Substeps.eval()),
    'jacobi': int(f.par.Jacobi.eval()), 'preset': f.par.Preset.eval(), 'cookRate': float(project.cookRate),
    'realTime': bool(project.realTime)}}))
'''
REALTIME = r'''
import json
project.realTime = {flag}
print('@@JSON@@' + json.dumps(bool(project.realTime)))
'''
SAMPLE = r'''
import json
f = op('/sims/fluid2d')
glsl = []
for o in sorted(f.findChildren(type=glslTOP, depth=1), key=lambda o: o.name):
    glsl.append({'name': o.name, 'gpuCookTime': float(o.gpuCookTime), 'cpuCookTime': float(o.cpuCookTime),
                  'cookTime': float(o.cookTime)})
print('@@JSON@@' + json.dumps({'frame': int(absTime.frame), 'realTime': bool(project.realTime),
    'cookRate': float(project.cookRate), 'glsl': glsl}))
'''
RESTORE = r'''
import json
project.realTime = True
project.cookRate = {cook_rate}
op('/sims/fluid2d').ApplyPreset('realtime')
op.simclock.par.Mode = 'realtime'
op.simclock.par.Running = True
print('@@JSON@@' + json.dumps({{'realTime': bool(project.realTime), 'cookRate': float(project.cookRate),
    'preset': op('/sims/fluid2d').par.Preset.eval(), 'res': list(op.simbus.ext.FieldBus.Res)}}))
'''


def measure(td, seconds: float) -> dict:
    snap = "import json\nprint('@@JSON@@' + json.dumps({'frame': int(absTime.frame), 'step': int(op.simclock.par.Step.eval())}))"
    start = td.run_json(snap)
    t0 = time.perf_counter()
    time.sleep(seconds)
    end = td.run_json(snap)
    dt = time.perf_counter() - t0
    frames = end["frame"] - start["frame"]
    steps = end["step"] - start["step"]
    return {"frames": frames, "steps": steps, "seconds": round(dt, 3),
            "fps": round(frames / dt, 2) if dt else 0.0, "step_fps": round(steps / dt, 2) if dt else 0.0}


def gpu_times(td) -> list[dict]:
    """The first read of gpuCookTime returns the previous query. Settle, then read the resolved sample."""
    td.run_json(SAMPLE)
    time.sleep(0.5)
    return td.run_json(SAMPLE)["glsl"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args(argv)
    td = attach()
    cook_rate = td.run_json("import json\nprint('@@JSON@@' + json.dumps(float(project.cookRate)))")
    report = {"seconds": args.seconds, "cook_rate": 60, "gpuCookTime_unit": "ms",
              "note": "Other TouchDesigner instances on this machine share the GPU.", "runs": []}
    try:
        for spec in RUNS:
            info = td.run_json(SETUP.format(preset=spec["preset"], width=spec["width"]))
            time.sleep(0.5)
            td.run_json(REALTIME.format(flag=False))
            time.sleep(0.25)
            uncapped = measure(td, args.seconds)
            glsl = gpu_times(td)
            gpu = {g["name"]: round(g["gpuCookTime"], 4) for g in glsl}
            td.run_json(REALTIME.format(flag=True))
            td.run_json(f"project.cookRate = 60\nprint('@@JSON@@' + __import__('json').dumps(float(project.cookRate)))")
            time.sleep(0.5)
            realtime = measure(td, args.seconds)
            row = {"name": spec["name"], "preset": info["preset"], "res": info["res"], "substeps": info["substeps"],
                   "jacobi": info["jacobi"], "uncapped_fps": uncapped["fps"], "uncapped_frames": uncapped["frames"],
                   "realtime_fps": realtime["fps"], "realtime_frames": realtime["frames"],
                   "uncapped_step_fps": uncapped["step_fps"], "realtime_step_fps": realtime["step_fps"],
                   "gpuCookTime_ms": gpu, "gpuCookTime_ms_sum": round(sum(gpu.values()), 4)}
            report["runs"].append(row)
            print(f"{row['name']:10} {row['res'][0]:4}x{row['res'][1]:<4}  uncapped {row['uncapped_fps']:7.2f} fps"
                  f" (steps {row['uncapped_step_fps']:7.2f})  gpu {row['gpuCookTime_ms_sum']:8.3f} ms"
                  f"   realtime {row['realtime_fps']:7.2f} fps (steps {row['realtime_step_fps']:7.2f})")
    finally:
        restored = td.run_json(RESTORE.format(cook_rate=cook_rate))
        print("restored:", restored)
    for prev, row in zip(report["runs"], report["runs"][1:]):
        if row["gpuCookTime_ms"] == prev["gpuCookTime_ms"]:
            row["gpuCookTime_stale"] = True
    jacobis = [r["gpuCookTime_ms"].get("jacobi") for r in report["runs"]]
    if len(jacobis) >= 2 and len(set(jacobis)) == 1:
        report["gpuCookTime_note"] = (
            "The multipass jacobi TOP reported the same gpuCookTime at every resolution, so that timestamp had "
            "latched. A later row marked gpuCookTime_stale repeated the previous row's whole sample. "
            "Uncapped fps is the cost that tracked resolution."
        )
    capped = [r["name"] for r in report["runs"] if r["uncapped_fps"] >= 119.0]
    if capped:
        report["uncapped_note"] = (
            f"{', '.join(capped)} reached about 120 fps with realTime off, consistent with a 120 Hz refresh cap. "
            "That figure is a lower bound on throughput, not a measured GPU ceiling."
        )
    realtime = next(r for r in report["runs"] if r["name"] == "realtime")
    ok = realtime["realtime_fps"] >= 30.0
    report["realtime_preset_fps"] = realtime["realtime_fps"]
    report["acceptance_30fps"] = ok
    (ROOT / "reports" / "bench_fluid2d.json").write_text(json.dumps(report, indent=2) + "\n")
    print("realtime preset", f"{realtime['realtime_fps']:.2f} fps", "PASS" if ok else "FAIL", "(need >= 30, target 60)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
