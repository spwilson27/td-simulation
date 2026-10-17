"""Phase 2 checks. Exit 0 is pass. Writes reports/phase2.json and reports/phase2_*.png.

Touches only the scratch TouchDesigner from td_scratch.py. Long cooks are split into short calls;
one multi-minute execute on the TD main thread has killed the process on this machine.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from td_scratch import ROOT, attach  # noqa: E402

SETUP = r'''
import json
f = op('/sims/fluid3d')
f.allowCooking = True
for name in ('advect_vel', 'mac_vel', 'advect_scalars', 'mac_scalars', 'combust', 'forces', 'divergence',
             'project', 'vel_state', 'scalars_state', 'pressure_state', 'shadow', 'render', 'render_out'):
    node = f.op(name)
    if node is not None:
        node.bypass = False
op('/sims/baker').Cancel()
op('/sims/clock/tick').par.active = False
f.par.Cookrender = False
f.par.Solver = 'jacobi'
f.par.Jacobi = 12
f.par.Velocitydecay = 0.4
f.par.Vorticity = 1.2
f.par.Maccormack = 0.4
f.par.Buoyancy = 1.6
f.par.Maxspeed = 6
f.par.Renderw = 1280
f.par.Renderh = 720
c = op.simclock
c.par.Startframe = 0
c.par.Fps = 60
c.par.Substeps = 1
c.par.Mode = 'realtime'
c.par.Running = True
c.par.Sims = '/sims/fluid3d'
project.realTime = False
c.RequestReset()
c.Tick()
print('@@JSON@@' + json.dumps({'res': list(op.simbus3d.Res), 'step': int(c.par.Step.eval())}))
'''
STEPS = r'''
import json, numpy as np
f = op('/sims/fluid3d')
c = op.simclock
for _ in range({n}):
    c.Tick()
vel = f.op('vel_out').numpyArray(delayed=False)
sc = f.op('scalars_out').numpyArray(delayed=False)
speed = float(np.sqrt((np.nan_to_num(vel[..., :3]) ** 2).sum(-1)).max())
print('@@JSON@@' + json.dumps({{'step': int(c.par.Step.eval()), 'speed': speed,
    'finite': bool(np.isfinite(vel).all() and np.isfinite(sc).all()),
    'temp': float(np.nanmax(sc[..., 1])), 'smoke': float(np.nanmax(sc[..., 0]))}}))
'''
OBSTACLES = r'''
import json, numpy as np
f = op('/sims/fluid3d')
d = op('/sims/demo3d')
c = op.simclock
t = float(c.par.Time.eval())
obs = op('/sims/bus3d/obstacles').numpyArray(delayed=False)
sc = f.op('scalars_out').numpyArray(delayed=False)
b = op.simbus3d.Bounds
res = op.simbus3d.Res

def at(world):
    uv = [(world[i] - b[i]) / (b[i + 3] - b[i]) for i in range(3)]
    x = int(np.clip(round(uv[0] * res[0] - 0.5), 0, res[0] - 1))
    y = int(np.clip(round(uv[1] * res[1] - 0.5), 0, res[1] - 1))
    z = int(np.clip(round(uv[2] * res[2] - 0.5), 0, res[2] - 1))
    return {'sdf': float(obs[z, y, x, 0]), 'smoke': float(sc[z, y, x, 0]), 'ijk': [x, y, z]}

sx = float(d.par.Spherex) + float(d.par.Sphereamp) * np.sin(t * float(d.par.Spherespeed))
sy, sz = float(d.par.Spherey), float(d.par.Spherez)
mx = float(d.par.Meshx) + float(d.par.Meshamp) * np.sin(t * float(d.par.Meshspeed))
my, mz = float(d.par.Meshy), float(d.par.Meshz)
pop = d.Triangles(d.op('box_pop'))
sop = d.Triangles(d.op('mesh_geo/sphere'))
print('@@JSON@@' + json.dumps({
    'sphere': at((sx, sy, sz)), 'mesh': at((mx, my, mz)),
    'pop_triangles': len(pop), 'sop_triangles': len(sop),
    'mesh_sdf_min': float(d.par.Sdfminx.eval()),
}))
'''
RENDER = r'''
import json
f = op('/sims/fluid3d')
f.par.Smokedensity = 5.5
f.par.Fireintensity = 4.5
f.par.Whitehot = 2.4
f.par.Raysteps = 48
f.allowCooking = True
f.op('render').cook(force=True)
f.op('render_out').save(project.folder + '/reports/phase2_fire.png' if {mid} else project.folder + '/reports/phase2_late.png')
print('@@JSON@@' + json.dumps({{'err': f.op('render').errors() or ''}}))
'''
FPS_MARK = r'''
import json
op('/sims/clock/tick').par.active = True
op('/sims/fluid3d').par.Cookrender = {render}
project.realTime = {realtime}
print('@@JSON@@' + json.dumps({{'step': int(op.simclock.par.Step.eval()), 'frame': int(absTime.frame)}}))
'''
PRESSURE = r'''
import json, time
f = op('/sims/fluid3d')
op('/sims/clock/tick').par.active = False
c = op.simclock
c.par.Sims = '/sims/fluid3d'
def div_after(steps, solver, jacobi):
    f.par.Solver = solver
    f.par.Jacobi = jacobi
    f.par.Coarsegain = 0.05
    c.RequestReset(); c.Tick()
    t0 = time.perf_counter()
    for _ in range(steps):
        c.Tick()
    sec = time.perf_counter() - t0
    import numpy as np
    pre = f.op('divergence').numpyArray(delayed=False)[..., 0]
    post = f.op('div_check')
    post.cook(force=True)
    d = post.numpyArray(delayed=False)[..., 0]
    return {'seconds': sec, 'div_pre': float(np.sqrt(np.mean(pre ** 2))), 'div_post': float(np.sqrt(np.mean(d ** 2)))}
jac = div_after(6, 'jacobi', 12)
mg = div_after(6, 'multigrid', 12)
print('@@JSON@@' + json.dumps({'jacobi12': jac, 'multigrid': mg}))
'''
BAKE_SETUP = r'''
import json
b = op('/sims/baker')
f = op('/sims/fluid3d')
op('/sims/clock/tick').par.active = False
f.par.Cookrender = False
f.par.Solver = 'jacobi'
f.par.Jacobi = 8
c = op.simclock
c.par.Sims = '/sims/fluid3d'
c.par.Running = True
b.par.Sim = '/sims/fluid3d'
b.par.Cachedir = {cache!r}
b.par.Framestart = 1
b.par.Frameend = {frames}
b.par.Resume = True
b.par.Warmup = 0
cleared = b.ClearCache() if {clear} else 0
print('@@JSON@@' + json.dumps(cleared))
'''
BAKE = "op('/sims/baker').Bake()\nprint('@@JSON@@' + __import__('json').dumps(op('/sims/baker').par.Status.eval()))"
CANCEL = "op('/sims/baker').Cancel()\nprint('@@JSON@@' + __import__('json').dumps(op('/sims/baker').par.Status.eval()))"
PUMP = r'''
import json
b = op('/sims/baker')
op('/sims/clock/tick').par.active = False
c = op.simclock
c.par.Running = True
n = 0
for _ in range({n}):
    if not b.ext.Baker.baking:
        break
    c.Tick()
    n += 1
replay = None
# The match has to be read in this same call. A later script cooks the solver again (a 3D Cache TOP still
# pulls a dirty input) and the live snapshot moves off the frame that was just saved.
if {replay} and not b.ext.Baker.baking:
    import numpy as np
    r = op('/sims/cache_reader')
    r.par.Cachedir = {cache!r}
    r.Reload()
    r.par.Frame.expr = ''
    r.par.Interpolate = False
    end = int(r.par.Frameend.eval())
    r.Load(end)
    fields = {{}}
    for field in ('vel', 'scalars', 'pressure'):
        live = op('/sims/fluid3d/' + field + '_out').numpyArray(delayed=False)
        cached = r.Output(field).numpyArray(delayed=False)
        if cached is None or live is None or cached.shape != live.shape:
            fields[field] = None
        else:
            fields[field] = float(np.max(np.abs(live.astype('float32') - cached.astype('float32'))))
    r.par.Frame.expr = 'op.simclock.par.Frame'
    replay = None if any(v is None for v in fields.values()) else max(fields.values())
    replay = {{'diff': replay, 'fields': fields, 'frame': end, 'status': r.par.Status.eval()}}
print('@@JSON@@' + json.dumps({{'status': b.par.Status.eval(), 'baking': b.ext.Baker.baking,
    'frame': int(c.par.Frame.eval()), 'pumped': n, 'replay': replay}}))
'''


def wait_bake(td, cache, timeout=300, cancel_at=None, replay=False):
    """Drive the bake with Tick() from here. The frame-start execute stays off so a step is not applied twice,
    and so a timeline that has run past its range cannot stall the bake."""
    t0 = time.time()
    state = {"baking": True, "frame": 0}

    def pump(n, do_replay):
        return td.run_json(PUMP.format(n=n, replay="True" if do_replay else "False", cache=cache), timeout=90)

    while time.time() - t0 < timeout:
        state = pump(8, replay)
        if cancel_at is not None and state["baking"] and state["frame"] >= cancel_at:
            td.run_json(CANCEL, timeout=20)
            return pump(0, False)
        if not state["baking"]:
            return state
        if state.get("pumped") == 0 and state["baking"]:
            raise SystemExit("3D bake stopped advancing: " + json.dumps(state))
    raise SystemExit("3D bake did not finish")


def files(cache: Path):
    return sorted(p for p in cache.rglob("*.exr"))


def main() -> int:
    td = attach()
    report = {"steps_target": 600}
    report["setup"] = td.run_json(SETUP, timeout=30)
    samples = []
    finite = True
    max_speed = 0.0
    done = 0
    while done < 600:
        n = min(20, 600 - done)
        row = td.run_json(STEPS.format(n=n), timeout=40)
        samples.append(row)
        finite = finite and row["finite"]
        max_speed = max(max_speed, row["speed"])
        done += n
        print(f"step {row['step']}: speed {row['speed']:.3f} finite {row['finite']} temp {row['temp']:.2f}")
        if done == 200:
            # The plume is up and wrapping the obstacles here. By step 600 most of it has left the open top.
            report["render_mid"] = td.run_json(RENDER.format(mid="True"), timeout=40)
        if not row["finite"] or row["speed"] > 20:
            break
    report["stability"] = {"samples": samples[::5], "finite": finite, "max_speed": max_speed, "steps": samples[-1]["step"]}
    report["obstacles"] = td.run_json(OBSTACLES, timeout=40)
    print("obstacles", report["obstacles"])
    report["render"] = td.run_json(RENDER.format(mid="False"), timeout=40)
    # timeline fps, render cooking
    start = td.run_json(FPS_MARK.format(render="True", realtime="False"), timeout=20)
    time.sleep(3.0)
    end = td.run_json("import json\nprint('@@JSON@@'+json.dumps({'step': int(op.simclock.par.Step.eval()), 'frame': int(absTime.frame)}))", timeout=20)
    dt = 3.0
    report["uncapped"] = {"steps": end["step"] - start["step"], "frames": end["frame"] - start["frame"],
                          "fps": round((end["frame"] - start["frame"]) / dt, 2)}
    start = td.run_json(FPS_MARK.format(render="True", realtime="True"), timeout=20)
    td.run_json("import json\nproject.cookRate = 60\nprint('@@JSON@@'+json.dumps(1))", timeout=15)
    time.sleep(3.0)
    end = td.run_json("import json\nprint('@@JSON@@'+json.dumps({'step': int(op.simclock.par.Step.eval()), 'frame': int(absTime.frame)}))", timeout=20)
    report["capped"] = {"steps": end["step"] - start["step"], "frames": end["frame"] - start["frame"],
                        "fps": round((end["frame"] - start["frame"]) / dt, 2)}
    print("fps", report["uncapped"], report["capped"])
    td.run_json("import json\nop('/sims/clock/tick').par.active = False\nproject.realTime = False\nprint('@@JSON@@'+json.dumps(1))", timeout=15)
    try:
        report["pressure"] = td.run_json(PRESSURE, timeout=90)
    except Exception as exc:
        report["pressure"] = {"error": str(exc)}
    print("pressure", report["pressure"])

    full, resume = "cache/phase2_full", "cache/phase2_resume"
    td.run_json(BAKE_SETUP.format(cache=full, frames=24, clear="True"), timeout=30)
    td.run_json(BAKE, timeout=20)
    report["full_bake"] = wait_bake(td, full, replay=True)
    report["replay"] = report["full_bake"].get("replay")
    td.run_json("import json\nop('/sims/clock/tick').par.active = False\nop.simclock.par.Running = False\nprint('@@JSON@@'+json.dumps(1))", timeout=15)
    print("full bake", report["full_bake"])
    td.run_json(BAKE_SETUP.format(cache=resume, frames=24, clear="True"), timeout=30)
    td.run_json(BAKE, timeout=20)
    report["cancelled"] = wait_bake(td, resume, cancel_at=12)
    td.run_json(BAKE, timeout=20)
    report["resumed"] = wait_bake(td, resume)
    td.run_json("import json\nop('/sims/clock/tick').par.active = False\nop.simclock.par.Running = False\nprint('@@JSON@@'+json.dumps(1))", timeout=15)
    a, b = files(ROOT / full), files(ROOT / resume)
    rel_a = [p.relative_to(ROOT / full) for p in a]
    rel_b = [p.relative_to(ROOT / resume) for p in b]
    mismatched = [str(r) for r in rel_a if r not in rel_b or (ROOT / full / r).read_bytes() != (ROOT / resume / r).read_bytes()]
    report["resume_identical"] = {"files": len(a), "same_names": rel_a == rel_b, "mismatched": mismatched[:8]}
    print("replay", report["replay"], "identical", not mismatched)

    # hand the clock back to the 2D sim before the phase 1 check
    td.run_json("""
op('/sims/clock/tick').par.active = True
op('/sims/fluid2d').allowCooking = True
op('/sims/demo').allowCooking = True
op.simclock.par.Sims = '/sims/fluid2d'
op('/sims/fluid2d').par.Cookrender = False
project.realTime = True
print('@@JSON@@' + __import__('json').dumps(op.simclock.par.Sims.eval()))
""", timeout=20)
    phase1 = subprocess.run([sys.executable, str(ROOT / "scripts" / "bake_check.py"), "--frames", "40"], cwd=ROOT)
    report["phase1_bake_check"] = phase1.returncode
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "phase2.json").write_text(json.dumps(report, indent=2) + "\n")

    sphere_sdf = report["obstacles"]["sphere"]["sdf"]
    mesh_sdf = report["obstacles"]["mesh"]["sdf"]
    pressure = report.get("pressure") or {}
    jac = pressure.get("jacobi12") or {}
    mg = pressure.get("multigrid") or {}
    div_factor = None
    if jac.get("div_post"):
        div_factor = jac["div_pre"] / jac["div_post"]
    beats = False
    if jac.get("div_post") and mg.get("div_post") and jac.get("seconds") and mg.get("seconds"):
        # lower divergence after the same number of steps; time is reported either way
        beats = mg["div_post"] < jac["div_post"] * 1.05 or (mg["div_post"] / max(mg["seconds"], 1e-6)) < (
            jac["div_post"] / max(jac["seconds"], 1e-6))
    replay_diff = report["replay"].get("diff")
    ok = (
        finite and max_speed < 15 and samples[-1]["step"] >= 600
        and sphere_sdf < 0 and mesh_sdf < 0
        and report["obstacles"]["pop_triangles"] >= 8
        and report["obstacles"]["sop_triangles"] >= 8
        and not mismatched and report["resume_identical"]["same_names"]
        and replay_diff == 0.0
        and report["phase1_bake_check"] == 0
        and div_factor is not None and div_factor > 1.05
        and report["uncapped"]["fps"] >= 30
    )
    report["pass"] = ok
    (ROOT / "reports" / "phase2.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
