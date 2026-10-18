"""Phase 5 checks. Exit 0 is pass. Writes reports/phase5.json and reports/phase5_*.png.

Touches only the scratch TouchDesigner from td_scratch.py. Long cooks are split across calls.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from td_scratch import ROOT, attach  # noqa: E402

import fracture as fracture_mod  # noqa: E402
import rigidbody as rigid_mod  # noqa: E402

REPORTS = ROOT / "reports"
CACHE_BUDGET = 30 * (1024 ** 3)
# Worst post-step RMS of div(u) - expansion in reports/phase4.json, 128^3 Jacobi, the realtime solver.
PHASE4_REALTIME_DIV_RMS = 0.05685112252831459


def cache_bytes() -> int:
    root = ROOT / "cache"
    if not root.is_dir():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def require_budget():
    used = cache_bytes()
    if used > CACHE_BUDGET:
        raise SystemExit(f"cache/ is {used} bytes, over the 30 GB budget")
    return used


CLEAN = r'''
import json
d = op('/sims/destruction')
if d is not None:
    d.par.reinitextensions.pulse()
    d.Release()
op('/sims/baker').par.reinitextensions.pulse()
op('/sims/baker').Cancel()
op('/sims/clock/tick').par.active = False
p = op('/sims/particles')
p.Unbind()
p.par.Resx = 1024
p.par.Resy = 1024
p.par.Bus = '2d'
p.par.Emitter = 'sphere'
p.par.Velmode = 'bus'
p.par.Couplingstrength = 1
p.par.Cookrender = False
op('/sims/fluid2d').par.Cookrender = False
f = op('/sims/fluid3d')
f.allowCooking = True
f.par.Cookrender = False
f.par.Detail = 0
f.par.Solver = 'jacobi'
f.par.Jacobi = 16
f.par.Raysteps = 48
f.par.Renderw = 1280
f.par.Renderh = 720
f.ApplyPreset('realtime')
for name in ('render_out', 'shadow'):
    node = f.op(name)
    if node is not None:
        node.bypass = True
view = f.op('view_scalars')
if view is not None:
    view.par.top = f.op('scalars_out')
vel = f.op('view_vel')
if vel is not None:
    vel.par.top = f.op('vel_out')
c = op.simclock
c.par.Sims = '/sims/fluid2d'
c.par.Substeps = 1
c.par.Fps = 60
c.par.Mode = 'realtime'
c.par.Seed = 0
c.par.Running = True
c.par.Startframe = 0
b = op('/sims/baker')
b.par.Sim = '/sims/fluid2d'
b.par.Groupsims = ''
b.par.Cachedir = 'cache/fluid2d'
b.par.Keyinterval = 0
b.par.Playbackfields = ''
b.par.Crop = False
b.par.Asyncsave = False
r = op('/sims/cache_reader')
r.par.Cachedir = 'cache/fluid2d'
r.par.Frame.expr = 'op.simclock.par.Frame'
r.par.Interpolate = True
r.par.Member = ''
bus = op.simbus3d
for name in ('sphere', 'box', 'capsule', 'mesh'):
    bus.Publish('obstacles', '/sims/demo3d/' + name)
bus.Publish('sources', '/sims/demo3d/emitter_sources')
bus.Publish('forces', '/sims/demo3d/emitter_forces')
op.simbus.Publish('obstacles', '/sims/demo/disc')
project.realTime = True
op('/sims/clock/tick').par.active = True
print('@@JSON@@' + json.dumps({'sims': c.par.Sims.eval()}))
'''

MARK = r'''
import json
print('@@JSON@@' + json.dumps({'step': int(op.simclock.par.Step.eval()), 'frame': int(absTime.frame)}))
'''


def timeline_rate(td, seconds=3.0):
    start = td.run_json(MARK, timeout=20)
    time.sleep(seconds)
    end = td.run_json(MARK, timeout=30)
    steps = end["step"] - start["step"]
    frames = end["frame"] - start["frame"]
    return {"steps": steps, "frames": frames, "seconds": seconds,
            "fps": (frames / seconds) if seconds else 0.0,
            "ms": (seconds * 1000.0 / steps) if steps else None}


def fracture_metrics() -> dict:
    solid = fracture_mod.box_solid((0.1, 0.5, -0.05), (0.45, 0.5, 0.3))
    pieces = fracture_mod.fracture(solid, 16, seed=11, pattern="uniform")
    again = fracture_mod.fracture(solid, 16, seed=11, pattern="uniform")
    original = solid.volume()
    vol_err = abs(sum(p.volume for p in pieces) - original) / original
    open_edges = 0
    for piece in pieces:
        counts = fracture_mod.edge_counts(piece)
        open_edges += sum(1 for n in counts.values() if n != 2)
    rng = np.random.default_rng(2)
    lo, hi = solid.bounds()
    samples = []
    while len(samples) < 60:
        p = lo + rng.random(3) * (hi - lo)
        if float(solid.sdf(p)[0]) < -1e-3:
            samples.append(p)
    owners = np.zeros(len(samples), dtype=np.int32)
    for piece in pieces:
        owners += piece.sdf(np.asarray(samples)) < -1e-5
    sphere = fracture_mod.sphere_solid((0, 0.4, 0), 0.3, subdiv=1)
    sp = fracture_mod.fracture(sphere, 6, seed=2)
    hull = fracture_mod.convex_hull(solid.vertices())
    cut = fracture_mod.fracture(hull, 5, seed=8)
    big = fracture_mod.fracture(fracture_mod.box_solid((0, 0.5, 0), (0.3, 0.5, 0.25)), 1000, seed=3)
    return {
        "pieces": len(pieces),
        "volume_error": vol_err,
        "open_edges": open_edges,
        "overlap_samples": int(np.sum(owners != 1)),
        "deterministic": bool(np.allclose([p.volume for p in pieces], [p.volume for p in again])),
        "sphere_volume_error": abs(sum(p.volume for p in sp) - sphere.volume()) / sphere.volume(),
        "hull_volume_error": abs(sum(p.volume for p in cut) - hull.volume()) / max(hull.volume(), 1e-9),
        "thousand_pieces": len(big),
        "thousand_volume_error": abs(sum(p.volume for p in big) - 0.3) / 0.3,
    }


def physics_metrics() -> dict:
    drop = rigid_mod.World(restitution=0.0, friction=0.8, iterations=24, position_iters=6)
    drop.add_box((0.0, 1.0, 0.0), (0.1, 0.1, 0.1), mass=2.0)
    e0 = drop.energy()
    peak = e0
    for _ in range(180):
        drop.step(1.0 / 120.0)
        peak = max(peak, drop.energy())
    stack = rigid_mod.World(restitution=0.0, friction=0.9, iterations=24, position_iters=6, sleep_after=30)
    half = 0.1
    for i in range(3):
        stack.add_box((0.0, half + i * 2 * half, 0.0), (half, half, half), mass=1.0)
    for _ in range(240):
        stack.step(1.0 / 120.0)
    return {
        "rest_y": float(drop.bodies[0].pos[1]),
        "rest_speed": float(np.linalg.norm(drop.bodies[0].lin)),
        "energy_ratio": peak / e0,
        "drop_miny": drop.min_vertex_y(),
        "stack_y": [float(b.pos[1]) for b in stack.bodies],
        "stack_xz": [float(max(abs(b.pos[0]), abs(b.pos[2]))) for b in stack.bodies],
        "stack_pen": stack.max_penetration(),
        "stack_miny": stack.min_vertex_y(),
    }


def image_stats(path: Path) -> dict:
    try:
        from PIL import Image  # noqa: PLC0415
        image = np.asarray(Image.open(path).convert("RGBA")).astype(np.float64) / 255.0
    except Exception:
        return {"path": str(path.relative_to(ROOT)), "unreadable": True}
    rgb = image[..., :3]
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return {
        "path": str(path.relative_to(ROOT)),
        "mean_luma": float(luma.mean()),
        "std_luma": float(luma.std()),
        "max_luma": float(luma.max()),
        "min_alpha": float(image[..., 3].min()),
        "blown": float(np.mean(luma > 0.98)),
        "coverage": float(np.mean(image[..., 3] > 0.2)),
    }


def main() -> int:
    td = attach()
    report = {"cache_bytes_start": require_budget()}
    report["fracture"] = fracture_metrics()
    print("fracture", report["fracture"])
    report["physics"] = physics_metrics()
    print("physics", {k: report["physics"][k] for k in ("rest_y", "energy_ratio", "stack_y", "stack_pen")})

    report["bullet"] = td.run_json(r'''
import json
names = [n for n in dir(td) if 'bullet' in n.lower()]
parent = op('/sims')
old = parent.op('bullet_probe')
if old is not None:
    old.destroy()
info = {'types': names}
try:
    solver = parent.create(bulletsolverCOMP, 'bullet_probe')
    # Default actors. Cooking the solver does not integrate; y stays put.
    actor = solver.op('actor1') or solver.findChildren(type=actorCOMP)[0]
    y0 = float(actor.par.ty.eval())
    solver.par.gravityy = -9.81
    solver.par.play = True
    for _ in range(5):
        solver.cook(force=True)
    y1 = float(actor.par.ty.eval())
    info.update({'created': True, 'y0': y0, 'y1': y1, 'moved': abs(y1 - y0) > 1e-4,
                 'note': 'cook() does not advance Bullet. Driving the timeline from a script wedged the process, so Bullet is not the bake solver.'})
    solver.destroy()
except Exception as exc:
    info['error'] = repr(exc)
print('@@JSON@@' + json.dumps(info))
''', timeout=40)
    print("bullet", report["bullet"])

    td.run_json(CLEAN, timeout=40)
    # Coupling: a kinematic box, smoke must stay out and pick up its velocity.
    report["coupling"] = td.run_json(r'''
import json
import numpy as np
d = op('/sims/destruction')
f = op('/sims/fluid3d')
bus = op.simbus3d
f.par.Cookrender = False
f.par.Solver = 'jacobi'
f.par.Jacobi = 8
f.par.Buoyancy = 0
f.par.Vorticity = 0
f.par.Maccormack = 0
bus.par.Resw = 64
bus.par.Resh = 64
bus.par.Resd = 64
# Unpublish the demo obstacles so the mover is the only solid.
for path in list(bus.Publishers('obstacles')):
    bus.Unpublish(path)
d.ArmMover()
c = op.simclock
c.par.Sims = '/sims/destruction /sims/fluid3d'
c.par.Running = True
op('/sims/clock/tick').par.active = False
c.RequestReset()
c.Tick()
for _ in range(8):
    c.Tick()
d.op('obstacles3d').bypass = False
d.op('obstacles3d').cook(force=True)
bus.op('obstacles').cook(force=True)
body = d.ext.Destruction.world.bodies[0]
vel = f.op('vel_out').numpyArray(delayed=False)
smoke = f.op('scalars_out').numpyArray(delayed=False)
obs = bus.op('obstacles').numpyArray(delayed=False)
b = bus.Bounds
res = bus.Res

def ijk(world):
    uv = [(world[i] - b[i]) / (b[i + 3] - b[i]) for i in range(3)]
    return tuple(int(np.clip(round(uv[i] * res[i] - 0.5), 0, res[i] - 1)) for i in range(3))

center = ijk(tuple(float(v) for v in body.pos))
inside = obs[center[2], center[1], center[0]]
# A cell just outside, along +x.
outside_p = (float(body.pos[0]) + 0.22, float(body.pos[1]), float(body.pos[2]))
out = ijk(outside_p)
near = vel[out[2], out[1], out[0], :3]
print('@@JSON@@' + json.dumps({
    'sdf': float(inside[0]), 'smoke': float(smoke[center[2], center[1], center[0], 0]),
    'obs_vel': [float(v) for v in inside[1:4]],
    'near_vel': [float(v) for v in near],
    'body_vel': [float(v) for v in body.lin],
    'finite': bool(np.isfinite(vel).all()),
}))
''', timeout=120)
    print("coupling", report["coupling"])

    # Wall, dust, debris, rate. Arm() is the scene a user gets: renders stay on, dust stays
    # on, and the clock steps the particles. Nothing is bypassed for the measurement.
    report["break"] = td.run_json(r'''
import json
d = op('/sims/destruction')
f = op('/sims/fluid3d')
d.Release()
d.par.Pieces = 100
d.par.Seed = 4
d.par.Pattern = 'clustered'
d.par.Shape = 'box'
d.par.Projectile = True
d.par.Breakmode = 'impact'
d.par.Grid = 16
info = d.Arm()
bus = op.simbus3d
for name in ('sphere', 'box', 'capsule', 'mesh'):
    bus.Unpublish('/sims/demo3d/' + name)
c = op.simclock
op('/sims/clock/tick').par.active = False
c.RequestReset()
c.Tick()
for _ in range(8):
    c.Tick()
import numpy as np
w = d.ext.Destruction.world
life = op('/sims/particles/pvel_out').numpyArray(delayed=False)
dust = d.op('dust3d')
dust.cook(force=True)
field = dust.numpyArray(delayed=False)
info['broken'] = w.glue_broken()
info['glues'] = len(w.glues)
info['miny'] = w.min_vertex_y()
info['alive'] = int(np.sum(life[..., 3] > 0))
info['dust_max'] = float(np.nanmax(field[..., 0]))
info['bodies'] = len(w.bodies)
d.op('pieces_render').cook(force=True)
d.op('pieces_render').save(project.folder + '/reports/phase5_pieces.png')
f.op('render_out').cook(force=True)
d.op('composite').cook(force=True)
# Saving this compute TOP writes a checker. The array is the picture.
np.save(project.folder + '/reports/phase5_smoke.npy',
        np.ascontiguousarray(np.array(d.op('composite').numpyArray(delayed=False), np.float32)))
pic = np.array(d.op('pieces_render').numpyArray(delayed=False), np.float32)
smoke = np.array(d.op('composite').numpyArray(delayed=False), np.float32)
def _luma(img):
    rgb = np.clip(img[..., :3], 0, 1)
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return {'mean_luma': float(luma.mean()), 'std_luma': float(luma.std()),
            'blown': float(np.mean(luma > 0.98)), 'coverage': float(np.mean(img[..., 3] > 0.05)),
            'max': float(img[..., :3].max()), 'min_alpha': float(img[..., 3].min())}
watched = {
    'render_out': f.op('render_out'), 'shadow': f.op('shadow'),
    'pieces_render': d.op('pieces_render'), 'composite': d.op('composite'), 'dust3d': d.op('dust3d'),
}
info['scene'] = {
    'res': list(bus.Res),
    'solver': str(f.par.Solver.eval()),
    'vcycles': int(f.par.Vcycles.eval()),
    'smooth': int(f.par.Smooth.eval()),
    'coarse': int(f.par.Coarse.eval()),
    'raysteps': int(f.par.Raysteps.eval()),
    'shadowsteps': int(f.par.Shadowsteps.eval()),
    'render': [int(f.par.Renderw.eval()), int(f.par.Renderh.eval())],
    'piece_render': [int(d.par.Renderw.eval()), int(d.par.Renderh.eval())],
    'cookrender_fluid': bool(f.par.Cookrender.eval()),
    'cookrender_pieces': bool(d.par.Cookrender.eval()),
    'detail': float(f.par.Detail.eval()),
    'mesh_contacts': bool(d.par.Meshcontacts.eval()),
    'pieces': int(d.par.Pieces.eval()),
    'iterations': int(d.par.Iterations.eval()),
    'sims': str(c.par.Sims.eval()),
    'bypass': {name: bool(node.bypass) for name, node in watched.items() if node is not None},
}
info['pictures'] = {'pieces': _luma(pic), 'smoke': _luma(smoke)}
print('@@JSON@@' + json.dumps(info))
''', timeout=120)
    print("break", {k: report["break"].get(k) for k in ("pieces", "broken", "alive", "dust_max", "scene")})
    td.run_json(r'''
import json
op('/sims/clock/tick').par.active = True
op.simclock.par.Running = True
project.realTime = False
print('@@JSON@@' + json.dumps({'step': int(op.simclock.par.Step.eval()), 'sims': str(op.simclock.par.Sims.eval())}))
''', timeout=20)
    report["realtime"] = timeline_rate(td, 3.0)
    td.run_json("op('/sims/clock/tick').par.active = False\nprint('@@JSON@@' + __import__('json').dumps(1))", timeout=20)
    report["realtime"]["scene"] = (report.get("break") or {}).get("scene")
    report["divergence"] = td.run_json(r'''
import json
import numpy as np
f = op('/sims/fluid3d')
post = f.op('div_check')
post.cook(force=True)
values = np.array(post.numpyArray(delayed=False)[..., 0], dtype=np.float64)
print('@@JSON@@' + json.dumps({
    'rms': float(np.sqrt(np.mean(values ** 2))),
    'max_abs': float(np.max(np.abs(values))),
    'finite': bool(np.isfinite(values).all()),
}))
''', timeout=60)
    report["realtime"]["divergence"] = report["divergence"]
    print("realtime", report["realtime"])
    report["pictures"] = (report.get("break") or {}).get("pictures") or {}
    smoke_npy = REPORTS / "phase5_smoke.npy"
    if smoke_npy.is_file():
        from PIL import Image
        smoke = np.clip(np.load(smoke_npy)[..., :3], 0.0, 1.0)
        Image.fromarray(np.ascontiguousarray((smoke * 255.0).astype(np.uint8)), mode="RGB").save(REPORTS / "phase5_smoke.png")

    # Exactness bake at 96^3, six frames, every field. This is not the showcase.
    td.run_json(CLEAN, timeout=40)
    report["bake"] = {}
    for folder in ("phase5_full", "phase5_resume"):
        dest = ROOT / "cache" / folder
        if dest.exists():
            shutil.rmtree(dest)
    setup = td.run_json(r'''
import json
d = op('/sims/destruction')
f = op('/sims/fluid3d')
d.par.Pieces = 40
d.par.Projectile = True
d.par.Iterations = 3
d.par.Positers = 1
d.Arm()
# Arm() installs the 64^3 realtime scene. The exactness bake is a separate 96^3 Jacobi run.
bus = op.simbus3d
bus.par.Resw = 96
bus.par.Resh = 96
bus.par.Resd = 96
f.par.Solver = 'jacobi'
f.par.Jacobi = 8
f.par.Cookrender = False
f.par.Maccormack = 0
d.par.Cookrender = False
p = op('/sims/particles')
p.par.Resx = 256
p.par.Resy = 256
b = op('/sims/baker')
b.par.Sim = '/sims/fluid3d'
b.par.Groupsims = '/sims/destruction /sims/fluid3d /sims/particles'
b.par.Cachedir = 'cache/phase5_full'
b.par.Framestart = 1
b.par.Frameend = 6
b.par.Warmup = 0
b.par.Resume = False
b.par.Keyinterval = 0
b.par.Playbackfields = ''
b.par.Crop = False
b.par.Asyncsave = False
op.simclock.par.Substeps = 1
op('/sims/clock/tick').par.active = False
b.Bake()
# Pump in this same call. A gap between scripts lets a dirty 3D solver recook and the two bakes diverge.
for _ in range(8):
    if not b.ext.Baker.baking:
        break
    b.PumpOnce()
print('@@JSON@@' + json.dumps({'status': b.par.Status.eval(), 'baking': b.ext.Baker.baking,
    'frame': int(op.simclock.par.Frame.eval())}))
''', timeout=180)
    print("bake setup", setup)
    report["bake"]["full"] = setup

    # Second run, cancel after 3, resume. One call, so nothing cooks between the saved frames.
    state = td.run_json(r'''
import json
b = op('/sims/baker')
b.par.Cachedir = 'cache/phase5_resume'
b.par.Resume = False
b.par.Frameend = 6
op('/sims/clock/tick').par.active = False
b.Bake()
for _ in range(4):
    if not b.ext.Baker.baking:
        break
    b.PumpOnce()
    if int(op.simclock.par.Frame.eval()) >= 3:
        break
b.Cancel()
b.par.Resume = True
b.Bake()
for _ in range(6):
    if not b.ext.Baker.baking:
        break
    b.PumpOnce()
print('@@JSON@@' + json.dumps({'status': b.par.Status.eval(), 'baking': b.ext.Baker.baking,
    'frame': int(op.simclock.par.Frame.eval())}))
''', timeout=180)
    print("resume", state)
    report["bake"]["resumed"] = state
    full = ROOT / "cache" / "phase5_full"
    resumed = ROOT / "cache" / "phase5_resume"
    mismatched = []
    files = []
    if full.exists():
        files = sorted(p.relative_to(full) for p in full.rglob("*") if p.suffix in (".exr", ".npy") and p.is_file())
    for rel in files:
        other = resumed / rel
        if not other.is_file() or (full / rel).read_bytes() != other.read_bytes():
            mismatched.append(str(rel))
    report["bake"]["files"] = len(files)
    report["bake"]["mismatched"] = mismatched
    report["bake"]["identical"] = len(files) > 0 and not mismatched
    print("resume identical", report["bake"]["identical"], "files", len(files), "bad", len(mismatched))
    for name in mismatched[:12]:
        print(" ", name)

    if full.exists():
        shutil.rmtree(full)
    if resumed.exists():
        shutil.rmtree(resumed)

    quick = "--quick" in sys.argv
    report["quick"] = quick
    # --quick skips the 180-frame showcase bake and leaves that cache in place.
    if not quick:
        # Showcase: 1000 pieces, 180 frames, 128^3, reduced keyframes. Renders are off while
        # baking; one mid-break frame is drawn from the cache afterwards.
        show = ROOT / "cache" / "phase5_showcase"
        if show.exists():
            shutil.rmtree(show)
        report["showcase_bake"] = td.run_json(r'''
import json
d = op('/sims/destruction')
f = op('/sims/fluid3d')
d.Release()
d.par.Pieces = 1000
d.par.Seed = 3
d.par.Pattern = 'clustered'
d.par.Shape = 'box'
d.par.Projectile = True
d.par.Breakmode = 'impact'
info = d.Arm()
bus = op.simbus3d
bus.par.Resw = 128
bus.par.Resh = 128
bus.par.Resd = 128
f.par.Solver = 'multigrid'
f.par.Vcycles = 1
f.par.Smooth = 4
f.par.Coarse = 16
f.par.Cookrender = False
d.par.Cookrender = False
p = op('/sims/particles')
p.par.Resx = 256
p.par.Resy = 256
p.par.Cookrender = False
b = op('/sims/baker')
b.par.Sim = '/sims/fluid3d'
b.par.Groupsims = '/sims/destruction /sims/fluid3d /sims/particles'
b.par.Cachedir = 'cache/phase5_showcase'
b.par.Framestart = 1
b.par.Frameend = 180
b.par.Warmup = 0
b.par.Resume = False
b.par.Keyinterval = 30
b.par.Playbackfields = 'scalars'
b.par.Crop = True
b.par.Croppad = 4
b.par.Asyncsave = False
op.simclock.par.Substeps = 1
op('/sims/clock/tick').par.active = False
b.Bake()
print('@@JSON@@' + json.dumps({'pieces': info.get('pieces'), 'baking': b.ext.Baker.baking,
    'status': str(b.par.Status.eval())}))
''', timeout=180)
        print("showcase arm", report["showcase_bake"])
        showcase_state = report["showcase_bake"]
        for _ in range(20):
            showcase_state = td.run_json(r'''
import json
b = op('/sims/baker')
for _ in range(12):
    if not b.ext.Baker.baking:
        break
    b.PumpOnce()
print('@@JSON@@' + json.dumps({'baking': b.ext.Baker.baking, 'status': str(b.par.Status.eval()),
    'frame': int(op.simclock.par.Frame.eval())}))
''', timeout=180)
            print(" showcase", showcase_state)
            if not showcase_state.get("baking"):
                break
        report["showcase_bake"]["pumped"] = showcase_state
        report["showcase_picture"] = td.run_json(r'''
import json
import numpy as np
d = op('/sims/destruction')
f = op('/sims/fluid3d')
r = op('/sims/cache_reader')
r.par.Cachedir = 'cache/phase5_showcase'
r.par.Member = 'fluid3d'
r.par.Interpolate = False
r.par.Frame.expr = ''
r.Reload()
folder = project.folder + '/reports'
paths = []
for frame in (31, 91, 151):
    d.SetInit({'bodies': 'cache/phase5_showcase/destruction/bodies/bodies.%05d.npy' % frame})
    r.Load(frame)
    f.op('view_scalars').par.top = r.Output('scalars')
    f.par.Cookrender = True
    f.op('render_out').bypass = False
    f.op('shadow').bypass = False
    d.par.Cookrender = True
    for name in ('pieces_render', 'composite', 'dust3d'):
        d.op(name).bypass = False
    f.op('render_out').cook(force=True)
    d.op('composite').cook(force=True)
    # TOP.save of this compute composite writes the error checker. The pixel
    # array is the picture; the validator turns it into a PNG.
    image = np.array(d.op('composite').numpyArray(delayed=False), np.float32)
    npy = folder + '/phase5_showcase_%05d.npy' % frame
    np.save(npy, image)
    paths.append(npy)
f.op('view_scalars').par.top = f.op('scalars_out')
f.op('view_vel').par.top = f.op('vel_out')
print('@@JSON@@' + json.dumps({'frames': [31, 91, 151], 'paths': paths}))
''', timeout=180)
        picture = report["showcase_picture"]
        picture["paths"] = [str(Path(p).relative_to(ROOT)) for p in picture.get("paths", [])]
        print("showcase picture", picture)
        manifest_path = show / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        member = ((manifest.get("members") or {}).get("destruction") or {}).get("params") or {}
        fluid_member = ((manifest.get("members") or {}).get("fluid3d") or {}).get("params") or {}
        show_bytes = sum(p.stat().st_size for p in show.rglob("*") if p.is_file()) if show.exists() else 0
        sheet_path = REPORTS / "phase5_showcase_sheet.png"
        showcase_png = REPORTS / "phase5_showcase.png"
        def _png_from_npy(frame: int):
            array_path = REPORTS / f"phase5_showcase_{frame:05d}.npy"
            if not array_path.is_file():
                return None
            image = np.load(array_path)
            rgb = np.clip(image[..., :3], 0.0, 1.0)
            from PIL import Image
            png = REPORTS / f"phase5_showcase_{frame:05d}.png"
            Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB").save(png)
            return png

        mid = _png_from_npy(91)
        if mid is not None and mid.is_file():
            shutil.copyfile(mid, showcase_png)
            try:
                from PIL import Image
                tiles = []
                for frame in (31, 91, 151):
                    made = _png_from_npy(frame)
                    if made is None:
                        continue
                    tiles.append(Image.open(made).convert("RGB"))
                if not tiles:
                    raise RuntimeError("no showcase frames")
                width = sum(tile.width for tile in tiles)
                height = max(tile.height for tile in tiles)
                sheet = Image.new("RGB", (width, height), (0, 0, 0))
                x = 0
                for tile in tiles:
                    sheet.paste(tile, (x, 0))
                    x += tile.width
                sheet.save(sheet_path)
            except Exception as exc:
                report["showcase_sheet_error"] = repr(exc)
        report["showcase"] = {
            "frames_written": int(manifest.get("frames_written") or 0),
            "pieces": int(((member.get("settings") or {}).get("count") or 0)),
            "fluid_res": fluid_member.get("res"),
            "bytes": show_bytes,
            "png": "reports/phase5_showcase.png",
            "sheet": "reports/phase5_showcase_sheet.png" if sheet_path.is_file() else None,
        }
        report["showcase_bytes"] = show_bytes
        report["showcase_kept"] = show_bytes > 0 and show_bytes <= 3 * (1024 ** 3)
        print("showcase", report["showcase"])

    report["pieces_png"] = image_stats(REPORTS / "phase5_pieces.png")
    report["smoke_png"] = image_stats(REPORTS / "phase5_smoke.png")

    td.run_json(CLEAN, timeout=40)

    def script(*args: str) -> int:
        return subprocess.run([sys.executable, *args], cwd=ROOT).returncode

    # Each earlier phase is defined against a fresh session. Its frame-rate gates drop after the
    # showcase bake has run for minutes in the same TouchDesigner, so restart before each one.
    def fresh(*args: str) -> int:
        scratch = str(ROOT / "scripts" / "td_scratch.py")
        if script(scratch, "stop") or script(scratch, "start") or script(str(ROOT / "scripts" / "build_project.py")):
            return 1
        return script(*args)

    report["bake_check"] = fresh(str(ROOT / "scripts" / "bake_check.py"), "--frames", "40")
    print("bake_check", report["bake_check"])
    report["phase2"] = fresh(str(ROOT / "scripts" / "validate_phase2.py"))
    print("phase2", report["phase2"])
    report["phase3"] = fresh(str(ROOT / "scripts" / "validate_phase3.py"))
    print("phase3", report["phase3"])
    report["phase4"] = fresh(str(ROOT / "scripts" / "validate_phase4.py"), "--quick")
    print("phase4", report["phase4"])
    report["shaders"] = script(str(ROOT / "scripts" / "check_shaders.py"))
    report["pytest"] = script("-m", "pytest", "-q")
    td = attach()
    td.run_json(CLEAN, timeout=40)
    report["cache_bytes_end"] = cache_bytes()

    frac = report["fracture"]
    phys = report["physics"]
    coup = report["coupling"]
    brk = report["break"]
    rate = report["realtime"]
    scene = rate.get("scene") or {}
    bypass = scene.get("bypass") or {}
    div = rate.get("divergence") or {}
    pics = (report.get("pictures") or {}).get("pieces") or {}
    showcase = report.get("showcase") or {}
    sims = str(scene.get("sims") or "")
    ok = (
        frac["volume_error"] < 1e-4
        and frac["open_edges"] == 0
        and frac["overlap_samples"] == 0
        and frac["deterministic"]
        and frac["thousand_pieces"] == 1000
        and frac["thousand_volume_error"] < 1e-3
        and abs(phys["rest_y"] - 0.1) < 2e-3
        and phys["energy_ratio"] <= 1.002
        and phys["drop_miny"] >= -1e-3
        and all(abs(y - (0.1 + i * 0.2)) < 3e-3 for i, y in enumerate(phys["stack_y"]))
        and phys["stack_pen"] < 2e-3
        and phys["stack_miny"] >= -1e-3
        and coup.get("sdf", 1) < 0
        and coup.get("smoke", 1) == 0
        and abs(coup.get("obs_vel", [0])[0] - 0.6) < 0.05
        and brk.get("broken", 0) >= 1
        and 10 < brk.get("alive", 0) < 20000
        and brk.get("dust_max", 0) > 0
        and brk.get("miny", -1) >= -5e-3
        and rate["steps"] == rate["frames"]
        and rate["steps"] > 0
        and rate["fps"] >= 33
        and scene.get("solver") == "multigrid"
        and int(scene.get("vcycles") or 0) >= 1
        and scene.get("res") == [64, 64, 64]
        and scene.get("cookrender_fluid") is True
        and scene.get("cookrender_pieces") is True
        and "/sims/particles" in sims.split()
        and bypass
        and not any(bypass.values())
        and div.get("finite") is True
        and div.get("rms", 1.0) <= 2.0 * PHASE4_REALTIME_DIV_RMS
        and report["bake"].get("identical") is True
        and (report.get("quick") or (
            report["showcase_kept"] is True
            and int(showcase.get("frames_written") or 0) >= 180
            and int(showcase.get("pieces") or 0) >= 1000
        ))
        and pics.get("std_luma", 0) > 0.005
        and pics.get("blown", 1) < 0.05
        and pics.get("coverage", 0) > 0.005
        and pics.get("min_alpha", 0) >= 0.99
        and (report.get("pictures") or {}).get("smoke", {}).get("std_luma", 0) > 0.002
        and report["bake_check"] == 0
        and report["phase2"] == 0
        and report["phase3"] == 0
        and report["phase4"] == 0
        and report["shaders"] == 0
        and report["pytest"] == 0
        and report["cache_bytes_end"] <= CACHE_BUDGET
    )
    report["pass"] = bool(ok)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "phase5.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
