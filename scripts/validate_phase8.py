"""Phase 8 checks. Exit 0 is pass. Writes reports/phase8.json and reports/phase8_*.png.

Touches only the scratch TouchDesigner from td_scratch.py. Nested phase checks each get a fresh
scratch session. This process does not change project.cookRate.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from td_scratch import ROOT, attach  # noqa: E402

REPORTS = ROOT / "reports"
CACHE_BUDGET = 30 * (1024 ** 3)
SHOWCASE = "cache/phase8_showcase"


def remove_tree(path: Path) -> None:
    """Delete a cache directory. Finder may drop a .DS_Store mid-delete on macOS."""
    path = Path(path)
    if not path.exists():
        return
    for _ in range(4):
        for ds in path.rglob(".DS_Store"):
            ds.unlink(missing_ok=True)
        try:
            shutil.rmtree(path)
        except OSError:
            continue
        if not path.exists():
            return
    shutil.rmtree(path, ignore_errors=True)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def save_png(array: np.ndarray, path: Path) -> dict:
    rgb = np.clip(np.asarray(array)[..., :3], 0, 1)
    if rgb.ndim != 3:
        raise RuntimeError(f"bad image {path}")
    view = rgb[::-1]
    Image.fromarray((view * 255.0).astype(np.uint8), mode="RGB").save(path)
    luma = view.mean(axis=2)
    return {
        "path": rel(path),
        "mean": float(luma.mean()),
        "std": float(luma.std()),
        "max": float(view.max()),
        "checker": checker_score(view),
    }


def checker_score(rgb: np.ndarray) -> float:
    """Fraction of dim grey pixels whose diagonal neighbour matches and whose
    side neighbour does not. The border-repeat bug draws that pattern on one
    face of the volume. A flat backdrop and a shaded object score near 0.
    """
    image = np.asarray(rgb)[..., :3]
    luma = image.mean(axis=2)
    sat = image.max(axis=2) - image.min(axis=2)
    grey = (sat < 0.04) & (luma > 0.05) & (luma < 0.45)
    best = 0.0
    for lag in (5, 6, 7, 8, 10):
        here = luma[:-lag, :-lag]
        diag = luma[lag:, lag:]
        side = luma[:-lag, lag:]
        mask = grey[:-lag, :-lag] & grey[lag:, lag:] & grey[:-lag, lag:]
        count = int(mask.sum())
        if count < 30:
            continue
        hit = (np.abs(here - diag) < 0.03) & (np.abs(here - side) > 0.06) & mask
        best = max(best, float(hit.sum()) / count)
    return best


def cache_bytes() -> int:
    root = ROOT / "cache"
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def write_report(report: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "phase8.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


def run(td, code: str, timeout: int = 60):
    return td.run_json(code, timeout=timeout)


def enable_sims(td) -> None:
    """The particle fps window turns the 3D comps off. Later checks need them cooking."""
    run(td, r'''
import json
for path in ('/sims/fluid2d', '/sims/fluid3d', '/sims/particles', '/sims/liquid', '/sims/destruction', '/sims/cloth', '/sims/scenes'):
    node = op(path)
    if node is not None:
        node.allowCooking = True
print('@@JSON@@' + json.dumps(1))
''', timeout=20)


def main() -> int:
    td = attach()
    report: dict = {"nested": {}}
    real = run(td, "import json\nprint('@@JSON@@'+json.dumps(bool(project.realTime)))")
    try:
        # Showcase first. Later checks leave particle state behind, and a bake started
        # from that state does not redraw to the saved textures.
        report["showcase"] = combined_showcase(td)
        print("showcase", {k: report["showcase"].get(k) for k in (
            "frames", "bad", "bytes", "replay", "replay_frames", "replay_neighbor")})
        write_report(report)
        report["export"] = export_showcase(td, report.get("showcase") or {})
        print("export", report["export"])
        write_report(report)
        report["splat"] = splat_checks(td)
        print("splat", report["splat"].get("fps"), report["splat"].get("recook"))
        write_report(report)
        report["group"] = group_resume(td)
        print("group", report["group"].get("bad"))
        write_report(report)
        report["liquid"] = liquid_surface(td)
        print("liquid", report["liquid"])
        write_report(report)
        report["upres"] = volume_detail(td)
        print("upres", report["upres"])
        write_report(report)
        report["scene"] = combined_realtime(td)
        print("scene", report["scene"].get("fps"), report["scene"].get("banner_z"), report["scene"].get("debris"),
              report["scene"].get("match_previous"), report["scene"].get("numbers"))
        write_report(report)
        report["nested"] = nested_validators()
        print("nested", report["nested"])
        write_report(report)
        report["repro"] = showcase_repro()
        print("repro", report["repro"])
        witness = REPORTS / "phase5_smoke.png"
        if witness.exists():
            report["checker_witness"] = checker_score(np.asarray(Image.open(witness)).astype(np.float32) / 255.0)
        shaders = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_shaders.py")], cwd=ROOT)
        tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
        report["shaders"] = shaders.returncode
        report["pytest"] = tests.returncode
        report["cache_bytes"] = cache_bytes()
        report["pass"] = evaluate(report)
    finally:
        try:
            run(attach(), f"project.realTime = {bool(real)}\nprint('@@JSON@@'+__import__('json').dumps(1))", timeout=20)
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            print("could not restore project.realTime:", exc)
        write_report(report)
    print("PASS" if report.get("pass") else "FAIL")
    return 0 if report.get("pass") else 1


def splat_checks(td) -> dict:
    stats = run(td, r'''
import json, numpy as np
p = op('/sims/particles')
f = op('/sims/fluid2d')
c = op.simclock
op('/sims/clock/tick').par.active = False
p.Unbind(); p.SetInit(None)
p.par.Bus = '2d'
p.par.Velmode = 'bus'
p.par.Emitter = 'point'
p.par.Resx = 256
p.par.Resy = 256
p.par.Couplingstrength = 1
p.par.Srcheat = 0.02
p.par.Drag = 3.5
p.par.Cookrender = False
p.par.Emitx = 0
p.par.Emity = 0.4
p.par.Emitradius = 0.15
p.Bind()
f.par.Cookrender = False
f.ApplyPreset('realtime')
c.par.Sims = '/sims/fluid2d /sims/particles'
c.par.Seed = 0
c.par.Running = True
c.RequestReset()
c.Tick()
summed = p.op('splat_sum')
a = np.array(summed.numpyArray(delayed=False), copy=True)
summed.cook(force=True)
b = np.array(summed.numpyArray(delayed=False), copy=True)
dep = np.array(p.op('deposit').numpyArray(delayed=False), dtype=np.float64)
pos = np.array(p.op('pos_out').numpyArray(delayed=False), dtype=np.float64)
scale = 1000000.0
bounds = np.array([-1.0, 0.0, 1.0, 2.0])
gw, gh = int(summed.width), int(summed.height)
q = np.rint(dep * scale).astype(np.int64)
live = np.any(q != 0, axis=2)
xy = pos[..., :2]
uv = np.clip((xy - bounds[:2]) / (bounds[2:] - bounds[:2]), 0, 1)
pix = np.minimum(np.floor(uv * np.array([float(gw), float(gh)])).astype(np.int64), np.array([gw - 1, gh - 1]))
acc = np.zeros((gh, gw, 4), np.int64)
np.add.at(acc, (pix[..., 1][live], pix[..., 0][live]), q[live])
ref = acc / scale
gpu = b.astype(np.float64)
print('@@JSON@@' + json.dumps({
    'res': [int(summed.width), int(summed.height)],
    'recook': float(np.max(np.abs(a.astype(np.float64) - gpu))),
    'cpu': float(np.max(np.abs(gpu - ref))),
    'cpu_sum': float(ref.sum()),
    'gpu_sum': float(gpu.sum()),
    'err': summed.errors() or '',
}))
''', timeout=120)
    prep = dict(bus="2d", coupling=1, render="True", rx=1024, ry=1024,
                sims="/sims/particles /sims/fluid2d", fluid2d="True", fluid3d="False")
    run(td, FPS_PREP.format(**prep), timeout=60)
    fps = measure_fps(td, 3.0)
    stats["fps"] = fps
    return stats


FPS_PREP = r'''
import json
p = op('/sims/particles')
c = op.simclock
op('/sims/clock/tick').par.active = False
p.Unbind(); p.SetInit(None)
p.par.Bus = {bus!r}
p.par.Velmode = 'bus'
p.par.Emitter = 'sphere'
p.par.Drag = 2
p.par.Gravity = 0.2
p.par.Turbulence = 0.1
p.par.Couplingstrength = {coupling}
p.par.Srcheat = 0
p.par.Cookrender = {render}
p.par.Resx = {rx}
p.par.Resy = {ry}
p.par.Radius = -1
if {coupling} != 0:
    p.Bind()
c.par.Fps = 60
c.par.Substeps = 1
c.par.Mode = 'realtime'
c.par.Running = True
c.par.Sims = {sims!r}
project.realTime = False
op('/sims/fluid3d').allowCooking = bool({fluid3d})
op('/sims/fluid2d').allowCooking = bool({fluid2d})
if {fluid2d}:
    op('/sims/fluid2d').par.Cookrender = True
    op('/sims/fluid2d').ApplyPreset('realtime')
c.RequestReset()
c.Tick()
print('@@JSON@@' + json.dumps({{'step': int(c.par.Step)}}))
'''


def measure_fps(td, seconds: float) -> dict:
    start = run(td, r'''
import json
op('/sims/clock/tick').par.active = True
op.simclock.par.Running = True
project.realTime = False
print('@@JSON@@' + json.dumps({'step': int(op.simclock.par.Step), 'frame': int(absTime.frame)}))
''', timeout=30)
    time.sleep(seconds)
    end = run(td, r'''
import json
print('@@JSON@@' + json.dumps({'step': int(op.simclock.par.Step), 'frame': int(absTime.frame)}))
''', timeout=30)
    run(td, "op('/sims/clock/tick').par.active = False\nprint('@@JSON@@'+__import__('json').dumps(1))", timeout=20)
    steps = end["step"] - start["step"]
    frames = end["frame"] - start["frame"]
    return {"steps": steps, "frames": frames, "fps": round(steps / seconds, 2),
            "steps_match_frames": abs(steps - frames) <= 1}


def group_resume(td) -> dict:
    """Short coupled bake. The full 40-frame gate is validate_phase3 in the nested session."""
    full, resume = "cache/phase8_group", "cache/phase8_group_resume"
    bake_group(td, full, frames=12, cancel=None)
    bake_group(td, resume, frames=12, cancel=5)
    a_root, b_root = ROOT / full, ROOT / resume
    rels = sorted(p.relative_to(a_root) for p in a_root.rglob("*") if p.is_file())
    bad = [str(r) for r in rels if not (b_root / r).exists() or (a_root / r).read_bytes() != (b_root / r).read_bytes()]
    for path in (a_root, b_root):
        remove_tree(path)
    return {"files": len(rels), "bad": len(bad), "mismatched": bad[:8]}


def bake_group(td, cache: str, frames: int, cancel: int | None) -> None:
    run(td, GROUP_SETUP.format(cache=cache, frames=frames), timeout=40)
    run(td, "op('/sims/baker').Bake()\nprint('@@JSON@@'+__import__('json').dumps(1))", timeout=30)
    while True:
        state = run(td, r'''
import json
b = op('/sims/baker')
baking = b.PumpOnce() if b.ext.Baker.baking else False
print('@@JSON@@' + json.dumps({'baking': bool(baking), 'frame': int(op.simclock.par.Frame.eval())}))
''', timeout=120)
        if cancel is not None and state["baking"] and state["frame"] >= cancel:
            run(td, "op('/sims/baker').Cancel()\nprint('@@JSON@@'+__import__('json').dumps(1))", timeout=30)
            run(td, "op('/sims/baker').Bake()\nprint('@@JSON@@'+__import__('json').dumps(1))", timeout=30)
            cancel = None
            continue
        if not state["baking"]:
            return


GROUP_SETUP = r'''
import json
p = op('/sims/particles'); b = op('/sims/baker'); c = op.simclock; f = op('/sims/fluid2d')
op('/sims/clock/tick').par.active = False
p.Unbind(); p.SetInit(None)
p.par.Bus = '2d'; p.par.Velmode = 'bus'; p.par.Emitter = 'fire'
p.par.Resx = 256; p.par.Resy = 256
p.par.Drag = 3.5; p.par.Gravity = 0.2; p.par.Buoyancy = 0.4; p.par.Turbulence = 0.2
p.par.Radius = 0; p.par.Couplingstrength = 1; p.par.Particlemass = 1; p.par.Density = 1
p.par.Srcheat = 0.02; p.par.Cookrender = False
p.par.Lifemin = 0.6; p.par.Lifemax = 1.2; p.par.Threshold = 0.05
p.par.Emitx = 0; p.par.Emity = 0.22; p.par.Emitradius = 0.14; p.par.Emitvar = 0.1
p.Bind()
f.allowCooking = True
f.par.Cookrender = False
f.ApplyPreset('realtime')
c.par.Seed = 0; c.par.Fps = 60; c.par.Substeps = 1; c.par.Mode = 'realtime'; c.par.Running = True
c.par.Sims = '/sims/fluid2d /sims/particles'
b.par.Sim = '/sims/fluid2d'
b.par.Groupsims = '/sims/fluid2d /sims/particles'
b.par.Cachedir = {cache!r}
b.par.Framestart = 1; b.par.Frameend = {frames}; b.par.Warmup = 0
b.par.Resume = True; b.par.Pauseafter = True; b.par.Keyinterval = 0
b.ClearCache()
print('@@JSON@@' + json.dumps(1))
'''


def liquid_surface(td) -> dict:
    enable_sims(td)
    pics = run(td, r'''
import json, numpy as np
liq = op('/sims/liquid')
op('/sims/clock/tick').par.active = False
liq.ApplyPreset('realtime')
liq.par.Cookrender = True
op.simclock.par.Sims = '/sims/liquid'
op.simclock.par.Running = True
project.realTime = False
op.simclock.RequestReset()
for _ in range(8):
    op.simclock.Tick()
rend = liq.op('render')

def grab(smooth):
    # uSmooth is a vec uniform on the render. Missing name means the build is old.
    found = False
    for i in range(int(rend.seq.vec.numBlocks)):
        if str(getattr(rend.par, f'vec{i}name').eval()) == 'uSmooth':
            getattr(rend.par, f'vec{i}valuex').val = float(smooth)
            found = True
    rend.cook(force=True)
    img = np.array(rend.numpyArray(delayed=False), copy=True)
    return found, img

found0, raw = grab(0.0)
found1, smooth = grab(1.0)
np.save(project.folder + '/reports/phase8_liquid_raw.npy', np.ascontiguousarray(raw, np.float32))
np.save(project.folder + '/reports/phase8_liquid_smooth.npy', np.ascontiguousarray(smooth, np.float32))
print('@@JSON@@' + json.dumps({'uniform': found0 and found1, 'err': rend.errors() or ''}))
''', timeout=180)
    raw = np.load(REPORTS / "phase8_liquid_raw.npy")
    smooth = np.load(REPORTS / "phase8_liquid_smooth.npy")
    for path in (REPORTS / "phase8_liquid_raw.npy", REPORTS / "phase8_liquid_smooth.npy"):
        path.unlink(missing_ok=True)
    raw_png = save_png(raw, REPORTS / "phase8_liquid_raw.png")
    smooth_png = save_png(smooth, REPORTS / "phase8_liquid_smooth.png")
    hf_raw, axis_raw = surface_metrics(raw)
    hf_smooth, axis_smooth = surface_metrics(smooth)
    run(td, FPS_LIQUID, timeout=40)
    fps = measure_fps(td, 3.0)
    return {
        "uniform": pics.get("uniform"),
        "hf_raw": hf_raw, "hf_smooth": hf_smooth,
        "axis_raw": axis_raw, "axis_smooth": axis_smooth,
        "raw": raw_png, "smooth": smooth_png,
        "fps": fps,
    }


def surface_metrics(image: np.ndarray) -> tuple[float, float]:
    rgb = np.clip(image[..., :3], 0, 1)
    luma = rgb.mean(axis=2)
    acc = -4 * luma + np.roll(luma, 1, 0) + np.roll(luma, -1, 0) + np.roll(luma, 1, 1) + np.roll(luma, -1, 1)
    gy, gx = np.gradient(luma)
    mag = np.hypot(gx, gy)
    surface = mag > 0.02
    if int(surface.sum()) < 20:
        return float(np.var(acc)), 0.0
    ang = np.degrees(np.arctan2(np.abs(gy), np.abs(gx) + 1e-8))
    # 0° or 90° is an axis-aligned screen gradient, which a faceted voxel edge produces.
    axis = (ang < 2.0) | (ang > 88.0)
    return float(np.var(acc)), float(axis[surface].mean())


FPS_LIQUID = r'''
import json
op('/sims/liquid').par.Cookrender = True
op.simclock.par.Sims = '/sims/liquid'
op.simclock.par.Running = True
project.realTime = False
op('/sims/clock/tick').par.active = False
op.simclock.RequestReset()
op.simclock.Tick()
print('@@JSON@@' + json.dumps(1))
'''


def volume_detail(td) -> dict:
    enable_sims(td)
    cache = ROOT / "cache" / "phase4_showcase"
    if not (cache / "manifest.json").exists():
        return {"skipped": True, "reason": "no phase4 showcase cache"}
    pics = run(td, r'''
import json, numpy as np
r = op('/sims/cache_reader')
f = op('/sims/fluid3d')
op('/sims/clock/tick').par.active = False
r.par.Cachedir = 'cache/phase4_showcase'
r.par.Interpolate = False
r.par.Member = ''
r.par.Frame.expr = ''
r.Reload()
end = int(r.par.Frameend.eval())
frame = max(int(r.par.Framestart.eval()), end - 10)
r.Load(frame)
f.op('view_scalars').par.top = r.Output('scalars')
if r.op('vel_vol') is not None and f.op('view_vel') is not None:
    f.op('view_vel').par.top = r.Output('vel')
f.par.Renderw = 640
f.par.Renderh = 360
f.par.Raysteps = 24
rend = f.op('render')

def shot(detail):
    f.par.Detail = detail
    f.par.Detailfreq = 12
    f.par.Detailadvect = 0.35
    rend.cook(force=True)
    return np.array(rend.numpyArray(delayed=False), copy=True)

plain = shot(0.0)
detail = shot(1.2)
np.save(project.folder + '/reports/phase8_detail_plain.npy', np.ascontiguousarray(plain, np.float32))
np.save(project.folder + '/reports/phase8_detail.npy', np.ascontiguousarray(detail, np.float32))
print('@@JSON@@' + json.dumps({'frame': frame, 'err': rend.errors() or '', 'status': str(r.par.Status.eval())}))
''', timeout=180)
    plain = np.load(REPORTS / "phase8_detail_plain.npy")
    detail = np.load(REPORTS / "phase8_detail.npy")
    for path in (REPORTS / "phase8_detail_plain.npy", REPORTS / "phase8_detail.npy"):
        path.unlink(missing_ok=True)
    plain_png = save_png(plain, REPORTS / "phase8_detail_plain.png")
    detail_png = save_png(detail, REPORTS / "phase8_detail.png")

    def lap(img):
        luma = np.clip(img[..., :3], 0, 1).mean(axis=2)
        acc = -4 * luma + np.roll(luma, 1, 0) + np.roll(luma, -1, 0) + np.roll(luma, 1, 1) + np.roll(luma, -1, 1)
        return float(np.var(acc))

    # Obstacle-like pixels are the grey surface, not the warm plume. Detail must not move those.
    grey = (np.abs(plain[..., 0] - plain[..., 1]) < 0.04) & (plain[..., :3].mean(axis=2) > 0.15)
    warm = (plain[..., 0] > plain[..., 2] + 0.03) & (plain[..., :3].mean(axis=2) > 0.02)
    diff = np.abs(detail - plain)[..., :3].mean(axis=2)
    moved = float(diff[grey].mean()) if int(grey.sum()) > 50 else 0.0
    smoke = float(diff[warm].mean()) if int(warm.sum()) > 50 else 0.0
    return {
        "frame": pics.get("frame"),
        "lap_plain": lap(plain),
        "lap_detail": lap(detail),
        "lap_ratio": lap(detail) / max(lap(plain), 1e-12),
        "smoke_move": smoke,
        "obstacle_move": moved,
        "plain": plain_png,
        "detail": detail_png,
        "checker": detail_png["checker"],
    }


def combined_realtime(td) -> dict:
    enable_sims(td)
    info = run(td, r'''
import json
s = op('/sims/scenes')
info = s.Apply('realtime')
op('/sims/clock/tick').par.active = False
op.simclock.RequestReset()
op.simclock.Tick()
import numpy as np
liq0 = np.array(op('/sims/liquid/pos_out').numpyArray(delayed=False), copy=True)
bus = op.simbus3d
b, res = list(bus.Bounds), list(bus.Res)
follow = 0.0
# Pieces are still moving in the first half-second. After that they sleep and the
# projectile has left the grid, so the follow score has to be taken during the motion.
for i in range(24):
    op.simclock.Tick()
    if i not in (2, 6, 10, 14, 20):
        continue
    fluid = np.array(op('/sims/fluid3d/vel_out').numpyArray(delayed=False))
    for body in op('/sims/destruction').ext.Destruction.world.bodies:
        speed = float(np.linalg.norm(body.lin))
        if speed < 0.2:
            continue
        uv = [(float(body.pos[k]) - b[k]) / (b[k + 3] - b[k]) for k in range(3)]
        if any(v < 0 or v > 1 for v in uv):
            continue
        ijk = [int(np.clip(uv[k] * res[k], 0, res[k] - 1)) for k in range(3)]
        if fluid.ndim == 4:
            vel = fluid[ijk[2], ijk[1], ijk[0], :3]
        elif fluid.ndim == 3 and fluid.shape[-1] >= 3:
            vel = fluid[ijk[1], ijk[0], :3]
        else:
            continue
        fspeed = float(np.linalg.norm(vel))
        if fspeed < 0.05:
            continue
        follow = max(follow, float(np.dot(vel, body.lin) / (speed * fspeed)))
# Debris lives about two seconds, so the count has to be the peak during the break.
# The banner deflection is taken after the plume has had time to arrive.
debris_peak = 0
for i in range(100):
    op.simclock.Tick()
    if i % 10 == 0:
        pvel_i = np.array(op('/sims/particles/pvel_out').numpyArray(delayed=False))
        debris_peak = max(debris_peak, int(np.sum(pvel_i[..., 3] > 0)))
s.OnStep(None)
pic = np.array(s.op('picture_out').numpyArray(delayed=False), copy=True)
np.save(project.folder + '/reports/phase8_scene.npy', np.ascontiguousarray(pic, np.float32))
scalars = np.array(op('/sims/fluid3d/scalars_out').numpyArray(delayed=False))
z0 = scalars[0, ..., 0]
# A cache left wired to the solver wrote 0 and ~0.7 into the first slice.
hi = np.abs(z0 - 0.7) < 0.05
lo = z0 < 0.02
slice_checker = float(min(hi.mean(), lo.mean()))
liq = op('/sims/liquid')
lpos = np.array(liq.op('pos_out').numpyArray(delayed=False))
alive = lpos[..., 3] > 0
ys = lpos[..., 1][alive]
delta = np.linalg.norm((lpos[..., :3] - liq0[..., :3]).reshape(-1, 3), axis=1)
speeds = delta
parts = op('/sims/particles')
pvel = np.array(parts.op('pvel_out').numpyArray(delayed=False))
debris = max(debris_peak, int(np.sum(pvel[..., 3] > 0)))
cloth = np.array(op('/sims/cloth/pos_out').numpyArray(delayed=False))
free = cloth[..., 3] > 0
pinned = cloth[..., 3] == 0
rest_z = float(np.median(cloth[..., 2][pinned])) if pinned.any() else 0.15
defl = float(np.mean(np.abs(cloth[..., 2][free] - rest_z))) if free.any() else 0.0
bypass = {}
for path in ('/sims/fluid3d/render', '/sims/liquid/render', '/sims/destruction/pieces_render',
             '/sims/cloth/cloth_render', '/sims/particles/beauty'):
    node = op(path)
    bypass[path] = None if node is None else bool(node.bypass)
print('@@JSON@@' + json.dumps({
    'preset': info.get('preset'), 'sims': info.get('sims'), 'res': info.get('res'),
    'pieces': info.get('pieces'), 'particles': info.get('particles'),
    'render': [int(s.op('composite').par.resolutionw), int(s.op('composite').par.resolutionh)],
    'project': bool(liq.par.Project.eval()),
    'follow': follow, 'slice_checker': slice_checker,
    'liquid_y_std': float(ys.std()) if ys.size else 0.0,
    'liquid_speed': float(speeds.max()) if speeds.size else 0.0,
    'liquid_moved': int(np.sum(delta > 1e-3)),
    'debris': debris, 'banner_z': defl,
    'bypass': bypass,
    'err': s.op('composite').errors() or '',
}))
''', timeout=180)
    image = np.load(REPORTS / "phase8_scene.npy")
    (REPORTS / "phase8_scene.npy").unlink(missing_ok=True)
    png = save_png(image, REPORTS / "phase8_combined.png")
    run(td, r'''
import json
op('/sims/clock/tick').par.active = False
op.simclock.RequestReset()
op.simclock.Tick()
print('@@JSON@@' + json.dumps(1))
''', timeout=40)
    fps = measure_fps(td, 3.0)
    stages = run(td, r'''
import json, time
c = op.simclock
ext = c.ext.SimClockExt
op('/sims/clock/tick').par.active = False
c.Tick()
ext._sync_config(); ext.clock.advance(); ext._publish()
out = {}
for path in c.par.Sims.eval().split():
    t0 = time.perf_counter()
    op(path).BeginStep(False); op(path).Step()
    out[path.split('/')[-1]] = round((time.perf_counter() - t0) * 1000, 2)
t0 = time.perf_counter()
op('/sims/scenes').OnStep(None)
out['picture'] = round((time.perf_counter() - t0) * 1000, 2)
print('@@JSON@@' + json.dumps(out))
''', timeout=60)
    info["fps"] = fps
    info["stages_ms"] = stages
    info["png"] = png
    # The interaction numbers come from a fixed tick count, so two passes should agree exactly.
    # The protocol copies the first pass to phase8_run1.json; the fps window is wall-clock.
    keys = ("banner_z", "debris", "liquid_moved", "follow", "liquid_speed", "liquid_y_std")
    info["numbers"] = {key: info.get(key) for key in keys}
    first = REPORTS / "phase8_run1.json"
    if first.is_file():
        previous = (json.loads(first.read_text()).get("scene") or {}).get("numbers") or {}
        info["previous_numbers"] = previous
        info["match_previous"] = bool(previous) and all(info["numbers"][key] == previous.get(key) for key in keys)
    else:
        info["match_previous"] = True
    return info


def combined_showcase(td) -> dict:
    remove_tree(ROOT / SHOWCASE)
    remove_tree(ROOT / "cache" / "phase8_showcase_resume")
    run(td, SHOWCASE_SETUP, timeout=120)
    run(td, "op('/sims/baker').Bake()\nprint('@@JSON@@'+__import__('json').dumps(1))", timeout=40)
    picture_dir = ROOT / "cache" / "exports" / "frames"
    remove_tree(picture_dir)
    pump_until_done(td, cancel_at=None, pictures=picture_dir)
    full_files = list((ROOT / SHOWCASE).rglob("*"))
    # second run, cancel then resume, into a sibling directory
    other = "cache/phase8_showcase_resume"
    run(td, SHOWCASE_SETUP.replace(SHOWCASE, other), timeout=120)
    run(td, "op('/sims/baker').Bake()\nprint('@@JSON@@'+__import__('json').dumps(1))", timeout=40)
    pump_until_done(td, cancel_at=40)
    run(td, "op('/sims/baker').Bake()\nprint('@@JSON@@'+__import__('json').dumps(1))", timeout=40)
    pump_until_done(td, cancel_at=None)
    a_root, b_root = ROOT / SHOWCASE, ROOT / other
    rels = sorted(
        p.relative_to(a_root) for p in a_root.rglob("*") if p.is_file() and p.name != ".DS_Store"
    )
    bad = [str(r) for r in rels if not (b_root / r).exists() or (a_root / r).read_bytes() != (b_root / r).read_bytes()]
    manifest_diff = []
    if bad == ["manifest.json"]:
        da = json.loads((a_root / "manifest.json").read_text())
        db = json.loads((b_root / "manifest.json").read_text())

        def _walk(x, y, path=""):
            if isinstance(x, dict) and isinstance(y, dict):
                for key in sorted(set(x) | set(y)):
                    if x.get(key) != y.get(key):
                        if isinstance(x.get(key), (dict, list)) and isinstance(y.get(key), (dict, list)) and path.count(".") < 5:
                            _walk(x.get(key), y.get(key), path + "." + key)
                        else:
                            manifest_diff.append(path + "." + key)
            elif x != y:
                manifest_diff.append(path)
        _walk(da, db)
    remove_tree(b_root)
    manifest = json.loads((a_root / "manifest.json").read_text()) if (a_root / "manifest.json").exists() else {}
    frames = int(manifest.get("frame_end", 0)) - int(manifest.get("frame_start", 1)) + 1
    nbytes = sum(p.stat().st_size for p in a_root.rglob("*") if p.is_file())
    shots = showcase_frames(td)
    return {
        "frames": frames,
        "bytes": nbytes,
        "bad": len(bad),
        "mismatched": bad[:6],
        "manifest_diff": manifest_diff[:12],
        "complete": bool(manifest.get("complete")),
        "replay": shots.get("replay"),
        "replay_frames": shots.get("replay_frames"),
        "replay_mean": shots.get("replay_mean"),
        "replay_neighbor": shots.get("replay_neighbor"),
        "replay_neighbor_mean": shots.get("replay_neighbor_mean"),
        "replay_pairs": shots.get("replay_pairs"),
        "stills": shots.get("stills"),
        "pngs": shots.get("pngs"),
        "sheet": shots.get("sheet"),
    }


_FRAME_INDEX = re.compile(r"\.(\d{5})$")


def _frame_index(path: Path) -> int | None:
    match = _FRAME_INDEX.search(path.stem)
    return int(match.group(1)) if match else None


def _state_files(root: Path, limit: int | None) -> dict[str, Path]:
    """Cache files other than the manifest. `limit` keeps frames 1..limit and every file
    without a frame number. The manifest is compared field by field (_manifest_diff)."""
    found: dict[str, Path] = {}
    if not root.exists():
        return found
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith(".") or path.name == "manifest.json":
            continue
        index = _frame_index(path)
        if limit is not None and index is not None and not 1 <= index <= limit:
            continue
        found[str(path.relative_to(root))] = path
    return found


REPRO_FRAMES = 40
# The prefix bake's manifest differs from the 180-frame one only in its frame range.
RANGE_FIELDS = {"frame_end", "frames_written"}


def _manifest_diff(a, b, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key in RANGE_FIELDS:
                continue
            out += _manifest_diff(a.get(key), b.get(key), f"{path}.{key}")
        return out
    return [] if a == b else [path or "."]


def showcase_repro() -> dict:
    """Bake frames 1-40 from a fresh reset in a new scratch session (stop, start, build) and
    compare every file with the same frames of the 180-frame showcase, byte for byte."""
    scratch = str(ROOT / "scripts" / "td_scratch.py")
    for args in ((scratch, "stop"), (scratch, "start"), (str(ROOT / "scripts" / "build_project.py"),)):
        if subprocess.run([sys.executable, *args], cwd=ROOT).returncode:
            return {"error": " ".join(Path(a).name for a in args) + " failed"}
    td = attach()
    showcase_root = ROOT / SHOWCASE
    other_rel = "cache/phase8_showcase_repro"
    other = ROOT / other_rel
    remove_tree(other)
    setup = SHOWCASE_SETUP.replace(SHOWCASE, other_rel).replace(
        "b.par.Frameend = 180", f"b.par.Frameend = {REPRO_FRAMES}",
    )
    run(td, setup, timeout=180)
    run(td, "op('/sims/baker').Bake()\nprint('@@JSON@@'+__import__('json').dumps(1))", timeout=40)
    pump_until_done(td, cancel_at=None)
    left = _state_files(showcase_root, REPRO_FRAMES)
    right = _state_files(other, REPRO_FRAMES)
    bad = [rel for rel in sorted(set(left) | set(right))
           if rel not in left or rel not in right or left[rel].read_bytes() != right[rel].read_bytes()]
    manifest_diff = ["manifest.json missing"]
    if (showcase_root / "manifest.json").is_file() and (other / "manifest.json").is_file():
        manifest_diff = _manifest_diff(json.loads((showcase_root / "manifest.json").read_text()),
                                       json.loads((other / "manifest.json").read_text()))
    remove_tree(other)
    return {
        "frames": REPRO_FRAMES,
        "files": len(left),
        "bad": len(bad),
        "mismatched": bad[:8],
        "manifest_diff": manifest_diff[:8],
        "manifest_skipped": sorted(RANGE_FIELDS),
    }


def pump_until_done(td, cancel_at: int | None, pictures: Path | None = None) -> None:
    """One group-bake frame per call.

    `pictures` keeps the composite. The download has to be queued in the same script as
    the tick: delayed=True then returns that image on the next tick, which is the picture
    OnStep cooked, not a second cook of picture_out. The last frame is read back once
    the bake has stopped.
    """
    if pictures is not None:
        pictures.mkdir(parents=True, exist_ok=True)
        run(td, "op('/sims/baker').store('pic_frame', 0)\nprint('@@JSON@@'+__import__('json').dumps(1))", timeout=20)
    rel_dir = pictures.relative_to(ROOT).as_posix() if pictures is not None else ""
    extra = ""
    if pictures is not None:
        extra = f'''
import numpy as np, os
pic = op('/sims/scenes/picture_out')
img = pic.numpyArray(delayed=True)
prev = int(op('/sims/baker').fetch('pic_frame', 0) or 0)
folder = project.folder + '/' + {rel_dir!r}
os.makedirs(folder, exist_ok=True)
if prev and img is not None:
    np.save(folder + '/f%04d.npy' % prev, np.ascontiguousarray(np.array(img, dtype=np.float32, copy=True)))
op('/sims/baker').store('pic_frame', int(op.simclock.par.Frame.eval()))
'''
    while True:
        state = run(td, f'''
import json
b = op('/sims/baker')
baking = b.PumpOnce() if b.ext.Baker.baking else False
{extra}
print('@@JSON@@' + json.dumps({{'baking': bool(baking), 'frame': int(op.simclock.par.Frame.eval()),
    'status': str(b.par.Status.eval())}}))
''', timeout=180)
        print(" bake", state.get("frame"), state.get("status", "")[:60])
        if cancel_at is not None and state["baking"] and state["frame"] >= cancel_at:
            run(td, "op('/sims/baker').Cancel()\nprint('@@JSON@@'+__import__('json').dumps(1))", timeout=30)
            return
        if not state["baking"]:
            if pictures is not None:
                run(td, f'''
import json, numpy as np, os
pic = op('/sims/scenes/picture_out')
img = pic.numpyArray(delayed=True)
frame = int(op('/sims/baker').fetch('pic_frame', 0) or 0)
folder = project.folder + '/' + {rel_dir!r}
if frame and img is not None:
    np.save(folder + '/f%04d.npy' % frame, np.ascontiguousarray(np.array(img, dtype=np.float32, copy=True)))
print('@@JSON@@' + json.dumps({{'frame': frame, 'ok': img is not None}}))
''', timeout=60)
            return


SHOWCASE_SETUP = r'''
import json
s = op('/sims/scenes')
info = s.Apply('showcase')
b = op('/sims/baker')
c = op.simclock
op('/sims/clock/tick').par.active = False
b.par.Sim = '/sims/fluid3d'
b.par.Groupsims = c.par.Sims.eval()
b.par.Cachedir = 'cache/phase8_showcase'
b.par.Framestart = 1
b.par.Frameend = 180
b.par.Warmup = 0
b.par.Resume = True
b.par.Pauseafter = True
b.par.Keyinterval = 15
b.par.Crop = False
b.par.Asyncsave = False
b.ClearCache()
print('@@JSON@@' + json.dumps({'sims': info.get('sims'), 'res': info.get('res')}))
'''


# Rebuild one combined frame from the group cache. The live composite captured during the
# bake is cache/exports/frames/fNNNN.npy. While baking, the liquid snaps phi_out from the
# particles the frame saves (Liquid._publish_preview), so the still cooks phi from that frame.
RESTORE_STILL = r'''
import json, numpy as np
from pathlib import Path
frame = @@FRAME@@
root = project.folder
cache = root + '/cache/phase8_showcase'
err = []

def load_npy(sim, field, fr):
    path = Path(cache) / sim / field / ('%s.%05d.npy' % (field, fr))
    if not path.is_file():
        return None
    return np.ascontiguousarray(np.load(str(path)), dtype=np.float32)

def hold(comp, name, image):
    top = comp.op(name)
    if top is None or top.OPType != 'scriptTOP':
        if top is not None:
            top.destroy()
        top = comp.create(scriptTOP, name)
    top.par.outputresolution = 'custom'
    top.par.resolutionw = int(image.shape[1])
    top.par.resolutionh = int(image.shape[0])
    top.par.format = 'rgba32float'
    top.lock = True
    top.copyNumpyArray(image)
    return top

def bind(par, path):
    expr = par.expr or ''
    current = par.eval()
    saved = (expr, '' if current is None else str(current))
    par.expr = ''
    par.val = path
    return saved

def unbind(par, saved):
    expr, val = saved
    if expr:
        par.expr = expr
    elif val:
        par.val = val

clk = op.simclock
start = int(clk.par.Startframe.eval())
sub = max(1, int(clk.par.Substeps.eval()))
step = max(0, (int(frame) - start) * sub)
dt = float(clk.par.Dt.eval()) or (1.0 / 60.0)
clk.par.Step = step
clk.par.Frame = int(frame)
clk.par.Time = step * dt
bound = []

def remember(par, path):
    bound.append((par, bind(par, path)))

try:
    dest = op('/sims/destruction')
    if load_npy('destruction', 'bodies', frame) is None:
        err.append('bodies')
    else:
        dest.ext.Destruction.SetInit({'bodies': cache + '/destruction/bodies/bodies.%05d.exr' % frame})
    for name in ('obstacles3d', 'pieces_render'):
        node = dest.op(name)
        if node is not None:
            node.bypass = False
            node.cook(force=True)
    parts = op('/sims/particles')
    pos, pvel = load_npy('particles', 'pos', frame), load_npy('particles', 'pvel', frame)
    mat = parts.op('beauty_mat')
    if pos is None or pvel is None or mat is None:
        err.append('particles')
    else:
        remember(mat.par.sampler0top, hold(parts, 'still_pos', pos).path)
        remember(mat.par.sampler1top, hold(parts, 'still_pvel', pvel).path)
        for name in ('beauty_render', 'beauty'):
            node = parts.op(name)
            node.bypass = False
            node.cook(force=True)
    cloth = op('/sims/cloth')
    cpos = load_npy('cloth', 'pos', frame)
    cmat = cloth.op('cloth_mat')
    if cpos is None or cmat is None:
        err.append('cloth')
    else:
        remember(cmat.par.sampler0top, hold(cloth, 'still_pos', cpos).path)
        node = cloth.op('cloth_render')
        node.bypass = False
        node.cook(force=True)
    lpos, lvel = load_npy('liquid', 'pos', frame), load_npy('liquid', 'vel', frame)
    liq = op('/sims/liquid')
    if lpos is None or lvel is None:
        err.append('liquid')
    else:
        remember(liq.op('src_pos').par.top, hold(liq, 'still_pos', lpos).path)
        remember(liq.op('src_vel').par.top, hold(liq, 'still_vel', lvel).path)
        # Cook phi from the restored particles at its real resolution. Bumping the
        # width to force a recook left a different surface than the live step.
        for name in ('src_pos', 'src_vel', 'keys', 'radix_scatter_5', 'bins', 'p2g', 'solids', 'classify', 'phi'):
            node = liq.op(name)
            if node is not None:
                node.cook(force=True)
        liq.ext.Liquid._snap('phi_out')
        rend = liq.op('render')
        rend.bypass = False
        rend.cook(force=True)
    reader = op('/sims/cache_reader')
    reader.par.Cachedir = 'cache/phase8_showcase'
    reader.par.Interpolate = False
    reader.par.Frame.expr = ''
    reader.par.Member = 'fluid3d'
    reader.ext.CacheReader.Reload()
    reader.ext.CacheReader.Load(int(frame))
    vol = reader.Output('scalars')
    vol.cook(force=True)
    view = op('/sims/fluid3d/view_scalars')
    remember(view.par.top, vol.path)
    op('/sims/fluid3d/shadow').cook(force=True)
    fluid_r = op('/sims/fluid3d/render')
    fluid_r.bypass = False
    fluid_r.cook(force=True)
    pic = op('/sims/scenes/picture_out')
    img = np.array(pic.numpyArray(delayed=False), dtype=np.float32, copy=True)
    np.save(root + '/reports/phase8_still.npy', np.ascontiguousarray(img))

    def mean3(path):
        node = op(path)
        arr = node.numpyArray(delayed=False)
        if arr is None:
            return 0.0
        arr = np.asarray(arr)
        if getattr(arr, 'ndim', 0) < 3 or arr.shape[-1] < 3:
            return 0.0
        return float(arr[..., :3].mean())

    layers = {
        'smoke': mean3('/sims/fluid3d/render_out'),
        'pieces': mean3('/sims/destruction/pieces_render'),
        'liquid': mean3('/sims/liquid/render_out'),
        'cloth': mean3('/sims/cloth/cloth_render'),
        'debris': mean3('/sims/particles/beauty'),
    }
    print('@@JSON@@' + json.dumps({
        'frame': int(frame), 'err': err, 'shape': list(img.shape),
        'alpha_min': float(img[..., 3].min()) if img.shape[-1] > 3 else 0.0,
        'layers': layers, 'composite_err': op('/sims/scenes/with_debris').errors() or '',
    }))
finally:
    for par, saved in reversed(bound):
        unbind(par, saved)
'''


def _restore_still(td, frame: int):
    info = run(td, RESTORE_STILL.replace("@@FRAME@@", str(int(frame))), timeout=180)
    path = REPORTS / "phase8_still.npy"
    image = np.load(path) if path.is_file() else None
    path.unlink(missing_ok=True)
    return info, image


def showcase_frames(td) -> dict:
    """Composites at frames 36, 90 and 180, rebuilt from the cache, against the live grab."""
    # The resume bake arms the wall again. Rebuild that mesh from the same seed before
    # loading the saved bodies, or the pieces and the smoke obstacles are a different fracture.
    run(td, '''
import json
op('/sims/clock/tick').par.active = False
op('/sims/scenes').Apply('showcase')
print('@@JSON@@' + json.dumps(1))
''', timeout=180)
    frames = (36, 90, 180)
    live_dir = ROOT / "cache" / "exports" / "frames"
    rebuilt = []
    infos = []
    for frame in frames:
        info, image = _restore_still(td, frame)
        infos.append(info)
        rebuilt.append(image)
        print(" still", frame, info.get("shape"), info.get("layers"), info.get("err"))
    pairs = []
    same = []
    neighbors = []
    for frame, image in zip(frames, rebuilt):
        live_path = live_dir / f"f{frame:04d}.npy"
        live = np.load(live_path) if live_path.is_file() else None
        other = frame + 1 if frame < 180 else frame - 1
        _info_n, neighbor = _restore_still(td, other)
        def _err(a, b):
            if a is None or b is None or a.shape != b.shape:
                return None, None
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            return float(d.max()), float(d.mean())
        diff, mean = _err(image, live)
        ndiff, nmean = _err(neighbor, live)
        pairs.append({
            "frame": frame, "neighbor": other,
            "diff": diff, "mean": mean, "neighbor_diff": ndiff, "neighbor_mean": nmean,
        })
        if diff is not None:
            same.append((diff, mean))
        if nmean is not None:
            neighbors.append((ndiff, nmean))
        print(" replay", pairs[-1])
    pngs = []
    rgb = []
    checkers = []
    for frame, image, name in zip(frames, rebuilt, ("phase8_showcase_a.png", "phase8_showcase_b.png", "phase8_showcase_c.png")):
        if image is None:
            continue
        pic = image[..., :3]
        rgb.append(np.clip(pic, 0, 1))
        saved = save_png(pic, REPORTS / name)
        checkers.append(saved["checker"])
        pngs.append(saved)
    distinct = 0.0
    if len(rgb) >= 2:
        gaps = []
        for i in range(len(rgb)):
            for j in range(i + 1, len(rgb)):
                gaps.append(float(np.max(np.abs(rgb[i].astype(np.float64) - rgb[j].astype(np.float64)))))
        distinct = min(gaps) if gaps else 0.0
    sheet = np.concatenate(rgb, axis=1) if len(rgb) == 3 else None
    sheet_png = save_png(sheet, REPORTS / "phase8_showcase_sheet.png") if sheet is not None else None
    layers = [info.get("layers") or {} for info in infos]
    def _min_layer(key):
        vals = [row.get(key) for row in layers if row.get(key) is not None]
        return float(min(vals)) if vals else 0.0
    def _max_layer(key):
        vals = [row.get(key) for row in layers if row.get(key) is not None]
        return float(max(vals)) if vals else 0.0
    shape = next((info.get("shape") for info in infos if info.get("shape")), [0, 0, 0])
    alpha = [info.get("alpha_min") for info in infos if info.get("alpha_min") is not None]
    stills = {
        "frames": list(frames),
        "width": int(shape[1]) if len(shape) >= 2 else 0,
        "height": int(shape[0]) if shape else 0,
        "opaque": bool(alpha) and min(alpha) >= 0.99,
        "distinct": distinct,
        "checker": max(checkers) if checkers else 1.0,
        "pieces": _min_layer("pieces"),
        "liquid": _min_layer("liquid"),
        "smoke": _min_layer("smoke"),
        "cloth": _min_layer("cloth"),
        "debris": _max_layer("debris"),
        "layers": layers,
        "errors": [info.get("err") for info in infos],
    }
    return {
        "pngs": pngs,
        "sheet": sheet_png,
        "replay": max((row[0] for row in same), default=None),
        "replay_frames": len(same),
        "replay_mean": max((row[1] for row in same), default=None),
        "replay_neighbor": min((row[0] for row in neighbors), default=None),
        "replay_neighbor_mean": min((row[1] for row in neighbors), default=None),
        "replay_pairs": pairs,
        "stills": stills,
        "frames": list(frames),
    }


def export_showcase(td, showcase: dict) -> dict:
    folder = ROOT / "cache" / "exports"
    folder.mkdir(parents=True, exist_ok=True)
    if not (ROOT / SHOWCASE / "manifest.json").exists():
        return {"skipped": True, "frames": 0, "width": 0, "height": 0}
    frame_dir = folder / "frames"
    existing = sorted(frame_dir.glob("f*.npy")) if frame_dir.exists() else []
    if not existing:
        return {
            "frames": 0, "width": 0, "height": 0, "bytes": 0, "movie": None,
            "error": "bake did not capture composite frames",
        }
    width = height = 0
    for npy in existing:
        arr = np.load(npy)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            npy.unlink(missing_ok=True)
            return {
                "frames": 0, "width": 0, "height": 0, "bytes": 0, "movie": None,
                "error": f"{npy.name} is not a composite image",
            }
        height, width = int(arr.shape[0]), int(arr.shape[1])
        pic = arr[..., :3]
        peak = float(np.max(pic)) if pic.size else 0.0
        if peak > 1.01:
            pic = pic / max(peak, 1e-6)
        save_png(pic, npy.with_suffix(".png"))
        npy.unlink(missing_ok=True)
    movie = folder / "phase8_showcase.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "24", "-start_number", "1", "-i", str(frame_dir / "f%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(movie),
    ], check=False, capture_output=True)
    nbytes = movie.stat().st_size if movie.exists() else 0
    return {
        "frames": len(existing),
        "width": width,
        "height": height,
        "movie": rel(movie) if movie.exists() else None,
        "bytes": nbytes,
        "sequence": rel(frame_dir),
    }


def nested_validators() -> dict:
    scratch = str(ROOT / "scripts" / "td_scratch.py")
    build = str(ROOT / "scripts" / "build_project.py")

    def fresh(*args: str) -> int:
        if subprocess.run([sys.executable, scratch, "stop"], cwd=ROOT).returncode:
            return 1
        if subprocess.run([sys.executable, scratch, "start"], cwd=ROOT).returncode:
            return 1
        if subprocess.run([sys.executable, build], cwd=ROOT).returncode:
            return 1
        return subprocess.run([sys.executable, *args], cwd=ROOT).returncode

    out = {}
    out["bake_check"] = fresh(str(ROOT / "scripts" / "bake_check.py"), "--frames", "40")
    print(" nested bake_check", out["bake_check"])
    out["phase2"] = fresh(str(ROOT / "scripts" / "validate_phase2.py"))
    print(" nested phase2", out["phase2"])
    out["phase3"] = fresh(str(ROOT / "scripts" / "validate_phase3.py"))
    print(" nested phase3", out["phase3"])
    # Every earlier phase runs once here in its own session; their own nested chains would repeat them
    # for an hour and the late phase 3 frame-rate windows decay.
    out["phase4"] = fresh(str(ROOT / "scripts" / "validate_phase4.py"), "--quick", "--no-regressions")
    print(" nested phase4", out["phase4"])
    out["phase5"] = fresh(str(ROOT / "scripts" / "validate_phase5.py"), "--quick", "--no-regressions")
    print(" nested phase5", out["phase5"])
    out["phase6"] = fresh(str(ROOT / "scripts" / "validate_phase6.py"), "--quick", "--no-regressions")
    print(" nested phase6", out["phase6"])
    out["phase7"] = fresh(str(ROOT / "scripts" / "validate_phase7.py"), "--quick", "--no-regressions")
    print(" nested phase7", out["phase7"])
    return out


def evaluate(report: dict) -> bool:
    splat = report.get("splat") or {}
    fps = splat.get("fps") or {}
    group = report.get("group") or {}
    liquid = report.get("liquid") or {}
    lfps = liquid.get("fps") or {}
    upres = report.get("upres") or {}
    scene = report.get("scene") or {}
    sfps = scene.get("fps") or {}
    show = report.get("showcase") or {}
    stills = show.get("stills") or {}
    export = report.get("export") or {}
    nested = report.get("nested") or {}
    repro = report.get("repro") or {}
    bypass = scene.get("bypass") or {}
    ok = (
        splat.get("res") == [1024, 1024]
        and splat.get("recook", 1) == 0
        and splat.get("cpu", 1) < 1e-3
        and fps.get("fps", 0) >= 30
        and fps.get("steps_match_frames")
        and group.get("bad") == 0
        and group.get("files", 0) > 0
        and lfps.get("fps", 0) >= 33
        and lfps.get("steps_match_frames")
        and liquid.get("hf_smooth", 1) < liquid.get("hf_raw", 0)
        and (upres.get("skipped") or (upres.get("smoke_move") or 0) > 0.02)
        and (upres.get("skipped") or upres.get("obstacle_move", 1) < 0.01)
        and sfps.get("fps", 0) >= 40
        and sfps.get("steps_match_frames")
        and scene.get("render") == [480, 270]
        and all(v is False for v in bypass.values())
        and (scene.get("png") or {}).get("checker", 1) < 0.20
        and scene.get("slice_checker", 1) < 0.05
        and (scene.get("follow") or 0) >= 0.8
        and (scene.get("liquid_moved") or 0) >= 1000
        and (scene.get("banner_z") or 0) >= 0.1
        and (scene.get("debris") or 0) >= 100
        and (scene.get("pieces") or 0) >= 32
        and scene.get("project") is True
        and scene.get("match_previous") is True
        and (report.get("checker_witness") or 0) > 0.1
        and show.get("frames", 0) >= 180
        and show.get("bad") == 0
        and show.get("bytes", 10 ** 18) <= 3 * (1024 ** 3)
        and show.get("replay_frames") == 3
        and show.get("replay") == 0
        and all((row.get("neighbor_diff") or 0) > 0 for row in show.get("replay_pairs") or [{}])
        and repro.get("frames") == 40
        and repro.get("files", 0) > 0
        and repro.get("bad") == 0
        and repro.get("manifest_diff") == []
        and stills.get("width") == 480
        and stills.get("height") == 270
        and stills.get("opaque") is True
        and stills.get("distinct", 0) > 0.05
        and stills.get("checker", 1) < 0.20
        and stills.get("pieces", 0) > 1e-4
        and stills.get("liquid", 0) > 1e-4
        and stills.get("smoke", 0) > 1e-4
        and stills.get("cloth", 0) > 1e-4
        and stills.get("debris", 0) > 1e-5
        and export.get("frames") == 180
        and export.get("width") == 480
        and export.get("height") == 270
        and (export.get("bytes") or 10 ** 18) <= 500 * (1024 ** 2)
        and all(v == 0 for v in nested.values())
        and report.get("shaders") == 0
        and report.get("pytest") == 0
        and report.get("cache_bytes", 10 ** 18) <= CACHE_BUDGET
    )
    return bool(ok)


if __name__ == "__main__":
    sys.exit(main())
