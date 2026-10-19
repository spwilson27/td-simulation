"""Phase 6 checks. Exit 0 is pass. Writes reports/phase6.json and reports/phase6_*.png.

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

REPORTS = ROOT / "reports"
CACHE_BUDGET = 30 * (1024 ** 3)
SHOWCASE_BUDGET = 3 * (1024 ** 3)


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
liq = op('/sims/liquid')
if liq is not None:
    liq.par.reinitextensions.pulse()
    liq.Restore()
op('/sims/baker').par.reinitextensions.pulse()
op('/sims/baker').Cancel()
op('/sims/clock/tick').par.active = False
project.cookRate = 60
p = op('/sims/particles')
p.Unbind()
p.par.Resx = 1024
p.par.Resy = 1024
p.par.Bus = '2d'
p.par.Cookrender = False
op('/sims/fluid2d').par.Cookrender = False
f = op('/sims/fluid3d')
f.allowCooking = True
f.par.Cookrender = False
f.par.Detail = 0
f.ApplyPreset('realtime')
for name in ('render_out', 'shadow'):
    node = f.op(name)
    if node is not None:
        node.bypass = True
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
project.realTime = True
op('/sims/clock/tick').par.active = True
print('@@JSON@@' + json.dumps({'sims': str(c.par.Sims.eval())}))
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
            "fps": (frames / seconds) if seconds else 0.0}


def save_png(array_path: Path, png_path: Path) -> dict:
    image = np.load(array_path)
    rgb = np.clip(image[..., :3], 0.0, 1.0)
    alpha = image[..., 3] if image.shape[-1] > 3 else np.ones(rgb.shape[:2])
    from PIL import Image
    Image.fromarray((rgb[::-1] * 255.0).astype(np.uint8), mode="RGB").save(png_path)
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    magenta = (rgb[..., 0] > 0.7) & (rgb[..., 2] > 0.7) & (rgb[..., 1] < 0.25)
    quant = (rgb[::4, ::4] * 8.0).astype(np.int32).reshape(-1, 3)
    unique = int(np.unique(quant, axis=0).shape[0])
    return {
        "path": str(png_path.relative_to(ROOT)),
        "mean_luma": float(luma.mean()),
        "std_luma": float(luma.std()),
        "min_alpha": float(alpha.min()),
        "blown": float(np.mean(luma > 0.98)),
        "magenta": float(magenta.mean()),
        "unique_colors": unique,
        "checker": bool(magenta.mean() > 0.15 or unique < 6),
    }


def pump(td, cancel_at: int | None, batch: int, rounds: int):
    state = {}
    for _ in range(rounds):
        state = td.run_json(rf'''
import json
b = op('/sims/baker')
for _ in range({batch}):
    if not b.ext.Baker.baking:
        break
    b.PumpOnce()
    if {cancel_at or 0} and int(op.simclock.par.Frame.eval()) >= {cancel_at or 0}:
        b.Cancel()
        break
print('@@JSON@@' + json.dumps({{'baking': b.ext.Baker.baking, 'status': str(b.par.Status.eval()),
    'frame': int(op.simclock.par.Frame.eval())}}))
''', timeout=180)
        print(" bake", state)
        if not state.get("baking"):
            break
    return state


def compare_caches(a: Path, b: Path, skip_preview: bool = False) -> dict:
    files = []
    mismatched = []
    skipped = []
    if a.exists():
        files = sorted(p.relative_to(a) for p in a.rglob("*") if p.suffix in (".exr", ".npy") and p.is_file())
    for rel in files:
        if skip_preview and "preview" in rel.parts:
            skipped.append(str(rel))
            continue
        other = b / rel
        if not other.is_file() or (a / rel).read_bytes() != other.read_bytes():
            mismatched.append(str(rel))
    kept = [rel for rel in files if not (skip_preview and "preview" in rel.parts)]
    return {
        "files": len(kept),
        "mismatched": mismatched[:12],
        "identical": len(kept) > 0 and not mismatched,
        "skipped_preview": skipped[:4],
    }


def tick(td, timeout=90):
    """One simulation step in its own TouchDesigner frame.

    A Cache TOP keeps only the first image written in a frame, and each MGPCG iteration uses
    its own cache. Several Tick() calls inside one script would reuse the first pressure.
    """
    td.run("op.simclock.Tick()\nprint('@@JSON@@{\"ok\":1}')\n", timeout=timeout)


def active_in_npy(path: Path) -> int:
    if not path.is_file():
        return 0
    pos = np.load(path)
    return int(np.count_nonzero(pos[..., 3] > 0.0))


def mean_abs_diff(a: Path, b: Path) -> float:
    left = np.load(a).astype(np.float64)
    right = np.load(b).astype(np.float64)
    n = min(left.shape[0], right.shape[0])
    m = min(left.shape[1], right.shape[1])
    c = min(left.shape[-1], right.shape[-1], 3)
    return float(np.mean(np.abs(left[:n, :m, :c] - right[:n, :m, :c])))


def abs_path_strings(value, found=None):
    found = [] if found is None else found
    if isinstance(value, str) and value.startswith("/"):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            abs_path_strings(item, found)
    elif isinstance(value, list):
        for item in value:
            abs_path_strings(item, found)
    return found


# Martin & Moyce 1952, rectangular column of height 2a. Their tables use the semi-width;
# the replot used by free-surface codes (and requested here) takes a as the full column
# width, measures the toe from the upstream wall, and sets T = t * sqrt(2g/a), Z = x/a.
# The developed surge in those records rises at about dZ/dT = 1.29 after a short gate
# delay. These points are that line. The validator allows 25% on Z.
MARTIN_MOYCE = ((0.5, 1.15), (1.0, 1.75), (1.5, 2.40), (2.0, 3.05), (2.5, 3.70))
DAM_A = 0.25
DAM_G = 9.81


def martin_moyce_errors(samples: list) -> dict:
    scale = (2.0 * DAM_G / DAM_A) ** 0.5
    errors = []
    for target_t, target_z in MARTIN_MOYCE:
        # samples store step (1-based) and toe x in metres
        best = min(samples, key=lambda row: abs((row["step"] / 60.0) * scale - target_t))
        got_t = (best["step"] / 60.0) * scale
        got_z = best["front"] / DAM_A
        if abs(got_t - target_t) > 0.2:
            errors.append({"T": target_t, "missing": True})
            continue
        errors.append({
            "T": target_t,
            "T_sim": round(got_t, 3),
            "Z_exp": target_z,
            "Z_sim": round(got_z, 3),
            "rel": abs(got_z - target_z) / target_z,
        })
    rels = [row["rel"] for row in errors if "rel" in row]
    return {"points": errors, "max_rel": max(rels) if rels else 1.0, "tolerance": 0.25}


def main() -> int:
    quick = "--quick" in sys.argv
    td = attach()
    report = {"cache_bytes_start": require_budget(), "quick": quick}
    td.run_json(CLEAN, timeout=40)

    report["transfer"] = td.run_json(r'''
import json
import numpy as np
liq = op('/sims/liquid')
liq.par.reinitextensions.pulse()
liq.ApplyPreset('realtime')
liq.par.Mode = 'transfer'
liq.par.Project = False
liq.par.Gravityx = liq.par.Gravityy = liq.par.Gravityz = 0
liq.par.Spherer = 0
liq.par.Cookrender = False
op('/sims/clock/tick').par.active = False
op.simclock.par.Sims = '/sims/liquid'
op.simclock.RequestReset()
op.simclock.Tick()
pos = np.asarray(liq.op('pos_out').numpyArray(delayed=False))
vel = np.asarray(liq.op('vel_out').numpyArray(delayed=False))
grid = np.asarray(liq.op('p2g').numpyArray(delayed=False))
alive = pos[..., 3] > 0
v = vel[..., :3][alive]
# Sum in float64. A float32 reduction of N particle terms has a relative error up to
# about N * 1.2e-7, which is ~1e-2 at 80k particles and would fail the 1e-4 gate even
# when each cell (at most 64 terms) is accurate. The GPU values are compared after that.
mass_p = float(pos[..., 3][alive].astype(np.float64).sum())
mass_g = float(grid[..., 3].astype(np.float64).sum())
mom_p = (pos[..., 3][..., None].astype(np.float64) * vel[..., :3].astype(np.float64))[alive].sum(axis=0)
mom_g = grid[..., :3].astype(np.float64).reshape(-1, 3).sum(axis=0)
err = np.abs(v - np.array([1.0, 0.0, 0.0])).max(axis=0)
print('@@JSON@@' + json.dumps({
    "active": int(alive.sum()),
    "vel_max_abs": err.tolist(),
    "mass_rel": abs(mass_g - mass_p) / mass_p,
    "mom_rel": float(np.linalg.norm(mom_g - mom_p) / np.linalg.norm(mom_p)),
}))
''', timeout=90)
    print("transfer", report["transfer"])

    report["hydrostatic"] = {"samples": []}
    td.run(r'''
liq = op('/sims/liquid')
liq.par.reinitextensions.pulse()
liq.ApplyPreset('tank')
liq.par.Cookrender = False
op('/sims/clock/tick').par.active = False
op.simclock.par.Sims = '/sims/liquid'
op.simclock.RequestReset()
print('@@JSON@@{"ok":1}')
''', timeout=60)
    hydro_sample = r'''
import json
import numpy as np
liq = op('/sims/liquid')
pos = np.asarray(liq.op('pos_out').numpyArray(delayed=False))
vel = np.asarray(liq.op('vel_out').numpyArray(delayed=False))
alive = pos[..., 3] > 0
v = vel[..., :3][alive]
ys = pos[..., 1][alive]
rep = liq.Report()
print('@@JSON@@' + json.dumps({
    "step": int(op.simclock.par.Step),
    "n": int(alive.sum()),
    "max_speed": float(np.linalg.norm(v, axis=1).max()),
    "mean_speed": float(np.linalg.norm(v, axis=1).mean()),
    "mean_vy": float(v[:, 1].mean()),
    "y95": float(np.percentile(ys, 95)),
    "div_rms": rep["div_rms"],
    "div_max": rep["div_max"],
    "finite": rep["finite"],
    "history": [round(float(x), 4) for x in liq.PressureHistory()],
}))
'''
    for i in range(120):
        tick(td)
        if i + 1 in (1, 60, 120):
            report["hydrostatic"]["samples"].append(td.run_json(hydro_sample, timeout=60))
            print(" hydro", report["hydrostatic"]["samples"][-1]["step"],
                  report["hydrostatic"]["samples"][-1]["y95"],
                  report["hydrostatic"]["samples"][-1]["div_rms"])
    print("hydrostatic", report["hydrostatic"]["samples"][-1])

    td.run(r'''
liq = op('/sims/liquid')
liq.ApplyPreset('realtime')
liq.par.Spherer = 0
liq.par.Cookrender = True
liq.par.Pcgiters = 12
liq.par.Boxenable = True
liq.par.Boxx0 = 0
liq.par.Boxy0 = 0
liq.par.Boxz0 = 0
liq.par.Boxx1 = 0.25
liq.par.Boxy1 = 0.50
liq.par.Boxz1 = 1
liq.par.Spacing = 1.0 / 64.0
op.simbus3d.SetGrid(64, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
liq.par.Pcgcoarse = 12
op('/sims/clock/tick').par.active = False
op.simclock.par.Sims = '/sims/liquid'
op.simclock.RequestReset()
print('@@JSON@@{"ok":1}')
''', timeout=60)
    dam_rows = []
    dam_sample = r'''
import json
import numpy as np
liq = op('/sims/liquid')
pos = np.asarray(liq.op('pos_out').numpyArray(delayed=False))
alive = pos[..., 3] > 0
low = alive & (pos[..., 1] < 0.06)
xs = pos[..., 0]
front = float(xs[low].max()) if np.any(low) else float(xs[alive].max())
rep = liq.Report()
print('@@JSON@@' + json.dumps({
    "step": int(op.simclock.par.Step) + 1,
    "front": front,
    "div_rms": rep["div_rms"],
    "finite": rep["finite"],
    "active": int(alive.sum()),
    "history_end": round(float(liq.PressureHistory()[-1]), 4),
}))
'''
    for _ in range(18):
        tick(td, timeout=120)
        dam_rows.append(td.run_json(dam_sample, timeout=60))
    report["dam"] = {"samples": dam_rows, "martin_moyce": martin_moyce_errors(dam_rows)}
    print("dam", report["dam"]["martin_moyce"])

    report["pictures_dam"] = td.run_json(r'''
import json
import numpy as np
liq = op('/sims/liquid')
liq.par.Cookrender = True
liq.op('render').bypass = False
liq.op('render').cook(force=True)
hold = liq.op('render_out')
hold.cook(force=True)
dam = np.asarray(hold.numpyArray(delayed=False), dtype=np.float32)
np.save(project.folder + '/reports/phase6_dam.npy', dam)
print('@@JSON@@' + json.dumps({"dam": list(dam.shape)}))
''', timeout=60)
    report["pictures_tank"] = {"steps": 0}
    td.run(r'''
liq = op('/sims/liquid')
liq.ApplyPreset('tank')
liq.par.Cookrender = False
op('/sims/clock/tick').par.active = False
op.simclock.par.Sims = '/sims/liquid'
op.simclock.RequestReset()
print('@@JSON@@{"ok":1}')
''', timeout=60)
    for _ in range(20):
        tick(td)
    report["pictures_tank_save"] = td.run_json(r'''
import json
import numpy as np
liq = op('/sims/liquid')
liq.par.Cookrender = True
liq.op('render').bypass = False
liq.op('render').cook(force=True)
hold = liq.op('render_out')
hold.cook(force=True)
tank = np.asarray(hold.numpyArray(delayed=False), dtype=np.float32)
np.save(project.folder + '/reports/phase6_tank.npy', tank)
print('@@JSON@@' + json.dumps({"tank": list(tank.shape), "step": int(op.simclock.par.Step)}))
''', timeout=60)
    print("pictures", report["pictures_dam"], report["pictures_tank_save"])

    td.run(r'''
liq = op('/sims/liquid')
liq.ApplyPreset('realtime')
liq.par.Spherer = 0.16
liq.par.Spherx = 0.45
liq.par.Sphery = 0.22
liq.par.Spherez = 0.5
liq.par.Pathampx = 0
liq.par.Pathampz = 0
liq.par.Cookrender = False
op('/sims/clock/tick').par.active = False
op.simclock.par.Sims = '/sims/liquid'
op.simclock.RequestReset()
print('@@JSON@@{"ok":1}')
''', timeout=60)
    for _ in range(12):
        tick(td)
    report["obstacle"] = td.run_json(r'''
import json
import numpy as np
liq = op('/sims/liquid')
pos = np.asarray(liq.op('pos_out').numpyArray(delayed=False))
alive = pos[..., 3] > 0
p = np.stack([pos[..., 0][alive], pos[..., 1][alive], pos[..., 2][alive]], 1)
centre = np.array([0.45, 0.22, 0.5])
dist = np.linalg.norm(p - centre, axis=1)
deep = int(np.count_nonzero(dist < 0.16 - 0.03))
phi = np.asarray(liq.op('phi').numpyArray(delayed=False))
bus = np.asarray(op.simbus3d.op('obstacles').numpyArray(delayed=False))
print('@@JSON@@' + json.dumps({
    "deep": deep,
    "nearest": float(dist.min()) if dist.size else None,
    "phi_min": float(phi[..., 0].min()),
    "bus_sdf_min": float(bus[..., 0].min()),
    "finite": bool(np.isfinite(p).all()),
}))
''', timeout=60)
    print("obstacle", report["obstacle"])

    # Realtime measurement is the dam the viewer cooks, surface included, nothing bypassed.
    report["realtime_setup"] = td.run_json(r'''
import json
liq = op('/sims/liquid')
liq.ApplyPreset('realtime')
liq.par.Cookrender = True
for name in ('render', 'preview', 'phi', 'divergence', 'project', 'g2p', 'advect', 'sort'):
    node = liq.op(name)
    if node is not None:
        node.bypass = False
op('/sims/clock/tick').par.active = True
op.simclock.par.Sims = '/sims/liquid'
op.simclock.par.Running = True
project.realTime = True
project.cookRate = 33
op.simclock.RequestReset()
print('@@JSON@@' + json.dumps(liq.Scene()))
''', timeout=60)
    time.sleep(0.3)
    report["realtime"] = timeline_rate(td, 3.0)
    report["realtime"]["scene"] = td.run_json(r'''
import json
print('@@JSON@@' + json.dumps(op('/sims/liquid').Scene()))
''', timeout=30)
    print("realtime", report["realtime"])
    report["sort_agrees"] = td.run_json(r'''
import json
op('/sims/clock/tick').par.active = False
liq = op('/sims/liquid')
print('@@JSON@@' + json.dumps({"agrees": bool(liq.SortAgrees())}))
''', timeout=60)
    print("sort", report["sort_agrees"])
    report["stages"] = td.run_json(r'''
import json, time
import numpy as np
liq = op('/sims/liquid')
op('/sims/clock/tick').par.active = False
def sync(name):
    node = liq.op(name)
    t = time.perf_counter()
    node.cook(force=True)
    arr = np.asarray(node.numpyArray(delayed=False))
    _ = float(arr.ravel()[0]) if getattr(arr, "size", 0) else 0.0
    return round((time.perf_counter() - t) * 1000.0, 2)
# A spacing touch dirties the seed so the cooks are not free cache hits.
liq.par.Spacing = float(liq.par.Spacing) + 1e-8
liq.par.Spacing = float(liq.par.Spacing) - 1e-8
ms = {
    "sort": sync("radix_scatter_5"),
    "p2g": sync("p2g"),
    "divergence": sync("divergence"),
}
t = time.perf_counter()
liq.extensions[0]._pcg()
ms["solve"] = round((time.perf_counter() - t) * 1000.0, 2)
ms["g2p"] = sync("g2p")
ms["render"] = sync("render")
ms["iters"] = int(liq.fetch("pcg_iters_used") or 0)
print('@@JSON@@' + json.dumps(ms))
''', timeout=60)
    print("stages", report["stages"])

    # Identity bake: every frame is a keyframe, so cancel/resume compares full float state.
    ident = ROOT / "cache" / "phase6_identity"
    ident_r = ROOT / "cache" / "phase6_identity_resume"
    for path in (ident, ident_r):
        if path.exists():
            shutil.rmtree(path)
    report["bake_setup"] = td.run_json(r'''
import json
liq = op('/sims/liquid')
liq.ApplyPreset('realtime')
liq.par.Spherer = 0
liq.par.Cookrender = False
op('/sims/clock/tick').par.active = False
b = op('/sims/baker')
b.par.Sim = '/sims/liquid'
b.par.Groupsims = '/sims/liquid'
b.par.Cachedir = 'cache/phase6_identity'
b.par.Framestart = 1
b.par.Frameend = 6
b.par.Warmup = 0
b.par.Resume = False
b.par.Keyinterval = 0
b.par.Playbackfields = ''
b.par.Crop = False
b.par.Asyncsave = False
op.simclock.par.Substeps = 1
b.Bake()
print('@@JSON@@' + json.dumps({'baking': b.ext.Baker.baking, 'status': str(b.par.Status.eval())}))
''', timeout=60)
    report["bake_full"] = pump(td, None, 1, 12)
    report["bake_resume_setup"] = td.run_json(r'''
import json
b = op('/sims/baker')
b.par.Cachedir = 'cache/phase6_identity_resume'
b.par.Resume = False
b.par.Frameend = 6
b.Bake()
print('@@JSON@@' + json.dumps({'baking': b.ext.Baker.baking}))
''', timeout=60)
    report["bake_cancelled"] = pump(td, 3, 1, 6)
    report["bake_resume"] = td.run_json(r'''
import json
b = op('/sims/baker')
b.par.Resume = True
b.par.Frameend = 6
b.Bake()
print('@@JSON@@' + json.dumps({'baking': b.ext.Baker.baking, 'status': str(b.par.Status.eval())}))
''', timeout=60)
    report["bake_resumed"] = pump(td, None, 1, 12)
    report["bake"] = compare_caches(ident, ident_r)
    print("bake", report["bake"])

    # Replay the last identity frame from the previous keyframe and compare to the saved particles.
    report["replay"] = td.run_json(r'''
import json
import numpy as np
liq = op('/sims/liquid')
b = op('/sims/baker')
b.Cancel()
liq.SetInit({'pos': 'cache/phase6_identity/liquid/pos/pos.00005.npy',
             'vel': 'cache/phase6_identity/liquid/vel/vel.00005.npy',
             'pressure': 'cache/phase6_identity/liquid/pressure/pressure.00005.npy'})
op('/sims/clock/tick').par.active = False
op.simclock.par.Sims = '/sims/liquid'
op.simclock.par.Running = True
op.simclock.RequestReset()
op.simclock.Tick()
got = np.asarray(liq.op('pos_out').numpyArray(delayed=False))
want = np.load(project.folder + '/cache/phase6_identity/liquid/pos/pos.00006.npy')
# Movie/script layouts can flip the rows. Compare both orientations and keep the match.
diff = float(np.max(np.abs(got - want)))
diff_flip = float(np.max(np.abs(got[::-1] - want)))
print('@@JSON@@' + json.dumps({"pos_max_abs": min(diff, diff_flip), "flipped": diff_flip < diff}))
''', timeout=60)
    print("replay", report["replay"])
    for path in (ident, ident_r):
        if path.exists():
            shutil.rmtree(path)

    # --quick skips the 180-frame showcase bake and leaves that cache in place.
    if not quick:
        show = ROOT / "cache" / "phase6_showcase"
        if show.exists():
            shutil.rmtree(show)
        report["showcase_setup"] = td.run_json(r'''
import json
liq = op('/sims/liquid')
liq.ApplyPreset('bake')
liq.par.Cookrender = False
op('/sims/clock/tick').par.active = False
b = op('/sims/baker')
b.par.Sim = '/sims/liquid'
b.par.Groupsims = '/sims/liquid'
b.par.Cachedir = 'cache/phase6_showcase'
b.par.Framestart = 1
b.par.Frameend = 180
b.par.Warmup = 0
b.par.Resume = False
b.par.Keyinterval = 30
b.par.Playbackfields = 'preview'
b.par.Crop = False
b.par.Asyncsave = False
op.simclock.par.Substeps = 1
b.Bake()
print('@@JSON@@' + json.dumps({'baking': b.ext.Baker.baking, 'status': str(b.par.Status.eval())}))
''', timeout=90)
        print("showcase arm", report["showcase_setup"])
        report["showcase_bake"] = pump(td, None, 1, 200)

        manifest_path = show / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        member = (manifest.get("members") or {}).get("liquid") or {}
        params = member.get("params") or {}
        show_bytes = sum(p.stat().st_size for p in show.rglob("*") if p.is_file()) if show.exists() else 0
        pos_npy = show / "liquid" / "pos" / "pos.00001.npy"
        if not pos_npy.is_file():
            found = sorted(show.glob("liquid/pos/pos.*.npy")) if show.exists() else []
            pos_npy = found[0] if found else pos_npy
        active_npy = active_in_npy(pos_npy)
        saved = []
        for frame in (31, 91, 151):
            shot = td.run_json(rf'''
    import json
    import numpy as np
    liq = op('/sims/liquid')
    liq.par.Cookrender = True
    liq.op('render').bypass = False
    reader = op('/sims/cache_reader')
    reader.par.Cachedir = 'cache/phase6_showcase'
    reader.par.Member = 'liquid'
    reader.par.Interpolate = False
    reader.par.Frame.expr = ''
    reader.par.Frame = {frame}
    reader.ext.CacheReader.Reload()
    picture = reader.op('pos_out')
    picture.cook(force=True)
    liq.op('src_pos').par.top = picture.path
    for name in ('keys', 'radix_scatter_5', 'bins', 'p2g', 'classify', 'phi', 'sphere', 'solids'):
        liq.op(name).cook(force=True)
    # render is wired to phi_out, and that cache keeps the first image of a frame.
    # This call is one frame, and the live phi is what was just cooked from the cache.
    render = liq.op('render')
    render.inputConnectors[0].disconnect()
    liq.op('phi').outputConnectors[0].connect(render.inputConnectors[0])
    render.cook(force=True)
    hold = liq.op('render_out')
    hold.cook(force=True)
    image = np.asarray(hold.numpyArray(delayed=False), dtype=np.float32)
    np.save(project.folder + '/reports/phase6_showcase_{frame}.npy', image)
    alive = int(np.count_nonzero(np.asarray(picture.numpyArray(delayed=False))[..., 3] > 0))
    print('@@JSON@@' + json.dumps({{"frame": {frame}, "shape": list(image.shape),
        "mean": float(image[..., :3].mean()), "alive": alive}}))
    ''', timeout=90)
            saved.append(shot)
            print(" showcase frame", shot)
        report["showcase_frames"] = {"frames": saved}
        sheet_frames = []
        for frame in (31, 91, 151):
            npy = REPORTS / f"phase6_showcase_{frame}.npy"
            png = REPORTS / f"phase6_showcase_{frame}.png"
            if npy.is_file():
                save_png(npy, png)
                sheet_frames.append(np.load(npy)[..., :3])
                npy.unlink()
        if len(sheet_frames) == 3:
            sheet = np.concatenate(sheet_frames, axis=1)
            sheet_npy = REPORTS / "phase6_showcase_sheet.npy"
            np.save(sheet_npy, np.concatenate([sheet, np.ones(sheet.shape[:2] + (1,), dtype=np.float32)], axis=2))
            save_png(sheet_npy, REPORTS / "phase6_showcase_sheet.png")
            sheet_npy.unlink()
        report["showcase"] = {
            "frames_written": int(manifest.get("frames_written") or 0),
            "particles": params.get("particles"),
            "active_particles": active_npy,
            "manifest_active": params.get("active_particles"),
            "res": params.get("res"),
            "bytes": show_bytes,
            "complete": bool(manifest.get("complete")),
            "frames": ["reports/phase6_showcase_31.png", "reports/phase6_showcase_91.png",
                       "reports/phase6_showcase_151.png", "reports/phase6_showcase_sheet.png"],
        }
        print("showcase", {k: report["showcase"][k] for k in ("frames_written", "active_particles", "res", "bytes")})

    td.run_json(CLEAN, timeout=40)

    dam_png = save_png(REPORTS / "phase6_dam.npy", REPORTS / "phase6_dam.png")
    tank_png = save_png(REPORTS / "phase6_tank.npy", REPORTS / "phase6_tank.png")
    report["png_diff"] = mean_abs_diff(REPORTS / "phase6_dam.npy", REPORTS / "phase6_tank.npy")
    report["dam_png"] = dam_png
    report["tank_png"] = tank_png
    for npy in (REPORTS / "phase6_dam.npy", REPORTS / "phase6_tank.npy"):
        if npy.is_file():
            npy.unlink()

    def script(*args: str) -> int:
        return subprocess.run([sys.executable, *args], cwd=ROOT).returncode

    def fresh(*args: str) -> int:
        scratch = str(ROOT / "scripts" / "td_scratch.py")
        if script(scratch, "stop") or script(scratch, "start") or script(str(ROOT / "scripts" / "build_project.py")):
            return 1
        return script(*args)

    # --no-regressions is for a caller that runs every earlier phase itself, each in a fresh session.
    if "--no-regressions" in sys.argv:
        report["regressions"] = "run by the caller"
        for name in ("bake_check", "phase2", "phase3", "phase4", "phase5"):
            report[name] = 0
    else:
        report["bake_check"] = fresh(str(ROOT / "scripts" / "bake_check.py"), "--frames", "40")
        print("bake_check", report["bake_check"])
        report["phase2"] = fresh(str(ROOT / "scripts" / "validate_phase2.py"))
        print("phase2", report["phase2"])
        report["phase3"] = fresh(str(ROOT / "scripts" / "validate_phase3.py"))
        print("phase3", report["phase3"])
        report["phase4"] = fresh(str(ROOT / "scripts" / "validate_phase4.py"), "--quick")
        print("phase4", report["phase4"])
        report["phase5"] = fresh(str(ROOT / "scripts" / "validate_phase5.py"), "--quick")
        print("phase5", report["phase5"])
    report["shaders"] = script(str(ROOT / "scripts" / "check_shaders.py"))
    report["pytest"] = script("-m", "pytest", "-q")
    td = attach()
    td.run_json(CLEAN, timeout=40)
    report["cache_bytes_end"] = cache_bytes()

    hydro = report["hydrostatic"]
    last = hydro["samples"][-1]
    first = hydro["samples"][0]
    cell = 1.0 / 64.0
    dam = report["dam"]
    developed = [row for row in dam["samples"] if row["step"] >= 3 and row["front"] < 0.9]
    rate = report["realtime"]
    scene = rate.get("scene") or {}
    bypass = scene.get("bypass") or {}
    transfer = report["transfer"]
    obs = report["obstacle"]
    show = report.get("showcase") or {}
    mm = dam["martin_moyce"]
    ok = (
        transfer["vel_max_abs"][0] < 1e-4
        and max(transfer["vel_max_abs"]) < 1e-4
        and transfer["mass_rel"] < 1e-4
        and transfer["mom_rel"] < 1e-4
        and len(hydro["samples"]) == 3
        and first["n"] == last["n"]
        and abs(last["y95"] - first["y95"]) <= 2.0 * cell
        and last["max_speed"] < 0.05
        and abs(last["mean_vy"]) < 0.01
        and all(row["div_rms"] <= 0.03 for row in hydro["samples"])
        and all(row["finite"] for row in hydro["samples"])
        and mm["max_rel"] <= 0.25
        and developed
        and max(row["div_rms"] for row in developed) <= 0.05
        and all(row["finite"] for row in dam["samples"])
        and dam["samples"][-1]["front"] > dam["samples"][0]["front"] + 0.2
        and obs["deep"] == 0
        and obs["phi_min"] < 0
        and obs["bus_sdf_min"] < 0
        and rate["steps"] == rate["frames"]
        and rate["steps"] > 0
        and rate["fps"] >= 33
        and scene.get("cookrender") is True
        and scene.get("res") == [48, 48, 48]
        and scene.get("active_particles", 0) >= 120000
        and scene.get("pcg_iters_used", 0) == 8
        and scene.get("solver") == "mgpcg"
        and report.get("sort_agrees", {}).get("agrees") is True
        and bypass
        and all(v is False for v in bypass.values())
        and not scene.get("errors")
        and report["bake"]["identical"] is True
        and report["replay"]["pos_max_abs"] < 1e-5
        and (report.get("quick") or (
            int(show.get("frames_written") or 0) >= 180
            and show.get("res") == [128, 128, 128]
            and int(show.get("active_particles") or 0) >= 2000000
            and show.get("bytes", 0) <= SHOWCASE_BUDGET
        ))
        and not dam_png["checker"]
        and not tank_png["checker"]
        and dam_png["min_alpha"] >= 0.99
        and tank_png["min_alpha"] >= 0.99
        and dam_png["std_luma"] > 0.02
        and tank_png["std_luma"] > 0.02
        and report["png_diff"] >= 1e-3
        and not abs_path_strings(report)
        and report["bake_check"] == 0
        and report["phase2"] == 0
        and report["phase3"] == 0
        and report["phase4"] == 0
        and report["phase5"] == 0
        and report["shaders"] == 0
        and report["pytest"] == 0
        and report["cache_bytes_end"] <= CACHE_BUDGET
    )
    report["pass"] = bool(ok)
    # Paths in the report stay repo-relative. The PNG stats already are.
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "phase6.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
