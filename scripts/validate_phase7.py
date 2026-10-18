"""Phase 7 checks. Exit 0 is pass. Writes reports/phase7.json and reports/phase7_*.png.

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

from cloth import (  # noqa: E402
    KINDS,
    edge_strains,
    grid_positions,
    inverse_mass,
    rest_lengths,
    spacing_of,
    theoretical_strain,
    xpbd_substep,
)

REPORTS = ROOT / "reports"
CACHE_BUDGET = 30 * (1024 ** 3)
SHOWCASE_BUDGET = 3 * (1024 ** 3)
KEEP = {"phase3_showcase", "phase4_showcase", "phase5_showcase", "phase6_showcase", "phase7_showcase"}


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


def drop_validation_caches() -> None:
    root = ROOT / "cache"
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.name in KEEP or not child.is_dir():
            continue
        shutil.rmtree(child)


CLEAN = r'''
import json
c = op('/sims/cloth')
if c is not None:
    c.par.reinitextensions.pulse()
    c.PublishSphere(False)
    c.PublishForce(False)
    c.SetUniformWind(0, 0, 0)
    c.par.Cookrender = False
    c.par.Couple = False
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
clock = op.simclock
clock.par.Sims = '/sims/fluid2d'
clock.par.Substeps = 1
clock.par.Fps = 60
clock.par.Mode = 'realtime'
clock.par.Seed = 0
clock.par.Running = True
clock.par.Startframe = 0
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
print('@@JSON@@' + json.dumps({'sims': str(clock.par.Sims.eval())}))
'''

MARK = r'''
import json
print('@@JSON@@' + json.dumps({'step': int(op.simclock.par.Step.eval()), 'frame': int(absTime.frame)}))
'''


def timeline_rate(td, seconds=3.0):
    start = td.run_json(MARK, timeout=30)
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


def ticks(td, n, timeout=180):
    return td.run_json(rf'''
import json
op('/sims/clock/tick').par.active = False
for _ in range({int(n)}):
    op.simclock.Tick()
print('@@JSON@@' + json.dumps({{'step': int(op.simclock.par.Step.eval()), 'frame': int(op.simclock.par.Frame.eval())}}))
''', timeout=timeout)


def pump(td, cancel_at, batch, rounds):
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


def compare_caches(a: Path, b: Path) -> dict:
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.suffix in (".exr", ".npy") and p.is_file()) if a.exists() else []
    mismatched = [str(rel) for rel in files if not (b / rel).is_file() or (a / rel).read_bytes() != (b / rel).read_bytes()]
    return {"files": len(files), "mismatched": mismatched[:12], "identical": len(files) > 0 and not mismatched}


def abs_path_strings(value, found=None):
    found = [] if found is None else found
    if isinstance(value, str) and value.startswith("/") and not value.startswith("/sims"):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            abs_path_strings(item, found)
    elif isinstance(value, list):
        for item in value:
            abs_path_strings(item, found)
    return found


def speckle_fraction(array_path: Path) -> dict:
    """Fraction of cloth pixels whose front/back colour disagrees with a neighbour.

    Distance bending flipped triangles and sprinkled the other colour through the sheet.
    Isometric bending should leave that fraction small: a real fold is a boundary, not a checker.
    """
    image = np.load(array_path)
    rgb = np.clip(image[..., :3].astype(np.float64), 0.0, 1.0)
    # Lighting scales the face colour, so compare hue (red front, blue back) and only count
    # a pixel that disagrees with both neighbours. A fold edge disagrees with one side;
    # a flipped triangle is an isolated speck.
    hue = rgb[..., 0] - rgb[..., 2]
    is_front = hue > 0.0
    cloth = (np.abs(hue) > 0.04) & (rgb.max(axis=-1) > 0.06)
    left = np.roll(is_front, 1, axis=1)
    right = np.roll(is_front, -1, axis=1)
    left_c = np.roll(cloth, 1, axis=1)
    right_c = np.roll(cloth, -1, axis=1)
    speck = cloth & left_c & right_c & (is_front != left) & (is_front != right)
    count = int(cloth.sum())
    return {"fraction": float(speck.sum() / max(count, 1)), "cloth_pixels": count}


def grab(td, name: str) -> Path:
    path = REPORTS / f"{name}.npy"
    td.run_json(rf'''
import json
import numpy as np
c = op('/sims/cloth')
c.par.Cookrender = True
c.op('cloth_render').cook(force=True)
c.op('composite').cook(force=True)
image = np.asarray(c.op('picture_out').numpyArray(delayed=False), dtype=np.float32)
np.save(project.folder + '/reports/{name}.npy', image)
err = c.errors(recurse=True)
print('@@JSON@@' + json.dumps({{'shape': list(image.shape), 'errors': str(err)[:300]}}))
''', timeout=90)
    return path


def orient(gpu: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float, bool]:
    xyz = gpu[..., :3].astype(np.float64)
    best = None
    flipped = False
    for candidate, flag in ((xyz, False), (xyz[::-1], True)):
        if candidate.shape != ref.shape:
            continue
        err = float(np.max(np.abs(candidate - ref)))
        if best is None or err < best[0]:
            best = (err, candidate, flag)
    if best is None:
        return xyz, 1e9, False
    return best[1], best[0], best[2]


def reference_steps(pos, vel, inv, rest, bits, steps, substeps, compliance, gravity, damping, lengths):
    """Match the GPU: substep velocity inside a frame, frame-average velocity between frames."""
    frame_dt = 1.0 / 60.0
    dt = frame_dt / substeps
    for _ in range(steps):
        start = pos.copy()
        for _sub in range(substeps):
            pos, vel, bits = xpbd_substep(
                pos, vel, inv, rest, bits, dt, compliance, gravity, damping, lengths,
                air=0.0, thickness=0.0, ground=None, pin="left")
        vel = (pos - start) / frame_dt
    return pos, vel


def main() -> int:
    td = attach()
    report = {"cache_bytes_start": require_budget()}
    td.run_json(CLEAN, timeout=60)
    td.run_json(r'''
import json
op('/sims/clock/tick').par.active = False
c = op('/sims/cloth')
c.par.Resx = 8
c.par.Resy = 8
c.par.Substeps = 2
c.par.Selfiters = 1
c.par.Plane = 'xy'
c.par.Pin = 'left'
c.par.Originx = -0.2
c.par.Originy = 1.0
c.par.Originz = 0.1
c.par.Sizex = 0.35
c.par.Sizey = 0.28
c.par.Density = 0.25
c.par.Stretch = 2e-5
c.par.Shear = 2e-5
c.par.Bend = 5e-4
c.par.Damping = 0.6
c.par.Gravityx = 0
c.par.Gravityy = -9.81
c.par.Gravityz = 0
c.par.Air = 0
c.par.Thickness = 0
c.par.Collide = False
c.par.Tear = 0
c.par.Couple = False
c.par.Cookrender = False
c.SetUniformWind(0, 0, 0)
c.PublishSphere(False)
op.simclock.par.Sims = '/sims/cloth'
op.simclock.par.Fps = 60
op.simclock.par.Substeps = 1
op.simclock.par.Running = True
op.simclock.RequestReset()
print('@@JSON@@' + json.dumps({'ok': 1}))
''', timeout=40)
    ticks(td, 4, timeout=120)
    gpu = td.run_json(r'''
import json
import numpy as np
c = op('/sims/cloth')
pos = np.asarray(c.Positions(), dtype=np.float32)
np.save(project.folder + '/reports/phase7_match.npy', pos)
err = c.errors(recurse=True)
print('@@JSON@@' + json.dumps({'shape': list(pos.shape), 'errors': str(err)[:400],
    'finite': bool(np.isfinite(pos).all())}))
''', timeout=40)
    report["gpu_errors"] = gpu
    print("gpu", gpu)
    got = np.load(REPORTS / "phase7_match.npy")
    nx = ny = 8
    origin = (-0.2, 1.0, 0.1)
    size = (0.35, 0.28)
    ref = grid_positions(nx, ny, origin, size, "xy")
    dx, dy = spacing_of(size[0], nx), spacing_of(size[1], ny)
    inv = inverse_mass(nx, ny, "left", 0.25, dx, dy)
    rest = ref.copy()
    vel = np.zeros_like(ref)
    bits = np.zeros((ny, nx), dtype=np.int32)
    lengths = rest_lengths(dx, dy)
    comp = {kind: (5e-4 if "bend" in kind else 2e-5) for kind in KINDS}
    ref, _ = reference_steps(ref, vel, inv, rest, bits, 4, 2, comp, (0, -9.81, 0), 0.6, lengths)
    aligned, err, flipped = orient(got, ref)
    report["match"] = {"max_abs": err, "flipped": flipped, "finite": bool(gpu.get("finite")),
                       "shape": gpu.get("shape")}
    print("match", report["match"])

    def settle_strain(res: int, pin: str, frames: int, substeps: int = 8, width: float = 1.0):
        """Hang a sheet until it is quiet and report the worst structural edge strain.

        Long-range attachments (Kim, Chentanez, Müller 2012) plus stretch colours each substep
        are what keep a wide sheet inside 2%. The bound is that 2%, which is tighter than the
        stretch a viewer would call sag.
        """
        td.run_json(rf'''
import json
c = op('/sims/cloth')
c.par.Resx = {res}
c.par.Resy = {res}
c.par.Substeps = {substeps}
c.par.Selfiters = 0
c.par.Plane = 'xy'
c.par.Pin = '{pin}'
c.par.Originx = -0.5
c.par.Originy = 1.6
c.par.Originz = 0.0
c.par.Sizex = {width}
c.par.Sizey = {width}
c.par.Stretch = 1e-5
c.par.Shear = 1e-5
c.par.Bend = 1e-3
c.par.Damping = 4
c.par.Air = 0
c.par.Collide = False
c.par.Tear = 0
c.par.Thickness = 0
c.par.Gravityy = -9.81
c.par.Cookrender = False
c.PublishSphere(False)
c.SetUniformWind(0, 0, 0)
op.simclock.par.Sims = '/sims/cloth'
op.simclock.par.Fps = 60
op.simclock.par.Substeps = 1
op.simclock.RequestReset()
print('@@JSON@@{{"ok":1}}')
''', timeout=60)
        batch = 10
        for _ in range(max(1, frames // batch)):
            print(f" strain {res}", ticks(td, batch, timeout=240))
        stats = td.run_json(r'''
import json
import numpy as np
c = op('/sims/cloth')
pos = np.asarray(c.Positions(), dtype=np.float64)[..., :3]
vel = np.asarray(c.Velocities(), dtype=np.float64)[..., :3]
np.save(project.folder + '/reports/phase7_strain.npy', pos.astype(np.float32))
speed = np.linalg.norm(vel, axis=2)
print('@@JSON@@' + json.dumps({'max_speed': float(speed.max()),
    'finite': bool(np.isfinite(pos).all())}))
''', timeout=60)
        xyz = np.load(REPORTS / "phase7_strain.npy").astype(np.float64)
        dx = spacing_of(width, res)
        strains = edge_strains(xyz, rest_lengths(dx, dx), res, res)
        stats.update({
            "res": res,
            "pin": pin,
            "substeps": substeps,
            "width": width,
            "max_strain": float(strains.max()),
            "mean_strain": float(strains.mean()),
            "sag": float(xyz[..., 1].max() - xyz[..., 1].min()),
            "bound": 0.02,
        })
        return stats

    report["strain_64"] = settle_strain(64, "top", 40, 8, 1.0)
    print("strain64", report["strain_64"])
    report["strain_256"] = settle_strain(256, "top", 25, 8, 1.0)
    print("strain256", report["strain_256"])

    def rest_shape(substeps: int, frames: int = 24):
        width = 1.0
        td.run_json(rf'''
import json
c = op('/sims/cloth')
c.par.Substeps = {substeps}
c.par.Resx = 64
c.par.Resy = 64
c.par.Plane = 'xy'
c.par.Pin = 'top'
c.par.Originx = -0.5
c.par.Originy = 1.6
c.par.Originz = 0
c.par.Sizex = {width}
c.par.Sizey = {width}
c.par.Stretch = 1e-5
c.par.Shear = 1e-5
c.par.Bend = 1e-3
c.par.Damping = 3
c.par.Air = 0
c.par.Collide = False
c.par.Tear = 0
c.par.Thickness = 0
c.par.Gravityy = -9.81
op.simclock.par.Sims = '/sims/cloth'
op.simclock.RequestReset()
print('@@JSON@@{{"ok":1}}')
''', timeout=40)
        ticks(td, frames, timeout=240)
        td.run_json(r'''
import json
import numpy as np
np.save(project.folder + '/reports/phase7_rest.npy', np.asarray(op('/sims/cloth').Positions(), dtype=np.float32))
print('@@JSON@@{"ok":1}')
''', timeout=40)
        image = np.load(REPORTS / "phase7_rest.npy")
        return image[..., :3]

    low = rest_shape(8)
    high = rest_shape(16)
    width = 1.0
    report["timestep"] = {
        "max_abs": float(np.max(np.abs(low - high))),
        "width": width,
        "rel": float(np.max(np.abs(low - high)) / width),
        "substeps": [8, 16],
        "bound": 0.01,
    }
    print("timestep", report["timestep"])

    # Self-collision: two distant vertices start on top of each other and constraints are slack.
    td.run_json(r'''
import json
import numpy as np
c = op('/sims/cloth')
c.par.Resx = 8
c.par.Resy = 8
c.par.Substeps = 1
c.par.Selfiters = 4
c.par.Plane = 'xz'
c.par.Pin = 'none'
c.par.Originx = 0
c.par.Originy = 1
c.par.Originz = 0
c.par.Sizex = 0.35
c.par.Sizey = 0.35
c.par.Thickness = 0.04
c.par.Air = 0
c.par.Collide = False
c.par.Tear = 0
c.par.Damping = 0
c.par.Gravityx = 0
c.par.Gravityy = 0
c.par.Gravityz = 0
c.par.Stretch = 1e3
c.par.Shear = 1e3
c.par.Bend = 1e3
c.par.Hashres = 32
op.simclock.par.Sims = '/sims/cloth'
c.op('init_pos').cook(force=True)
pos = np.array(c.op('init_pos').numpyArray(delayed=False), dtype=np.float32, copy=True)
# Two texels that are not grid neighbours. The array is written back through SetInit,
# so these corners are the same texels the next step reads.
pos[0, 0, :3] = (0.0, 1.0, 0.0)
pos[-1, -1, :3] = (0.0, 1.0, 0.001)
root = c.parent.sims.par.Root.eval()
np.save(root + '/reports/phase7_self_pos.npy', pos)
np.save(root + '/reports/phase7_self_vel.npy', np.zeros_like(pos))
c.SetInit({'pos': 'reports/phase7_self_pos.npy', 'vel': 'reports/phase7_self_vel.npy'})
op.simclock.RequestReset()
print('@@JSON@@{"ok":1}')
''', timeout=40)
    ticks(td, 1, timeout=60)
    self_hit = td.run_json(r'''
import json
import numpy as np
pos = np.asarray(op('/sims/cloth').Positions(), dtype=np.float64)
a = pos[0, 0, :3]
b = pos[-1, -1, :3]
print('@@JSON@@' + json.dumps({'pair': float(np.linalg.norm(a - b)), 'finite': bool(np.isfinite(pos).all())}))
''', timeout=30)
    report["self_collision"] = self_hit
    print("self", self_hit)

    def wind_deflection(speed: float) -> dict:
        td.run_json(rf'''
import json
c = op('/sims/cloth')
c.par.Resx = 16
c.par.Resy = 16
c.par.Substeps = 4
c.par.Selfiters = 1
c.par.Plane = 'xy'
c.par.Pin = 'left'
c.par.Originx = -0.4
c.par.Originy = 1.2
c.par.Originz = 0.0
c.par.Sizex = 0.6
c.par.Sizey = 0.45
c.par.Air = 2.0
c.par.Drag = 0.6
c.par.Thickness = 0.01
c.par.Collide = False
c.par.Tear = 0
c.par.Damping = 0.4
c.par.Stretch = 2e-5
c.par.Shear = 2e-5
c.par.Bend = 4e-4
c.par.Gravityy = -9.81
c.SetUniformWind(0, 0, {speed})
op.simclock.par.Sims = '/sims/cloth'
op.simclock.RequestReset()
print('@@JSON@@{{"ok":1}}')
''', timeout=30)
        ticks(td, 25, timeout=150)
        return td.run_json(r'''
import json
import numpy as np
pos = np.asarray(op('/sims/cloth').Positions(), dtype=np.float64)
free = pos[:, 1:, :3]
print('@@JSON@@' + json.dumps({'mean_z': float(free[..., 2].mean()), 'mean_y': float(free[..., 1].mean())}))
''', timeout=30)

    slow = wind_deflection(1.5)
    fast = wind_deflection(5.0)
    report["wind"] = {"slow": slow, "fast": fast}
    print("wind", report["wind"])

    td.run_json(r'''
import json
c = op('/sims/cloth')
c.par.Resx = 16
c.par.Resy = 16
c.par.Substeps = 4
c.par.Plane = 'xz'
c.par.Pin = 'none'
c.par.Originx = -0.2
c.par.Originy = 0.62
c.par.Originz = -0.2
c.par.Sizex = 0.4
c.par.Sizey = 0.4
c.par.Collide = True
c.par.Air = 0
c.par.Tear = 0
c.par.Thickness = 0.01
c.par.Friction = 0.8
c.par.Damping = 1.0
c.par.Gravityy = -4
c.par.Spherx = 0
c.par.Sphery = 0.4
c.par.Spherez = 0
c.par.Spherer = 0.24
c.par.Sphervx = 0.8
c.par.Sphervy = 0
c.par.Sphervz = 0
c.SetUniformWind(0, 0, 0)
c.PublishSphere(True)
op.simclock.par.Sims = '/sims/cloth'
op.simclock.RequestReset()
print('@@JSON@@{"ok":1}')
''', timeout=30)
    pos0 = td.run_json(r'''
import json
import numpy as np
c = op('/sims/cloth')
op.simclock.Tick()
pos = np.asarray(c.Positions(), dtype=np.float64)
print('@@JSON@@' + json.dumps({'mean_x': float(pos[..., 0].mean())}))
''', timeout=60)
    ticks(td, 20, timeout=120)
    pos1 = td.run_json(r'''
import json
import numpy as np
pos = np.asarray(op('/sims/cloth').Positions(), dtype=np.float64)
print('@@JSON@@' + json.dumps({'mean_x': float(pos[..., 0].mean()), 'finite': bool(np.isfinite(pos).all())}))
''', timeout=30)
    report["moving"] = {"start_x": pos0["mean_x"], "end_x": pos1["mean_x"],
                        "delta": pos1["mean_x"] - pos0["mean_x"], "finite": pos1["finite"]}
    print("moving", report["moving"])

    # Drape: nothing inside the sphere, nothing below the ground.
    td.run_json(r'''
import json
c = op('/sims/cloth')
c.ApplyPreset('drape')
c.par.Cookrender = False
c.par.Sphervx = 0
c.PublishSphere(True)
c.SetUniformWind(0, 0, 0)
op.simclock.par.Sims = '/sims/cloth'
op.simclock.RequestReset()
print('@@JSON@@{"ok":1}')
''', timeout=30)
    for _ in range(6):
        print(" drape", ticks(td, 10, timeout=180))
    drape = td.run_json(r'''
import json
import numpy as np
c = op('/sims/cloth')
pos = np.asarray(c.Positions(), dtype=np.float64)[..., :3]
centre = np.array([float(c.par.Spherx), float(c.par.Sphery), float(c.par.Spherez)])
radius = float(c.par.Spherer)
thick = float(c.par.Thickness)
dist = np.linalg.norm(pos - centre, axis=2)
deep = int(np.count_nonzero(dist < radius - thick))
below = int(np.count_nonzero(pos[..., 1] < float(c.par.Ground) - 1e-4))
print('@@JSON@@' + json.dumps({'deep': deep, 'below': below, 'min_phi': float(dist.min() - radius),
    'min_y': float(pos[..., 1].min()), 'finite': bool(np.isfinite(pos).all())}))
''', timeout=40)
    report["drape"] = drape
    print("drape", drape)
    grab(td, "phase7_drape")

    # Determinism: two resets, same positions.
    td.run_json(r'''
import json
c = op('/sims/cloth')
c.par.Resx = 16
c.par.Resy = 16
c.par.Substeps = 2
c.par.Collide = True
c.par.Air = 0
c.par.Tear = 0
c.PublishSphere(True)
op.simclock.par.Sims = '/sims/cloth'
op.simclock.RequestReset()
print('@@JSON@@{"ok":1}')
''', timeout=30)
    ticks(td, 6, timeout=90)
    td.run_json(r'''
import json
import numpy as np
np.save(project.folder + '/reports/phase7_det_a.npy', np.asarray(op('/sims/cloth').Positions(), dtype=np.float32))
op.simclock.RequestReset()
print('@@JSON@@{"ok":1}')
''', timeout=30)
    ticks(td, 6, timeout=90)
    det = td.run_json(r'''
import json
import numpy as np
a = np.load(project.folder + '/reports/phase7_det_a.npy')
b = np.asarray(op('/sims/cloth').Positions(), dtype=np.float32)
print('@@JSON@@' + json.dumps({'max_abs': float(np.max(np.abs(a - b)))}))
''', timeout=30)
    report["determinism"] = det
    print("determinism", det)

    # Realtime scene, measured before the long bakes.
    td.run_json(CLEAN, timeout=60)
    td.run_json(r'''
import json
op('/sims/clock/tick').par.active = False
c = op('/sims/cloth')
c.ApplyPreset('realtime')
f = op('/sims/fluid3d')
for name in ('render_out', 'shadow'):
    f.op(name).bypass = False
c.op('bg_src').par.top = f.path + '/render_out'
op.simclock.par.Sims = '/sims/fluid3d /sims/cloth'
op.simclock.par.Running = True
op.simclock.par.Fps = 60
op.simclock.par.Substeps = 1
op('/sims/clock/tick').par.active = True
project.cookRate = 40
project.realTime = True
op.simclock.RequestReset()
op.simclock.Tick()
print('@@JSON@@' + json.dumps({'sims': str(op.simclock.par.Sims.eval())}))
''', timeout=90)
    report["realtime"] = timeline_rate(td, 3.0)
    report["realtime"]["scene"] = td.run_json(r'''
import json
print('@@JSON@@' + json.dumps(op('/sims/cloth').Scene()))
''', timeout=30)
    report["stages"] = td.run_json(r'''
import json
print('@@JSON@@' + json.dumps(op('/sims/cloth').StageTimes()))
''', timeout=60)
    print("realtime", report["realtime"])
    print("stages", report["stages"])
    grab(td, "phase7_flag")
    z0 = td.run_json(r'''
import json
import numpy as np
pos = np.asarray(op('/sims/cloth').Positions(), dtype=np.float64)[..., :3]
print('@@JSON@@' + json.dumps({'mean_z': float(pos[:, 1:, 2].mean())}))
''', timeout=30)
    time.sleep(0.5)
    grab(td, "phase7_flag_later")
    z1 = td.run_json(r'''
import json
import numpy as np
pos = np.asarray(op('/sims/cloth').Positions(), dtype=np.float64)[..., :3]
print('@@JSON@@' + json.dumps({'mean_z': float(pos[:, 1:, 2].mean()), 'pin_z': float(op('/sims/cloth').par.Originz.eval())}))
''', timeout=30)
    report["flutter"] = {
        "mean_z_0": z0["mean_z"],
        "mean_z_1": z1["mean_z"],
        "pin_z": z1["pin_z"],
        "picture_diff": float(np.mean(np.abs(
            np.load(REPORTS / "phase7_flag.npy")[..., :3] - np.load(REPORTS / "phase7_flag_later.npy")[..., :3]))),
    }
    print("flutter", report["flutter"])
    report["couple_ms"] = td.run_json(r'''
import json, time
c = op('/sims/cloth')
op('/sims/clock/tick').par.active = False
c.par.Couple = True
op.simclock.Tick()
node = c.op('cloth_force')
t0 = time.perf_counter()
node.cook(force=True)
node.numpyArray(delayed=False)
ms = (time.perf_counter() - t0) * 1000.0
c.par.Couple = False
c.PublishForce(False)
print('@@JSON@@' + json.dumps({'ms': ms}))
''', timeout=60)
    print("couple", report["couple_ms"])

    # Cancel / resume identity on a small grid.
    ident = ROOT / "cache" / "phase7_identity"
    ident_r = ROOT / "cache" / "phase7_identity_resume"
    for path in (ident, ident_r):
        if path.exists():
            shutil.rmtree(path)
    td.run_json(r'''
import json
op('/sims/clock/tick').par.active = False
c = op('/sims/cloth')
c.PublishSphere(False)
c.PublishForce(False)
c.SetUniformWind(0, 0, 0)
c.par.Resx = 16
c.par.Resy = 16
c.par.Substeps = 2
c.par.Selfiters = 1
c.par.Plane = 'xz'
c.par.Pin = 'corners'
c.par.Collide = False
c.par.Air = 0
c.par.Couple = False
c.par.Cookrender = False
c.par.Tear = 0
op.simclock.par.Sims = '/sims/cloth'
op.simclock.par.Fps = 60
op.simclock.par.Substeps = 1
b = op('/sims/baker')
b.par.Sim = '/sims/cloth'
b.par.Groupsims = '/sims/cloth'
b.par.Cachedir = 'cache/phase7_identity'
b.par.Framestart = 1
b.par.Frameend = 6
b.par.Warmup = 0
b.par.Resume = False
b.par.Keyinterval = 0
b.par.Playbackfields = ''
b.par.Crop = False
b.par.Asyncsave = False
b.Bake()
print('@@JSON@@' + json.dumps({'baking': b.ext.Baker.baking}))
''', timeout=40)
    pump(td, None, 2, 8)
    td.run_json(r'''
import json
b = op('/sims/baker')
b.par.Cachedir = 'cache/phase7_identity_resume'
b.par.Resume = False
b.par.Frameend = 6
b.Bake()
print('@@JSON@@' + json.dumps({'baking': b.ext.Baker.baking}))
''', timeout=40)
    pump(td, 3, 1, 8)
    td.run_json(r'''
import json
b = op('/sims/baker')
b.par.Resume = True
b.par.Frameend = 6
b.Bake()
print('@@JSON@@' + json.dumps({'baking': b.ext.Baker.baking, 'status': str(b.par.Status.eval())}))
''', timeout=40)
    pump(td, None, 2, 8)
    report["bake"] = compare_caches(ident, ident_r)
    print("bake", report["bake"])
    replay = td.run_json(r'''
import json
import numpy as np
c = op('/sims/cloth')
b = op('/sims/baker')
b.Cancel()
c.SetInit({'pos': 'cache/phase7_identity/cloth/pos/pos.00005.npy',
           'vel': 'cache/phase7_identity/cloth/vel/vel.00005.npy'})
op('/sims/clock/tick').par.active = False
op.simclock.par.Sims = '/sims/cloth'
op.simclock.par.Running = True
op.simclock.RequestReset()
op.simclock.Tick()
got = np.asarray(c.Positions(), dtype=np.float32)
want = np.load(project.folder + '/cache/phase7_identity/cloth/pos/pos.00006.npy')
diff = float(np.max(np.abs(got - want)))
diff_flip = float(np.max(np.abs(got[::-1] - want)))
print('@@JSON@@' + json.dumps({'pos_max_abs': min(diff, diff_flip)}))
''', timeout=60)
    report["replay"] = replay
    print("replay", replay)
    for path in (ident, ident_r):
        if path.exists():
            shutil.rmtree(path)

    show = ROOT / "cache" / "phase7_showcase"
    if show.exists():
        shutil.rmtree(show)
    td.run_json(r'''
import json
op('/sims/clock/tick').par.active = False
c = op('/sims/cloth')
c.ApplyPreset('showcase')
c.par.Cookrender = False
op.simclock.par.Sims = '/sims/cloth'
op.simclock.par.Substeps = 1
op.simclock.par.Fps = 60
b = op('/sims/baker')
b.par.Sim = '/sims/cloth'
b.par.Groupsims = '/sims/cloth'
b.par.Cachedir = 'cache/phase7_showcase'
b.par.Framestart = 1
b.par.Frameend = 180
b.par.Warmup = 0
b.par.Resume = False
b.par.Keyinterval = 15
b.par.Playbackfields = 'pos'
b.par.Crop = False
b.par.Asyncsave = False
b.Bake()
print('@@JSON@@' + json.dumps({'baking': b.ext.Baker.baking, 'status': str(b.par.Status.eval())}))
''', timeout=60)
    report["showcase_bake"] = pump(td, None, 2, 120)
    manifest_path = show / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    member = (manifest.get("members") or {}).get("cloth") or {}
    params = member.get("params") or manifest.get("params") or {}
    if isinstance(params, dict) and "cloth" in params and "vertices" not in params:
        params = params["cloth"]
    show_bytes = sum(p.stat().st_size for p in show.rglob("*") if p.is_file()) if show.exists() else 0
    saved = []
    for frame in (30, 90, 150):
        shot = td.run_json(rf'''
import json
import numpy as np
c = op('/sims/cloth')
reader = op('/sims/cache_reader')
reader.par.Cachedir = 'cache/phase7_showcase'
reader.par.Member = 'cloth'
reader.par.Interpolate = False
reader.par.Frame.expr = ''
reader.par.Frame = {frame}
reader.ext.CacheReader.Reload()
reader.ext.CacheReader.Load({frame})
picture = reader.Output('pos')
picture.cook(force=True)
mat = c.op('cloth_mat')
mat.par.sampler0top = picture.path
c.par.Cookrender = True
c.par.Camaz = 0.55
c.par.Camel = 0.28
c.par.Camdist = 2.8
c.par.Camfov = 46
c.par.Targetx = 0
c.par.Targety = 0.75
c.par.Targetz = 0
c.op('bg_src').par.top = c.op('plate').path
c.op('cloth_render').cook(force=True)
c.op('composite').cook(force=True)
image = np.asarray(c.op('picture_out').numpyArray(delayed=False), dtype=np.float32)
np.save(project.folder + '/reports/phase7_showcase_{frame}.npy', image)
loaded = np.asarray(picture.numpyArray(delayed=False), dtype=np.float32)
print('@@JSON@@' + json.dumps({{'frame': {frame}, 'mean': float(image[..., :3].mean()),
    'pos_finite': bool(np.isfinite(loaded).all())}}))
''', timeout=90)
        saved.append(shot)
        print(" showcase", shot)
    report["showcase_frames"] = saved
    pngs = []
    sheet_frames = []
    report["speckle_showcase"] = {}
    for frame in (30, 90, 150):
        npy = REPORTS / f"phase7_showcase_{frame}.npy"
        if npy.is_file():
            pngs.append(save_png(npy, REPORTS / f"phase7_showcase_{frame}.png"))
            sheet_frames.append(np.load(npy)[..., :3])
            report["speckle_showcase"][str(frame)] = speckle_fraction(npy)
            npy.unlink()
    if len(sheet_frames) == 3:
        sheet = np.concatenate(sheet_frames, axis=1)
        sheet_npy = REPORTS / "phase7_showcase_sheet.npy"
        np.save(sheet_npy, np.concatenate([sheet, np.ones(sheet.shape[:2] + (1,), np.float32)], axis=2))
        pngs.append(save_png(sheet_npy, REPORTS / "phase7_showcase_sheet.png"))
        sheet_npy.unlink()
    report["showcase"] = {
        "frames_written": int(manifest.get("frames_written") or 0),
        "vertices": int(params.get("vertices") or 0),
        "res": params.get("res"),
        "bytes": show_bytes,
        "complete": bool(manifest.get("complete")),
        "frames": [row["path"] for row in pngs],
    }
    print("showcase", report["showcase"])

    td.run_json(CLEAN, timeout=60)
    for npy_name in ("phase7_match.npy", "phase7_hang.npy", "phase7_rest.npy", "phase7_det_a.npy"):
        path = REPORTS / npy_name
        if path.is_file():
            path.unlink()
    flag_png = save_png(REPORTS / "phase7_flag.npy", REPORTS / "phase7_flag.png")
    drape_png = save_png(REPORTS / "phase7_drape.npy", REPORTS / "phase7_drape.png")
    report["flag_png"] = flag_png
    report["drape_png"] = drape_png
    report["png_diff"] = float(np.mean(np.abs(
        np.load(REPORTS / "phase7_flag.npy")[..., :3] - np.load(REPORTS / "phase7_drape.npy")[..., :3])))
    report["speckle"] = {
        "flag": speckle_fraction(REPORTS / "phase7_flag.npy"),
        "flag_later": speckle_fraction(REPORTS / "phase7_flag_later.npy"),
        "drape": speckle_fraction(REPORTS / "phase7_drape.npy"),
    }
    for npy in (REPORTS / "phase7_flag.npy", REPORTS / "phase7_flag_later.npy", REPORTS / "phase7_drape.npy"):
        if npy.is_file():
            npy.unlink()

    def script(*args: str) -> int:
        return subprocess.run([sys.executable, *args], cwd=ROOT).returncode

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
    report["phase5"] = fresh(str(ROOT / "scripts" / "validate_phase5.py"), "--quick")
    print("phase5", report["phase5"])
    report["phase6"] = fresh(str(ROOT / "scripts" / "validate_phase6.py"), "--quick")
    print("phase6", report["phase6"])
    report["shaders"] = script(str(ROOT / "scripts" / "check_shaders.py"))
    report["pytest"] = script("-m", "pytest", "-q")
    td = attach()
    td.run_json(CLEAN, timeout=60)
    drop_validation_caches()
    report["cache_bytes_end"] = cache_bytes()

    scene = (report["realtime"].get("scene") or {})
    bypass = scene.get("bypass") or {}
    fluid_bypass = scene.get("fluid_bypass") or {}
    rate = report["realtime"]
    show = report["showcase"]
    wind = report["wind"]
    ok = (
        not gpu.get("errors")
        and report["match"]["finite"]
        and report["match"]["max_abs"] < 2e-3
        and report["strain_64"]["finite"]
        and report["strain_64"]["max_speed"] < 0.25
        and report["strain_64"]["max_strain"] <= 0.02
        and report["strain_256"]["finite"]
        and report["strain_256"]["max_speed"] < 0.35
        and report["strain_256"]["max_strain"] <= 0.02
        and report["timestep"]["rel"] <= 0.01
        and self_hit["finite"]
        and self_hit["pair"] > 0.5 * 0.04
        and wind["slow"]["mean_z"] > 0.005
        and wind["fast"]["mean_z"] > wind["slow"]["mean_z"] + 0.01
        and report["moving"]["finite"]
        and report["moving"]["delta"] > 0.02
        and drape["finite"]
        and drape["deep"] == 0
        and drape["below"] == 0
        and drape["min_y"] >= -1e-4
        and det["max_abs"] == 0.0
        and rate["steps"] == rate["frames"]
        and rate["steps"] > 0
        and rate["fps"] >= 33
        and scene.get("cookrender") is True
        and scene.get("fluid_cookrender") is True
        and scene.get("vertices", 0) >= 4096
        and scene.get("iterations") == 1
        and scene.get("substeps", 0) >= 8
        and report["flutter"]["picture_diff"] >= 1e-3
        and report["flutter"]["mean_z_1"] > report["flutter"]["pin_z"] + 0.02
        and all(row["fraction"] <= 0.05 and row["cloth_pixels"] > 200 for row in report["speckle"].values())
        and all(row["fraction"] <= 0.05 and row["cloth_pixels"] > 200 for row in report["speckle_showcase"].values())
        and scene.get("fluid_res") == [48, 48, 48]
        and scene.get("fluid_solver") == "jacobi"
        and "fluid3d" in str(scene.get("sims"))
        and "cloth" in str(scene.get("sims"))
        and bypass
        and all(v is False for v in bypass.values())
        and fluid_bypass.get("render_out") is False
        and fluid_bypass.get("jacobi") is False
        and not scene.get("errors")
        and report["bake"]["identical"] is True
        and report["replay"]["pos_max_abs"] < 1e-5
        and int(show.get("frames_written") or 0) >= 180
        and int(show.get("vertices") or 0) >= 128 * 128
        and show["bytes"] <= SHOWCASE_BUDGET
        and show.get("complete") is True
        and not flag_png["checker"]
        and not drape_png["checker"]
        and flag_png["min_alpha"] >= 0.99
        and drape_png["min_alpha"] >= 0.99
        and flag_png["std_luma"] > 0.02
        and drape_png["std_luma"] > 0.02
        and flag_png["blown"] < 0.05
        and drape_png["blown"] < 0.02
        and report["png_diff"] >= 1e-3
        and len(pngs) == 4
        and all(not row["checker"] for row in pngs)
        and not abs_path_strings(report)
        and report["bake_check"] == 0
        and report["phase2"] == 0
        and report["phase3"] == 0
        and report["phase4"] == 0
        and report["phase5"] == 0
        and report["phase6"] == 0
        and report["shaders"] == 0
        and report["pytest"] == 0
        and report["cache_bytes_end"] <= CACHE_BUDGET
    )
    report["pass"] = bool(ok)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "phase7.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
