"""Phase 3 checks. Exit 0 is pass. Writes reports/phase3.json and reports/phase3_*.png.

Touches only the scratch TouchDesigner from td_scratch.py. Long cooks are split into short calls.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from td_scratch import ROOT, attach  # noqa: E402

ADVECT = r'''
import json
p = op('/sims/particles')
c = op.simclock
op('/sims/clock/tick').par.active = False
p.Unbind()
p.SetInit(None)
p.par.Bus = '2d'
p.par.Velmode = 'constant'
p.par.Emitter = 'tracer'
p.par.Drag = 0
p.par.Gravity = 0
p.par.Buoyancy = 0
p.par.Turbulence = 0
p.par.Radius = -1
p.par.Couplingstrength = 0
p.par.Srcheat = 0
p.par.Cookrender = False
p.par.Resx = 64
p.par.Resy = 64
p.par.Maxspeed = 30
p.par.Analyticx = 1
p.par.Analyticy = 0.25
p.par.Analyticz = 0
p.par.Spawnx = -0.4
p.par.Spawny = 0.6
p.par.Spawnz = 0
p.par.Spawnvx = 1
p.par.Spawnvy = 0.25
p.par.Spawnvz = 0
c.par.Sims = '/sims/particles'
c.par.Fps = 60
c.par.Substeps = 1
c.par.Startframe = 0
c.par.Mode = 'realtime'
c.par.Running = True
project.realTime = False
n = 30
c.RequestReset()
for _ in range(n):
    c.Tick()
pos = p.op('pos_out').numpyArray(delayed=False)
vel = p.op('pvel_out').numpyArray(delayed=False)
dt = 1.0 / 60.0
got = [float(pos[0, 0, i]) for i in range(3)]
expect = [-0.4 + n * dt * 1.0, 0.6 + n * dt * 0.25, 0.0]
err = max(abs(got[i] - expect[i]) for i in range(3))
others = float(abs(vel[0, 1, 3])) + float(abs(vel[1, 0, 3]))
print('@@JSON@@' + json.dumps({'err': err, 'got': got, 'expect': expect, 'others_life': others,
    'finite': bool(__import__('numpy').isfinite(pos).all())}))
'''

COLLIDE2 = r'''
import json, numpy as np
p = op('/sims/particles')
c = op.simclock
d = op('/sims/demo')
op('/sims/clock/tick').par.active = False
d.par.Discamplitude = 0
p.Unbind()
p.SetInit(None)
p.par.Bus = '2d'
p.par.Velmode = 'bus'
p.par.Emitter = 'box'
p.par.Drag = 0
p.par.Gravity = 0
p.par.Buoyancy = 0
p.par.Turbulence = 0
p.par.Radius = 0
p.par.Restitution = 0.45
p.par.Friction = 0.15
p.par.Couplingstrength = 0
p.par.Srcheat = 0
p.par.Cookrender = False
p.par.Resx = 48
p.par.Resy = 48
p.par.Emitx = 0
p.par.Emity = 0.42
p.par.Emitz = 0
p.par.Boxhx = 0.05
p.par.Boxhy = 0.05
p.par.Boxhz = 0.05
p.par.Emitvelx = 0
p.par.Emitvely = 3.2
p.par.Emitvelz = 0
p.par.Emitvar = 0
p.par.Lifemin = 8
p.par.Lifemax = 8
c.par.Sims = '/sims/particles'
c.par.Running = True
project.realTime = False
c.RequestReset()
for _ in range(18):
    c.Tick()
pos = p.op('pos_out').numpyArray(delayed=False)
vel = p.op('pvel_out').numpyArray(delayed=False)
cy = float(d.par.Discy)
rad = float(d.par.Discradius)
cell = (float(op.simbus.par.Boundsx1) - float(op.simbus.par.Boundsx0)) / float(op.simbus.par.Resw)
inside = 0
alive = 0
min_sdf = 1e9
for y in range(pos.shape[0]):
    for x in range(pos.shape[1]):
        if vel[y, x, 3] <= 0:
            continue
        alive += 1
        dx = float(pos[y, x, 0]) - 0.0
        dy = float(pos[y, x, 1]) - cy
        sd = (dx * dx + dy * dy) ** 0.5 - rad
        min_sdf = min(min_sdf, sd)
        if sd < -cell:
            inside += 1
print('@@JSON@@' + json.dumps({'inside': inside, 'alive': alive, 'min_sdf': min_sdf, 'cell': cell}))
'''

COLLIDE3 = r'''
import json, numpy as np
p = op('/sims/particles')
c = op.simclock
d = op('/sims/demo3d')
op('/sims/clock/tick').par.active = False
d.par.Sphereamp = 0
p.Unbind()
p.SetInit(None)
p.par.Bus = '3d'
p.par.Velmode = 'bus'
p.par.Emitter = 'box'
p.par.Drag = 0
p.par.Gravity = 0
p.par.Buoyancy = 0
p.par.Turbulence = 0
p.par.Radius = 0
p.par.Restitution = 0.4
p.par.Friction = 0.2
p.par.Couplingstrength = 0
p.par.Srcheat = 0
p.par.Cookrender = False
p.par.Resx = 32
p.par.Resy = 32
p.par.Emitx = 0
p.par.Emity = 0.28
p.par.Emitz = 0
p.par.Boxhx = 0.04
p.par.Boxhy = 0.04
p.par.Boxhz = 0.04
p.par.Emitvelx = 0
p.par.Emitvely = 2.4
p.par.Emitvelz = 0
p.par.Emitvar = 0
p.par.Lifemin = 8
p.par.Lifemax = 8
c.par.Sims = '/sims/particles'
c.par.Running = True
project.realTime = False
c.RequestReset()
for _ in range(16):
    c.Tick()
pos = p.op('pos_out').numpyArray(delayed=False)
vel = p.op('pvel_out').numpyArray(delayed=False)
cx, cy, cz = float(d.par.Spherex), float(d.par.Spherey), float(d.par.Spherez)
rad = float(d.par.Sphereradius)
cell = (float(op.simbus3d.par.Boundsx1) - float(op.simbus3d.par.Boundsx0)) / float(op.simbus3d.par.Resw)
inside = 0
alive = 0
min_sdf = 1e9
for y in range(pos.shape[0]):
    for x in range(pos.shape[1]):
        if vel[y, x, 3] <= 0:
            continue
        alive += 1
        dx = float(pos[y, x, 0]) - cx
        dy = float(pos[y, x, 1]) - cy
        dz = float(pos[y, x, 2]) - cz
        sd = (dx * dx + dy * dy + dz * dz) ** 0.5 - rad
        min_sdf = min(min_sdf, sd)
        if sd < -cell:
            inside += 1
print('@@JSON@@' + json.dumps({'inside': inside, 'alive': alive, 'min_sdf': min_sdf, 'cell': cell,
    'finite': bool(np.isfinite(pos).all())}))
'''

COUPLE_ON = r'''
import json, numpy as np
p = op('/sims/particles')
f = op('/sims/fluid2d')
c = op.simclock
op('/sims/clock/tick').par.active = False
op.simbus.SetGrid(384)
f.par.Jacobi = 8
f.par.Cookrender = False
f.par.Vorticity = 0.4
p.par.Bus = '2d'
p.par.Velmode = 'bus'
p.par.Emitter = 'sphere'
p.par.Resx = 128
p.par.Resy = 128
p.par.Emitx = 0
p.par.Emity = 0.35
p.par.Emitz = 0
p.par.Emitradius = 0.18
p.par.Emitvelx = 0
p.par.Emitvely = 0
p.par.Emitvelz = 0
p.par.Emitvar = 0.2
p.par.Drag = 8
p.par.Gravity = 0
p.par.Buoyancy = 0
p.par.Turbulence = 0
p.par.Radius = -1
p.par.Lifemin = 6
p.par.Lifemax = 6
p.par.Couplingstrength = 1
p.par.Particlemass = 1
p.par.Density = 1
p.par.Srcheat = 0
p.par.Cookrender = False
c.par.Sims = '/sims/particles /sims/fluid2d'
c.par.Fps = 60
c.par.Substeps = 1
c.par.Running = True
c.par.Seed = 1
project.realTime = False
print('@@JSON@@' + json.dumps(1))
'''

# The off-run is a second call. The on-run stores nothing; we compare by rerunning and diffing in Python
# after both numpy arrays are fetched. Fetching full 384^2 is fine.

def _vel_code(strength: float, tag: str) -> str:
    return f'''
import json, numpy as np
p = op('/sims/particles')
f = op('/sims/fluid2d')
c = op.simclock
op('/sims/clock/tick').par.active = False
p.SetInit(None)
p.par.Couplingstrength = {strength}
p.par.Particlemass = 1
p.par.Density = 1
p.par.Srcheat = 0
if {strength} == 0:
    p.Unbind()
else:
    p.Bind()
c.par.Sims = '/sims/particles /sims/fluid2d'
c.par.Running = True
c.par.Seed = 1
project.realTime = False
c.RequestReset()
peak = 0.0
for _ in range(24):
    c.Tick()
    spun = op('/sims/particles/forces_2d').numpyArray(delayed=False)
    if spun is not None:
        peak = max(peak, float(np.max(np.abs(spun))))
vel = f.op('vel_state').numpyArray(delayed=False).astype('float32')
np.save(project.folder + '/reports/phase3_vel_{tag}.npy', vel)
print('@@JSON@@' + json.dumps({{'finite': bool(np.isfinite(vel).all()), 'mean': float(np.mean(np.abs(vel))),
    'max': float(np.max(np.abs(vel))), 'force_max': peak}}))
'''

RENDER2 = r'''
import json
op('/sims/demo_particles').ApplyDemo('embers2d')
p = op('/sims/particles')
c = op.simclock
op('/sims/clock/tick').par.active = False
# Keep the demo look but a shorter particle count would hide the 1M path; the demo is 1024^2.
p.par.Resx = 256
p.par.Resy = 256
p.par.Pointsize = 8
c.par.Running = True
project.realTime = False
c.RequestReset()
for _ in range(140):
    c.Tick()
    op('/sims/fluid2d').Snapshot()
print('@@JSON@@' + json.dumps({'err': p.op('beauty').errors() or '', 'step': int(c.par.Step)}))
'''

# The cache replace commits when the ticking script returns, so the picture is saved
# in a second call. Saving in the same call writes the previous TouchDesigner frame.

RENDER3 = r'''
import json
op('/sims/demo_particles').ApplyDemo('sparks3d')
p = op('/sims/particles')
f = op('/sims/fluid3d')
c = op.simclock
op('/sims/clock/tick').par.active = False
f.par.Jacobi = 8
f.par.Cookrender = True
p.par.Resx = 128
p.par.Resy = 128
c.par.Running = True
project.realTime = False
c.RequestReset()
for _ in range(64):
    c.Tick()
    if hasattr(op('/sims/fluid3d'), 'Snapshot'):
        op('/sims/fluid3d').Snapshot()
print('@@JSON@@' + json.dumps({'err': (p.op('beauty').errors() or '') + (f.op('render').errors() or ''),
    'step': int(c.par.Step)}))
'''

def _save_picture(filename: str) -> str:
    return f'''
import json, numpy as np
p = op('/sims/particles')
def _stats(top):
    arr = top.numpyArray(delayed=False)
    rgb = arr[:, :, :3]
    lum = rgb.mean(axis=2)
    return {{'lit': float((lum > 0.05).mean()), 'sat': float((rgb.max(axis=2) >= 0.999).mean()),
             'alpha_min': float(arr[:, :, 3].min()), 'max': float(rgb.max())}}
p.op('picture_out').cook(force=True)
p.op('picture_out').save(project.folder + '/reports/{filename}')
plume = None
if p.par.Bus.eval() == '2d':
    beauty = p.op('beauty').numpyArray(delayed=False)
    sc = op('/sims/fluid2d/scalars_out').numpyArray(delayed=False)
    lum = beauty[:, :, :3].mean(axis=2)
    ys, xs = np.where(lum > 0.05)
    y0 = float(op.simbus.par.Boundsy0)
    y1 = float(op.simbus.par.Boundsy1)
    h = max(lum.shape[0] - 1, 1)
    world_y = y0 + (float(ys.mean()) / h) * (y1 - y0) if len(ys) else y0
    hot = (sc[:, :, 0] > 0.02) | (sc[:, :, 1] > 0.02)
    dil = hot.copy()
    for shift in range(-6, 7, 2):
        dil = np.logical_or(dil, np.roll(hot, shift, axis=0))
        dil = np.logical_or(dil, np.roll(hot, shift, axis=1))
    if sc.shape[0] != lum.shape[0] or sc.shape[1] != lum.shape[1]:
        sy = (ys.astype('float64') / h * (sc.shape[0] - 1)).astype('int32')
        sx = (xs.astype('float64') / max(lum.shape[1] - 1, 1) * (sc.shape[1] - 1)).astype('int32')
        inside = float(dil[sy, sx].mean()) if len(ys) else 0.0
    else:
        inside = float(dil[ys, xs].mean()) if len(ys) else 0.0
    plume = {{'centroid_y': world_y, 'emitter_y': float(p.par.Emity), 'inside': inside, 'lit': int(len(ys))}}
print('@@JSON@@' + json.dumps({{'err': p.op('beauty').errors() or '', 'step': int(op.simclock.par.Step),
    'beauty': _stats(p.op('beauty')), 'picture': _stats(p.op('picture_out')),
    'fluid': _stats(p.op('fluid_fit')), 'plume': plume}}))
'''

FPS_PREP = r'''
import json
p = op('/sims/particles')
c = op.simclock
op('/sims/clock/tick').par.active = False
p.Unbind()
p.SetInit(None)
p.par.Bus = {bus!r}
p.par.Velmode = 'bus'
p.par.Emitter = 'sphere'
p.par.Drag = 2
p.par.Gravity = 0.2
p.par.Turbulence = 0.1
p.par.Couplingstrength = {coupling}
p.par.Srcheat = 0
p.par.Cookrender = {render}
p.par.Pointsize = 4
p.par.Resx = {rx}
p.par.Resy = {ry}
p.par.Radius = -1
if {coupling} != 0:
    p.Bind()
c.par.Fps = 60
c.par.Substeps = 1
c.par.Startframe = 0
c.par.Mode = 'realtime'
c.par.Running = True
c.par.Sims = {sims!r}
project.realTime = False
if {fluid3d}:
    f = op('/sims/fluid3d')
    f.allowCooking = True
    f.par.Solver = 'jacobi'
    f.par.Jacobi = 12
    f.par.Cookrender = True
    f.par.Renderw = 1280
    f.par.Renderh = 720
if {fluid2d}:
    op('/sims/fluid2d').allowCooking = True
    op('/sims/fluid2d').par.Cookrender = True
    op('/sims/fluid2d').ApplyPreset('realtime')
# A fluid that is not in this rate sample must not cook. After a 256³ bake its network is still
# dirty, and pulling it drops the particle rate under 30 fps.
op('/sims/fluid3d').allowCooking = bool({fluid3d})
op('/sims/fluid2d').allowCooking = bool({fluid2d})
c.RequestReset()
c.Tick()
print('@@JSON@@' + json.dumps({{'step': int(c.par.Step)}}))
'''

FPS_ARM = r'''
import json
op('/sims/clock/tick').par.active = True
op.simclock.par.Running = True
project.realTime = False
print('@@JSON@@' + json.dumps({'step': int(op.simclock.par.Step), 'frame': int(absTime.frame)}))
'''

FPS_READ = "import json\nprint('@@JSON@@' + json.dumps({'step': int(op.simclock.par.Step), 'frame': int(absTime.frame)}))"

BAKE_SETUP = r'''
import json
p = op('/sims/particles')
b = op('/sims/baker')
c = op.simclock
op('/sims/clock/tick').par.active = False
p.Unbind()
p.SetInit(None)
p.par.Bus = '2d'
p.par.Velmode = 'constant'
p.par.Emitter = 'sphere'
p.par.Drag = 0
p.par.Gravity = 0
p.par.Buoyancy = 0
p.par.Turbulence = 0
p.par.Radius = -1
p.par.Couplingstrength = 0
p.par.Srcheat = 0
p.par.Cookrender = False
p.par.Resx = 256
p.par.Resy = 256
p.par.Emitvar = 0
p.par.Emitvelx = 0.4
p.par.Emitvely = 0.1
p.par.Emitvelz = 0
p.par.Analyticx = 0.4
p.par.Analyticy = 0.1
p.par.Analyticz = 0
p.par.Lifemin = 20
p.par.Lifemax = 20
c.par.Sims = '/sims/particles'
c.par.Fps = 60
c.par.Substeps = 1
c.par.Running = True
b.par.Sim = '/sims/particles'
b.par.Cachedir = {cache!r}
b.par.Framestart = 1
b.par.Frameend = {frames}
b.par.Warmup = 0
b.par.Resume = True
cleared = b.ClearCache() if {clear} else 0
print('@@JSON@@' + json.dumps(cleared))
'''

BAKE = "op('/sims/baker').Bake()\nprint('@@JSON@@' + __import__('json').dumps(str(op('/sims/baker').par.Status.eval())))"
CANCEL = "op('/sims/baker').Cancel()\nprint('@@JSON@@' + __import__('json').dumps(str(op('/sims/baker').par.Status.eval())))"

PUMP = r'''
import json
b = op('/sims/baker')
c = op.simclock
op('/sims/clock/tick').par.active = False
c.par.Running = True
n = 0
for _ in range({n}):
    if not b.ext.Baker.baking:
        break
    c.Tick()
    n += 1
replay = None
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
    for field in ('pos', 'pvel'):
        live = op('/sims/particles/' + field + '_out').numpyArray(delayed=False)
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

RESTORE = r'''
import json
p = op('/sims/particles')
p.Unbind()
p.par.Couplingstrength = 0
p.par.Srcheat = 0
p.par.Cookrender = False
p.SetInit(None)
for name in ('step_2d_pos', 'step_2d_vel', 'step_3d_pos', 'step_3d_vel', 'deposit_2d', 'deposit_3d',
             'splat_render', 'splat_count', 'splat_scan', 'splat_base', 'splat_scatter', 'splat_sum',
             'splat_part', 'splat_fence', 'splat_vol', 'forces_3d', 'sources_3d', 'beauty_render', 'beauty', 'emissive'):
    node = p.op(name)
    if node is not None:
        node.bypass = True
op('/sims/demo').par.Discamplitude = 0.45
op('/sims/demo3d').par.Sphereamp = 0.1
op('/sims/clock/tick').par.active = True
op.simclock.par.Sims = '/sims/fluid2d'
op.simclock.par.Mode = 'realtime'
op.simclock.par.Running = True
op('/sims/fluid2d').allowCooking = True
op('/sims/fluid2d').par.Cookrender = False
op('/sims/fluid2d').par.Vorticity = 2.5
op('/sims/fluid2d').ApplyPreset('realtime')
op('/sims/fluid3d').allowCooking = True
op('/sims/demo').allowCooking = True
op('/sims/demo3d').allowCooking = True
op.simclock.par.Seed = 0
op('/sims/baker').par.Sim = '/sims/fluid2d'
op('/sims/baker').par.Groupsims = ''
op('/sims/baker').par.Cachedir = 'cache/fluid2d'
reader = op('/sims/cache_reader')
reader.par.Member = ''
reader.par.Cachedir = 'cache/fluid2d'
for movie in reader.findChildren(type=moviefileinTOP):
    movie.par.playmode = 'locked'
    movie.par.outputresolution = 'useinput'
reader.par.Frame.expr = ''
reader.par.Interpolate = True
reader.par.Frame.expr = 'op.simclock.par.Frame'
project.realTime = True
print('@@JSON@@' + json.dumps(op.simclock.par.Sims.eval()))
'''


def wait_bake(td, cache, timeout=180, cancel_at=None, replay=False):
    t0 = time.time()
    state = {"baking": True, "frame": 0}

    def pump(n, do_replay):
        return td.run_json(PUMP.format(n=n, replay="True" if do_replay else "False", cache=cache), timeout=90)

    while time.time() - t0 < timeout:
        state = pump(4, replay)
        if cancel_at is not None and state["baking"] and state["frame"] >= cancel_at:
            td.run_json(CANCEL, timeout=20)
            return pump(0, False)
        if not state["baking"]:
            return state
        if state.get("pumped") == 0 and state["baking"]:
            raise SystemExit("particle bake stopped advancing: " + json.dumps(state))
    raise SystemExit("particle bake did not finish")


def files(cache: Path):
    return sorted(p for p in cache.rglob("*") if p.is_file())


def measure_fps(td, seconds=3.0):
    start = td.run_json(FPS_ARM, timeout=30)
    time.sleep(seconds)
    end = td.run_json(FPS_READ, timeout=30)
    td.run_json("op('/sims/clock/tick').par.active = False\nprint('@@JSON@@' + __import__('json').dumps(1))", timeout=20)
    steps = end["step"] - start["step"]
    frames = end["frame"] - start["frame"]
    return {"steps": steps, "frames": frames, "fps": round(steps / seconds, 2),
            "steps_match_frames": abs(steps - frames) <= 1}


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def read_png(path: Path):
    """8-bit RGB or RGBA PNG, stdlib only. Returns (height, width, channels, rows of bytes)."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path.name} is not a PNG")
    pos, width, height, color, idat = 8, 0, 0, 0, []
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bit, color = struct.unpack(">IIBB", chunk[:10])
            if bit != 8 or color not in (2, 6):
                raise ValueError(f"{path.name} is not 8-bit RGB(A)")
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"IEND":
            break
    channels = 4 if color == 6 else 3
    raw = zlib.decompress(b"".join(idat))
    stride = width * channels
    rows = []
    i = 0
    prev = bytearray(stride)
    for _y in range(height):
        filt = raw[i]
        i += 1
        row = bytearray(raw[i:i + stride])
        i += stride
        for x in range(stride):
            left = row[x - channels] if x >= channels else 0
            up = prev[x]
            ul = prev[x - channels] if x >= channels else 0
            if filt == 1:
                row[x] = (row[x] + left) & 255
            elif filt == 2:
                row[x] = (row[x] + up) & 255
            elif filt == 3:
                row[x] = (row[x] + ((left + up) // 2)) & 255
            elif filt == 4:
                row[x] = (row[x] + _paeth(left, up, ul)) & 255
        prev = row
        rows.append(row)
    return height, width, channels, rows


def png_stats(path: Path) -> dict:
    height, width, channels, rows = read_png(path)
    pixels = height * width
    lit = sat = alpha_bad = 0
    for row in rows:
        for x in range(width):
            i = x * channels
            r, g, b = row[i], row[i + 1], row[i + 2]
            if (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0 > 0.05:
                lit += 1
            if r >= 255 or g >= 255 or b >= 255:
                sat += 1
            if channels < 4 or row[i + 3] != 255:
                alpha_bad += 1
    return {
        "lit": lit / pixels,
        "sat": sat / pixels,
        "alpha_one": alpha_bad == 0,
        "pixels": pixels,
    }


def _picture_ok(stats: dict) -> bool:
    return stats.get("lit", 0) > 0.001 and stats.get("sat", 1) < 0.5 and stats.get("alpha_one", False)


GROUP_SETUP = r'''
import json
p = op('/sims/particles')
f = op('/sims/fluid2d')
b = op('/sims/baker')
c = op.simclock
f.allowCooking = True
f.ApplyPreset('realtime')
f.par.Vorticity = 2.5
f.par.Cookrender = False
p.Unbind()
p.par.Bus = '2d'
p.par.Velmode = 'bus'
p.par.Emitter = 'fire'
p.par.Resx = 256
p.par.Resy = 256
p.par.Drag = 3.5
p.par.Gravity = 0.2
p.par.Buoyancy = 0.4
p.par.Turbulence = 0.2
p.par.Radius = 0
p.par.Couplingstrength = 1
p.par.Particlemass = 1
p.par.Density = 1
p.par.Srcheat = 0.02
p.par.Cookrender = False
p.par.Lifemin = 0.6
p.par.Lifemax = 1.2
p.par.Threshold = 0.05
p.par.Emitx = 0
p.par.Emity = 0.22
p.par.Emitz = 0
p.par.Emitradius = 0.14
p.par.Emitvar = 0.1
p.Bind()
c.par.Seed = 0
c.par.Fps = 60
c.par.Substeps = 1
c.par.Mode = 'realtime'
c.par.Running = True
b.par.Sim = '/sims/fluid2d'
b.par.Groupsims = '/sims/fluid2d /sims/particles'
b.par.Cachedir = {cache!r}
b.par.Framestart = 1
b.par.Frameend = 40
b.par.Warmup = 0
b.par.Resume = True
b.par.Pauseafter = True
cleared = b.ClearCache() if {clear} else 0
op('/sims/clock/tick').par.active = False
print('@@JSON@@' + json.dumps(cleared))
'''

GROUP_REPLAY = r'''
import json
from pathlib import Path
r = op('/sims/cache_reader')
r.par.Cachedir = {cache!r}
r.par.Interpolate = False
r.par.Frame.expr = ''
end = 40

def diff(node, field):
    # numpy of a velocity snapshot can sit one quantize bin off the EXR save() just wrote.
    # Byte-compare a fresh save of the live TOP with the cache file instead.
    r.par.Frame = end
    src = r.Path(field, 0)
    if not src or node is None:
        return None
    tmp = src + '.replay.exr'
    node.save(tmp, asynchronous=False)
    same = Path(tmp).read_bytes() == Path(src).read_bytes()
    Path(tmp).unlink(missing_ok=True)
    return 0.0 if same else 1.0

# Particle snapshots first. Reading a fluid output can pull the solver and step the particles.
live = {{
    'pos': op('/sims/particles/pos_out'),
    'pvel': op('/sims/particles/pvel_out'),
    'vel': op('/sims/fluid2d/vel_out'),
    'scalars': op('/sims/fluid2d/scalars_out'),
    'pressure': op('/sims/fluid2d/pressure_out'),
}}
r.par.Member = 'fluid2d'
r.Reload()
fluid = {{
    'vel': diff(live['vel'], 'vel'),
    'scalars': diff(live['scalars'], 'scalars'),
    'pressure': diff(live['pressure'], 'pressure'),
}}
r.par.Member = 'particles'
r.Reload()
parts = {{
    'pos': diff(live['pos'], 'pos'),
    'pvel': diff(live['pvel'], 'pvel'),
}}
r.par.Member = ''
r.par.Frame.expr = 'op.simclock.par.Frame'
print('@@JSON@@' + json.dumps({{'fluid': fluid, 'particles': parts}}))
'''

GROUP_STALE = r'''
import json
p = op('/sims/particles')
b = op('/sims/baker')
before = b.CacheStatus()
old = float(p.par.Drag)
p.par.Drag = old + 0.25
changed = b.CacheStatus()
p.par.Drag = old
print('@@JSON@@' + json.dumps({'before': before, 'changed': changed, 'restored': b.CacheStatus()}))
'''


TIMELINE_STATUS = r'''
import json
b = op('/sims/baker')
print('@@JSON@@' + json.dumps({'status': str(b.par.Status.eval()), 'baking': bool(b.ext.Baker.baking),
    'frame': int(op.simclock.par.Frame.eval())}))
'''


def _wait_timeline_bake(td, cancel_at=None, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        state = td.run_json(r'''
import json
b = op('/sims/baker')
baking = b.PumpOnce() if b.ext.Baker.baking else False
print('@@JSON@@' + json.dumps({'status': str(b.par.Status.eval()), 'baking': bool(baking),
    'frame': int(op.simclock.par.Frame.eval())}))
''', timeout=60)
        if cancel_at is not None and state["baking"] and state["frame"] >= cancel_at:
            td.run_json(CANCEL)
            return td.run_json(TIMELINE_STATUS)
        if not state["baking"]:
            return state
    raise SystemExit("group bake did not finish")


def _cache_max_diff(a: Path, b: Path) -> float:
    import numpy as np
    import OpenEXR

    def pixels(path):
        if path.suffix == ".npy":
            return {"": np.load(path).astype(np.float64)}
        with OpenEXR.File(str(path)) as f:
            return {k: np.asarray(c.pixels, dtype=np.float64) for k, c in f.channels().items()}

    pa, pb = pixels(a), pixels(b)
    if pa.keys() != pb.keys() or any(pa[k].shape != pb[k].shape for k in pa):
        return float("inf")
    return max(float(np.nan_to_num(np.abs(pa[k] - pb[k]), nan=np.inf).max()) for k in pa)


def group_bake(td) -> dict:
    full, resume = "cache/phase3_group", "cache/phase3_group_resume"
    td.run_json(GROUP_SETUP.format(cache=full, clear="True"), timeout=30)
    td.run_json("op('/sims/baker').par.Frameend = 40\nprint('@@JSON@@'+str(1))", timeout=20)
    td.run_json(BAKE, timeout=20)
    full_state = _wait_timeline_bake(td)
    replay = td.run_json(GROUP_REPLAY.format(cache=full), timeout=120)
    td.run_json(GROUP_SETUP.format(cache=resume, clear="True"), timeout=30)
    td.run_json(BAKE, timeout=20)
    _wait_timeline_bake(td, cancel_at=16)
    td.run_json(BAKE, timeout=20)
    _wait_timeline_bake(td)
    a_files, b_files = files(ROOT / full), files(ROOT / resume)
    rel_a = [p.relative_to(ROOT / full) for p in a_files]
    rel_b = [p.relative_to(ROOT / resume) for p in b_files]
    mismatched = [str(r) for r in rel_a if r not in rel_b or (ROOT / full / r).read_bytes() != (ROOT / resume / r).read_bytes()]
    # Coupled resume is byte-exact. pvel_max_diff is diagnostic only: a non-zero value fails the check.
    pvel_max = 0.0
    for rel in mismatched:
        if rel.startswith("particles/pvel/") and (ROOT / resume / rel).exists():
            pvel_max = max(pvel_max, _cache_max_diff(ROOT / full / rel, ROOT / resume / rel))
    stale = td.run_json(GROUP_STALE, timeout=30)
    for path in (ROOT / full, ROOT / resume):
        if path.exists():
            shutil.rmtree(path)
    fluid_diffs = list((replay.get("fluid") or {}).values())
    part_diffs = list((replay.get("particles") or {}).values())
    return {
        "full": full_state,
        "replay": replay,
        "files": len(a_files),
        "mismatched": mismatched[:8],
        "mismatched_count": len(mismatched),
        "pvel_max_diff": pvel_max,
        "identical": not mismatched and rel_a == rel_b and len(a_files) > 0,
        "replay_fluid": None if any(v is None for v in fluid_diffs) else max(fluid_diffs),
        "replay_particles": None if any(v is None for v in part_diffs) else max(part_diffs),
        "stale": stale,
    }


def _coupling_report(td, off, default, strong) -> dict:
    compared = td.run_json(r'''
import json, numpy as np
off = np.load(project.folder + '/reports/phase3_vel_off.npy').astype('float32')
default = np.load(project.folder + '/reports/phase3_vel_default.npy').astype('float32')
strong = np.load(project.folder + '/reports/phase3_vel_strong.npy').astype('float32')
print('@@JSON@@' + json.dumps({
    'mean_diff_default': float(np.mean(np.abs(default - off))),
    'mean_diff_strong': float(np.mean(np.abs(strong - off))),
    'max_off': float(np.max(np.abs(off))),
    'max_default': float(np.max(np.abs(default))),
    'max_strong': float(np.max(np.abs(strong))),
    'finite': bool(np.isfinite(off).all() and np.isfinite(default).all() and np.isfinite(strong).all()),
}))
''', timeout=60)
    max_off = max(compared["max_off"], 1e-6)
    compared["speed_ratio"] = compared["max_default"] / max_off
    compared["off"] = off
    compared["default"] = default
    compared["strong"] = strong
    return compared


def main() -> int:
    td = attach()
    td.run_json(RESTORE, timeout=30)
    report = {}
    shaders = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_shaders.py")], cwd=ROOT, capture_output=True, text=True)
    report["shaders"] = {"returncode": shaders.returncode, "tail": shaders.stdout.strip().splitlines()[-1:]}
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    report["pytest"] = {"returncode": tests.returncode, "tail": (tests.stdout or "").strip().splitlines()[-1:]}
    print("pytest", report["pytest"]["tail"], "shaders", report["shaders"])

    report["advection"] = td.run_json(ADVECT, timeout=60)
    print("advection", report["advection"])
    report["collide2d"] = td.run_json(COLLIDE2, timeout=60)
    print("collide2d", report["collide2d"])
    report["collide3d"] = td.run_json(COLLIDE3, timeout=90)
    print("collide3d", report["collide3d"])

    td.run_json(RESTORE, timeout=30)
    td.run_json(COUPLE_ON, timeout=30)
    off = td.run_json(_vel_code(0.0, "off"), timeout=180)
    default = td.run_json(_vel_code(1.0, "default"), timeout=180)
    strong = td.run_json(_vel_code(10.0, "strong"), timeout=180)
    report["coupling"] = _coupling_report(td, off, default, strong)
    print("coupling", report["coupling"])
    for name in ("phase3_vel_off.npy", "phase3_vel_default.npy", "phase3_vel_strong.npy"):
        (ROOT / "reports" / name).unlink(missing_ok=True)

    try:
        td.run_json(RENDER2, timeout=180)
        report["render2d"] = td.run_json(_save_picture("phase3_embers.png"), timeout=60)
    except Exception as exc:
        report["render2d"] = {"error": str(exc)}
    print("render2d", report["render2d"])
    try:
        td.run_json(RENDER3, timeout=180)
        report["render3d"] = td.run_json(_save_picture("phase3_sparks.png"), timeout=60)
    except Exception as exc:
        report["render3d"] = {"error": str(exc)}
    print("render3d", report["render3d"])
    report["plume"] = (report.get("render2d") or {}).get("plume")
    report["png_embers"] = png_stats(ROOT / "reports" / "phase3_embers.png")
    report["png_sparks"] = png_stats(ROOT / "reports" / "phase3_sparks.png")
    print("png", report["png_embers"], report["png_sparks"])

    # fps: particles alone, then with each fluid
    prep = dict(bus="2d", coupling=0, render="False", rx=1024, ry=1024, sims="/sims/particles", fluid2d="False", fluid3d="False")
    td.run_json(FPS_PREP.format(**prep), timeout=60)
    report["fps_alone"] = measure_fps(td, 3.0)
    prep["render"] = "True"
    td.run_json(FPS_PREP.format(**prep), timeout=60)
    report["fps_alone_rendered"] = measure_fps(td, 3.0)
    prep.update(render="False", coupling=1, sims="/sims/particles /sims/fluid2d", fluid2d="True")
    td.run_json(FPS_PREP.format(**prep), timeout=60)
    report["fps_with_2d"] = measure_fps(td, 3.0)
    prep.update(bus="3d", coupling=0, sims="/sims/fluid3d /sims/particles", fluid2d="False", fluid3d="True", render="False")
    td.run_json(FPS_PREP.format(**prep), timeout=90)
    report["fps_with_3d"] = measure_fps(td, 3.0)
    report["fps_with_3d"]["note"] = "1M particles advected by the 128^3 fluid while the volume render cooks. The 3D atlas splat is correct but too heavy to include in this rate."
    print("fps", report["fps_alone"], report["fps_alone_rendered"], report["fps_with_2d"], report["fps_with_3d"])

    full, resume = "cache/phase3_full", "cache/phase3_resume"
    td.run_json(BAKE_SETUP.format(cache=full, frames=8, clear="True"), timeout=30)
    td.run_json(BAKE, timeout=20)
    report["full_bake"] = wait_bake(td, full, replay=True)
    report["replay"] = report["full_bake"].get("replay")
    td.run_json(BAKE_SETUP.format(cache=resume, frames=8, clear="True"), timeout=30)
    td.run_json(BAKE, timeout=20)
    report["cancelled"] = wait_bake(td, resume, cancel_at=4)
    td.run_json(BAKE, timeout=20)
    report["resumed"] = wait_bake(td, resume)
    a_files, b_files = files(ROOT / full), files(ROOT / resume)
    rel_a = [p.relative_to(ROOT / full) for p in a_files]
    rel_b = [p.relative_to(ROOT / resume) for p in b_files]
    mismatched = [str(r) for r in rel_a if r not in rel_b or (ROOT / full / r).read_bytes() != (ROOT / resume / r).read_bytes()]
    report["resume_identical"] = {"files": len(a_files), "same_names": rel_a == rel_b, "mismatched": mismatched[:8]}
    print("bake", report["replay"], "mismatch", len(mismatched))
    report["group"] = group_bake(td)
    print("group", report["group"])

    # one small showcase, drop the resume duplicate
    showcase = ROOT / "cache" / "phase3_showcase"
    if showcase.exists():
        shutil.rmtree(showcase)
    shutil.copytree(ROOT / full, showcase)
    shutil.rmtree(ROOT / resume, ignore_errors=True)
    shutil.rmtree(ROOT / full, ignore_errors=True)

    td.run_json(RESTORE, timeout=30)
    phase1 = subprocess.run([sys.executable, str(ROOT / "scripts" / "bake_check.py"), "--frames", "40"], cwd=ROOT)
    report["phase1_bake_check"] = phase1.returncode
    print("phase1", phase1.returncode)
    phase2 = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_phase2.py")], cwd=ROOT)
    report["phase2"] = phase2.returncode
    print("phase2", phase2.returncode)

    adv = report.get("advection") or {}
    c2 = report.get("collide2d") or {}
    c3 = report.get("collide3d") or {}
    couple = report.get("coupling") or {}
    replay = report.get("replay") or {}
    ok = (
        report["shaders"]["returncode"] == 0
        and report["pytest"]["returncode"] == 0
        and adv.get("err", 1) < 1e-4
        and adv.get("others_life", 1) == 0
        and c2.get("inside", 1) == 0 and c2.get("alive", 0) > 10 and c2.get("min_sdf", 1) < 0.25
        and c3.get("inside", 1) == 0 and c3.get("alive", 0) > 4 and c3.get("finite", False)
        and couple.get("finite", False)
        and couple.get("mean_diff_default", 0) > 0
        and couple.get("speed_ratio", 99) <= 1.25
        and couple.get("mean_diff_strong", 0) > couple.get("mean_diff_default", 0)
        and (report.get("plume") or {}).get("centroid_y", 0) > (report.get("plume") or {}).get("emitter_y", 1)
        and (report.get("plume") or {}).get("inside", 0) > 0.5
        and (report.get("render2d") or {}).get("beauty", {}).get("lit", 0) > 0.001
        and (report.get("render3d") or {}).get("beauty", {}).get("lit", 0) > 0.001
        and (report.get("render2d") or {}).get("beauty", {}).get("alpha_min", -1) >= 0
        and (report.get("render3d") or {}).get("beauty", {}).get("alpha_min", -1) >= 0
        and (report.get("group") or {}).get("identical", False)
        and (report.get("group") or {}).get("replay_fluid") == 0
        and (report.get("group") or {}).get("replay_particles") == 0
        and (report.get("group") or {}).get("stale", {}).get("changed") == "stale"
        and (report.get("group") or {}).get("stale", {}).get("restored") == "complete"
        and _picture_ok(report.get("png_embers") or {})
        and _picture_ok(report.get("png_sparks") or {})
        and report["fps_alone"]["fps"] >= 60 and report["fps_alone"]["steps_match_frames"]
        and report["fps_with_2d"]["fps"] >= 30 and report["fps_with_2d"]["steps_match_frames"]
        and report["fps_with_3d"]["fps"] >= 30 and report["fps_with_3d"]["steps_match_frames"]
        and not mismatched and report["resume_identical"]["same_names"]
        and replay.get("diff") == 0.0
        and report["phase1_bake_check"] == 0
        and report["phase2"] == 0
    )
    report["pass"] = bool(ok)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "phase3.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
