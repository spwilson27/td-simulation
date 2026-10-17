"""Probe which GPU features the GLSL TOP supports on this machine; writes reports/gpu_probe.json.

    python scripts/probe_gpu.py            # uses the scratch TD from `td_scratch.py start`, else launches its own

Each probe builds a small network in /probe of the scratch instance, cooks it, reads pixels back and records
ok / value / compile log. /probe is destroyed afterwards.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from td_scratch import ROOT, attach, scratch_td  # noqa: E402

TD_CODE = r'''
import json, os, time
SHADERS = {shaders!r}
OUT_DIR = {out_dir!r}
results = {{}}

old = op('/probe')
if old is not None:
    old.destroy()
P = op('/').create(baseCOMP, 'probe')

def shader(top, fname, compute):
    d = P.create(textDAT, top.name + '_src')
    d.text = open(os.path.join(SHADERS, fname)).read()
    if compute:
        top.par.mode = 'compute'
        top.par.computedat = d
    else:
        top.par.pixeldat = d
    return d

def glsl(name, fname, compute=True, res=(16, 8), fmt='rgba32float', inputs=()):
    g = P.create(glslTOP, name)
    g.par.glslversion = 'glsl460'
    g.par.outputresolution = 'custom'
    g.par.resolutionw, g.par.resolutionh = res
    g.par.format = fmt
    shader(g, fname, compute)
    for i, src in enumerate(inputs):
        src.outputConnectors[0].connect(g.inputConnectors[i])
    return g

def log_of(g):
    info = P.op(g.name + '_info') or P.op(g.name + '_info1')
    txt = info.text if info is not None else ''
    errs = g.errors() or ''
    return (errs + '\n' + txt).strip()[-1500:]

def read(g):
    g.cook(force=True)
    return g.numpyArray(delayed=False)

def record(name, fn):
    try:
        ok, detail = fn()
        results[name] = dict(ok=bool(ok), **detail)
    except Exception as e:
        results[name] = dict(ok=False, exception=repr(e))

def t_compute2d():
    g = glsl('c2d', 'compute2d.glsl')
    a = read(g)
    want = (15.0, 7.0, 15 * 1000.5, -7.25)
    got = tuple(float(v) for v in a[7, 15])
    return got == want and not g.errors(), dict(texel_15_7=got, expected=want, log=log_of(g))

def t_compute3d():
    g = glsl('c3d', 'compute3d.glsl', res=(8, 8))
    g.par.type = 'texture3d'
    g.par.depth = 'custom'
    g.par.customdepth = 8
    r = glsl('r3d', 'read3d.glsl', compute=False, res=(8, 8), inputs=[g])
    r.par.vec0name = 'uSlice'
    r.par.vec0valuex = 5
    a = read(r)
    got = tuple(float(v) for v in a[2, 3])
    want = (3.0, 2.0, 5.0, 1.0)
    return got == want, dict(texel_3_2_5=got, expected=want, log=log_of(g) + '\n' + log_of(r))

def t_multipass_pixel():
    c = P.create(constantTOP, 'zero')
    c.par.outputresolution = 'custom'; c.par.resolutionw = 4; c.par.resolutionh = 4
    c.par.format = 'rgba32float'
    c.par.colorr = c.par.colorg = c.par.colorb = 0; c.par.alpha = 0
    g = glsl('mp', 'multipass.glsl', compute=False, res=(4, 4), inputs=[c])
    g.par.npasses = 4
    a = read(g)
    got = tuple(float(v) for v in a[0, 0][:2])
    return True, dict(r_g=got, feeds_back=got[0] == 4.0, utdpass_sum=got[1],
                      note='r=4 means pass k reads pass k-1 via input 0; g=6 means uTDPass counts 0..3', log=log_of(g))

def t_multipass_compute():
    g = glsl('mpc', 'multipass_compute.glsl', res=(8, 8))
    g.par.outputaccess = 'readwrite'
    g.par.npasses = 4
    a = read(g)
    got = tuple(float(v) for v in a[0, 0][:2])
    return got[0] == 4.0, dict(r_g=got, log=log_of(g))

def t_atomic_shared():
    g = glsl('ash', 'atomic_shared.glsl', res=(16, 16))
    a = read(g)
    got = float(a[:, :, 0].min()), float(a[:, :, 0].max())
    return got == (64.0, 64.0), dict(min_max=got, log=log_of(g))

def t_atomic_counter():
    g = glsl('acnt', 'atomic_counter.glsl', res=(16, 16))
    g.par.ac = 1
    g.par.ac0name = 'counter'
    a = read(g)
    vals = sorted(set(int(v) for v in a[:, :, 0].ravel()))
    return vals == list(range(256)), dict(n_unique=len(vals), max=max(vals), log=log_of(g),
                                          note='not needed: shared-memory atomics cover workgroup reductions')

def t_atomic_image_float():
    g = glsl('afl', 'atomic_image_float.glsl', res=(16, 16), fmt='mono32float')
    g.par.outputaccess = 'readwrite'
    a = read(g)
    got = float(a[0, 0, 0])
    return got == 256.0, dict(value=got, expected=256.0, log=log_of(g),
                              note='without float atomics, FLIP particle-to-grid uses a gather over binned particles')

def exr_roundtrip(fmt, w, h):
    g = glsl('rt_%s_%dx%d' % (fmt, w, h), 'compute2d.glsl', res=(w, h), fmt=fmt)
    src = read(g)
    path = os.path.join(OUT_DIR, 'probe_%s_%dx%d.exr' % (fmt, w, h))
    if os.path.exists(path):
        os.remove(path)
    g.save(path)
    m = P.create(moviefileinTOP)
    m.par.file = path
    m.cook(force=True)
    back = m.numpyArray(delayed=False)
    if back.shape != src.shape:
        return False, dict(error='read back %s, wrote %s' % (list(back.shape), list(src.shape)), warning=m.warnings())
    return float(abs(back - src).max()) == 0.0, dict(max_abs_diff=float(abs(back - src).max()))

def t_exr_roundtrip():
    detail, ok = {{}}, True
    for fmt in ('rgba32float', 'rgba16float', 'rg32float', 'mono32float'):
        o, d = exr_roundtrip(fmt, 512, 256)
        ok = ok and o
        detail[fmt] = d
    o, d = exr_roundtrip('rgba32float', 64, 32)
    detail['small_64x32_loads'] = o
    detail['note'] = 'Movie File In fails to open EXRs only a few rows tall (64x32, 256x8); 96x96 and up load fine'
    return ok, detail

def t_exr_save_speed():
    g = glsl('big', 'compute2d.glsl', res=(1024, 1024))
    g.cook(force=True)
    path = os.path.join(OUT_DIR, 'probe_1024.exr')
    t0 = time.perf_counter()
    g.save(path)
    sync_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    try:
        g.save(path.replace('.exr', '_async.exr'), asynchronous=True)
        async_ok = True
    except TypeError:
        async_ok = False
    async_ms = (time.perf_counter() - t0) * 1000
    return True, dict(sync_ms=round(sync_ms, 2), async_supported=async_ok, async_call_ms=round(async_ms, 2),
                      bytes=os.path.getsize(path))

for name, fn in [('compute2d_float32', t_compute2d), ('compute3d_texture', t_compute3d),
                 ('multipass_pixel', t_multipass_pixel), ('multipass_compute_readwrite', t_multipass_compute),
                 ('atomic_shared_uint', t_atomic_shared), ('atomic_counter', t_atomic_counter),
                 ('atomic_image_float', t_atomic_image_float), ('exr_roundtrip', t_exr_roundtrip),
                 ('exr_save_1024', t_exr_save_speed)]:
    record(name, fn)

results['_env'] = dict(td=app.version + '.' + app.build, os=app.osName + ' ' + app.osVersion)
P.destroy()
print('@@JSON@@' + json.dumps(results))
'''


def main() -> int:
    out_dir = ROOT / "cache" / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    code = TD_CODE.format(shaders=str(ROOT / "shaders" / "probe"), out_dir=str(out_dir))
    try:
        td = attach()
        results = td.run_json(code, timeout=120)
    except SystemExit:
        with scratch_td() as td:
            results = td.run_json(code, timeout=120)
    results["_probed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report = ROOT / "reports" / "gpu_probe.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(results, indent=2) + "\n")
    for name, r in results.items():
        if not name.startswith("_"):
            print(f"{'PASS' if r.get('ok') else 'FAIL'}  {name}")
    print(f"wrote {report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
