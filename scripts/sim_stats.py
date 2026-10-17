"""Reset the fluid with parameter overrides, let it run, and print field statistics (a quick stability check).

    python scripts/sim_stats.py Vorticity=0 Maccormack=0 --seconds 4 [--png reports/x.png]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from td_scratch import ROOT, attach  # noqa: E402

SET = r'''
import json
f = op('/sims/fluid2d')
for k, v in {overrides!r}.items():
    getattr(f.par, k).val = v
op.simclock.RequestReset()
print('@@JSON@@' + json.dumps(absTime.frame))
'''

STATS = r'''
import json, numpy as np
f = op('/sims/fluid2d')
v = f.op('vel_out').numpyArray(delayed=False)[..., :2]
sp = np.sqrt((v ** 2).sum(-1))
sc = f.op('scalars_out').numpyArray(delayed=False)
d = f.op('divergence').numpyArray(delayed=False)[..., 0]
png = {png!r}
if png:
    f.op('render_out').save(png)
print('@@JSON@@' + json.dumps({{'step': op.simclock.par.Step.eval(), 'speed_max': float(sp.max()),
    'speed_p99': float(np.percentile(sp, 99)), 'speed_median': float(np.median(sp)),
    'div_rms': float(np.sqrt((d ** 2).mean())), 'smoke_max': float(sc[..., 0].max()),
    'temp_max': float(sc[..., 1].max()), 'fuel_max': float(sc[..., 2].max()), 'finite': bool(np.isfinite(sc).all() and np.isfinite(v).all())}}))
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--png", default="")
    args = ap.parse_args(argv)
    overrides = {}
    for item in args.overrides:
        k, v = item.split("=", 1)
        overrides[k] = float(v)
    td = attach()
    td.run_json(SET.format(overrides=overrides))
    time.sleep(args.seconds)
    png = str((ROOT / args.png).resolve()) if args.png else ""
    print(json.dumps(td.run_json(STATS.format(png=png)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
