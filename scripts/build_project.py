"""Build /sims in a scratch TouchDesigner from the repo sources and save sims.toe and tox/sims.tox.

    python scripts/build_project.py            # uses the instance from `td_scratch.py start`, else launches one
    python scripts/build_project.py --no-save  # build and check only

The project is saved to the repo root first so `project.folder` (the default Root of /sims) is the repo, then the
network is built, the demo publishers are registered and operator errors are checked. The td-mcp debug component
in the project is saved with a blank Token (it generates a new one when the project starts).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from td_scratch import ROOT, attach, scratch_td  # noqa: E402

SAVE_PROJECT = r'''
import json
dbg = op('/td_python_debug')
tok = dbg.par.Token.eval() if dbg is not None else None
if dbg is not None:
    dbg.par.Token = ''
try:
    project.save({toe!r})
finally:
    if dbg is not None:
        dbg.par.Token = tok
print('@@JSON@@' + json.dumps({{'folder': project.folder, 'name': project.name}}))
'''

BUILD = r'''
import importlib.util, json, sys
path = {root!r} + '/python'
if path not in sys.path:
    sys.path.insert(0, path)
spec = importlib.util.spec_from_file_location('build_network', path + '/build_network.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('@@JSON@@' + json.dumps(m.build('/')))
'''

CONFIGURE = r'''
import importlib.util, json
spec = importlib.util.spec_from_file_location('build_network', {root!r} + '/python/build_network.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.register_demo()
op('/sims/fluid2d').ApplyPreset('realtime')
op.simclock.RequestReset()
print('@@JSON@@' + json.dumps({{'publishers': {{k: op.simbus.Publishers(k) for k in ('sources', 'forces', 'obstacles')}},
                                'res': list(op.simbus.Res)}}))
'''

SAVE_TOX = r'''
import json
op('/sims').save({tox!r})
print('@@JSON@@' + json.dumps('ok'))
'''


def build(td, save: bool) -> int:
    toe = str(ROOT / "sims.toe")
    print("project:", td.run_json(SAVE_PROJECT.format(toe=toe)))
    print("built:", td.run_json(BUILD.format(root=str(ROOT)), timeout=120))
    time.sleep(1.0)   # let extensions initialise and synced DATs load
    print("configured:", td.run_json(CONFIGURE.format(root=str(ROOT))))
    time.sleep(1.0)
    errs = td.call("operator_errors", {"path": "/sims", "max_depth": 6, "limit": 50})
    problems = errs.get("operators") or errs.get("results") or []
    if problems:
        print("operator errors:\n" + json.dumps(problems, indent=1)[:6000])
    if save:
        print("tox:", td.run_json(SAVE_TOX.format(tox=str(ROOT / "tox" / "sims.tox"))))
        print("project:", td.run_json(SAVE_PROJECT.format(toe=toe)))
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    try:
        td = attach()
    except SystemExit:
        with scratch_td() as td:
            return build(td, not args.no_save)
    return build(td, not args.no_save)


if __name__ == "__main__":
    sys.exit(main())
