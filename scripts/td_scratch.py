"""Launch and drive a private TouchDesigner instance through the td-mcp debug component.

    python scripts/td_scratch.py start            # launch a detached scratch TD, remember it in .scratch/state.json
    python scripts/td_scratch.py run file.py      # execute a file (or '-' for stdin) on its main thread
    python scripts/td_scratch.py status
    python scripts/td_scratch.py stop             # kill only the process group we started

Library use:

    with scratch_td() as td:      # launches and kills a fresh instance
        td.run("1 + 1")

    td = attach()                  # the instance started with `start`

The instance gets its own port and connection file (under .scratch/), so TouchDesigner instances the user is running
are never touched. td-mcp is located via $TD_MCP_HOME, else a sibling checkout named td-mcp.
"""

from __future__ import annotations

import contextlib
import glob
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".scratch"
STATE_FILE = STATE_DIR / "state.json"
JSON_MARKER = "@@JSON@@"


def td_mcp_home() -> Path:
    env = os.environ.get("TD_MCP_HOME")
    home = Path(env).expanduser() if env else ROOT.parent / "td-mcp"
    if not (home / "src" / "td_mcp" / "client.py").is_file():
        raise SystemExit(f"td-mcp checkout not found at {home}; set TD_MCP_HOME")
    return home


def _client():
    src = str(td_mcp_home() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from td_mcp import client  # noqa: PLC0415

    return client


def find_td_binary() -> str:
    env = os.environ.get("TD_BIN")
    if env:
        return env
    candidates = ["/Applications/TouchDesigner.app/Contents/MacOS/TouchDesigner"]
    candidates += sorted(glob.glob("/Applications/TouchDesigner*.app/Contents/MacOS/TouchDesigner"), reverse=True)
    for c in candidates:
        if os.path.exists(c):
            return c
    raise SystemExit("TouchDesigner not found: set TD_BIN")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TDError(RuntimeError):
    pass


@dataclass
class ScratchTD:
    pid: int
    connection_file: Path

    def call(self, method: str, params: dict | None = None, timeout: float = 60) -> dict:
        client = _client()
        conn = client.load_connection(self.connection_file)
        return client.call(method, params or {}, connection=conn, timeout=timeout)

    def run(self, code: str, timeout: float = 60, quiet: bool = False):
        """Execute code in TD; returns the expression result (or None). Raises TDError with the traceback on failure."""
        res = self.call("execute_python", {"code": code, "timeout": timeout}, timeout=timeout)
        if not res.get("ok"):
            raise TDError(str(res.get("traceback") or res.get("error") or res))
        out = (res.get("stdout") or "").rstrip()
        if out and not quiet:
            print(out)
        if res.get("stderr") and not quiet:
            print(res["stderr"].rstrip(), file=sys.stderr)
        return res.get("result")

    def run_json(self, code: str, timeout: float = 60):
        """Execute code that prints `@@JSON@@<json>` as its last marker line; returns the decoded value.
        `json` is imported for the code."""
        code = "import json\n" + code
        res = self.call("execute_python", {"code": code, "timeout": timeout}, timeout=timeout)
        out = res.get("stdout") or ""
        if not res.get("ok"):
            raise TDError(out[-2000:] + "\n" + str(res.get("traceback") or res.get("error") or res))
        lines = [ln for ln in out.splitlines() if ln.startswith(JSON_MARKER)]
        if not lines:
            raise TDError("no JSON result in output:\n" + out[-2000:])
        return json.loads(lines[-1][len(JSON_MARKER):])

    def run_file(self, path: Path, timeout: float = 60):
        return self.run(Path(path).read_text(encoding="utf-8"), timeout=timeout)

    def alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
        except OSError:
            return False
        return True

    def kill(self) -> None:
        try:
            os.killpg(self.pid, signal.SIGTERM)
        except OSError:
            return
        for _ in range(40):
            if not self.alive():
                return
            time.sleep(0.25)
        with contextlib.suppress(OSError):
            os.killpg(self.pid, signal.SIGKILL)


def launch(work: Path, seed: Path | None = None, timeout: float = 150) -> ScratchTD:
    seed = seed or td_mcp_home() / "tox" / "td_python_debug.tox"
    work.mkdir(parents=True, exist_ok=True)
    conn_file = work / "connection.json"
    conn_file.unlink(missing_ok=True)
    env = dict(os.environ, TD_MCP_PORT=str(free_port()), TD_MCP_CONNECTION=str(conn_file))
    env.pop("TD_MCP_MODULEPATH", None)
    log = open(work / "td.log", "w")
    proc = subprocess.Popen([find_td_binary(), str(seed)], env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    td = ScratchTD(proc.pid, conn_file)
    client = _client()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"scratch TouchDesigner exited early, see {work / 'td.log'}")
        if conn_file.exists():
            with contextlib.suppress(Exception):
                client.load_connection(conn_file)
                if td.call("td_status", timeout=10).get("ok"):
                    return td
        time.sleep(0.5)
    td.kill()
    raise SystemExit("scratch TouchDesigner did not start the debug server in time")


@contextlib.contextmanager
def scratch_td(seed: Path | None = None):
    work = Path(tempfile.mkdtemp(prefix="td_sims_"))
    td = launch(work, seed)
    try:
        yield td
    finally:
        td.kill()
        for f in work.iterdir():
            f.unlink()
        work.rmdir()


def attach() -> ScratchTD:
    if not STATE_FILE.is_file():
        raise SystemExit("no scratch TD running: python scripts/td_scratch.py start")
    state = json.loads(STATE_FILE.read_text())
    td = ScratchTD(int(state["pid"]), Path(state["connection_file"]))
    if not td.alive():
        raise SystemExit("scratch TD is not running any more: start it again")
    return td


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "status"
    if cmd == "start":
        if STATE_FILE.is_file():
            with contextlib.suppress(SystemExit):
                td = attach()
                print(f"already running: pid {td.pid}")
                return 0
        td = launch(STATE_DIR)
        STATE_FILE.write_text(json.dumps({"pid": td.pid, "connection_file": str(td.connection_file)}))
        print(f"scratch TouchDesigner pid {td.pid}")
        return 0
    if cmd == "stop":
        if STATE_FILE.is_file():
            state = json.loads(STATE_FILE.read_text())
            ScratchTD(int(state["pid"]), Path(state["connection_file"])).kill()
            STATE_FILE.unlink()
            (STATE_DIR / "connection.json").unlink(missing_ok=True)
        print("stopped")
        return 0
    if cmd == "status":
        td = attach()
        res = td.call("td_status")
        res.pop("token", None)
        print(json.dumps(res, indent=2))
        return 0
    if cmd == "run":
        td = attach()
        src = sys.stdin.read() if len(argv) < 2 or argv[1] == "-" else Path(argv[1]).read_text(encoding="utf-8")
        timeout = float(argv[2]) if len(argv) > 2 else 60
        try:
            result = td.run(src, timeout=timeout)
        except TDError as exc:
            print(exc, file=sys.stderr)
            return 1
        if result is not None:
            print(result)
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
