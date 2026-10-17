"""Simulation cache layout, manifests and staleness rules. Pure Python: imported by TD extensions and by tests.

Layout of one cache:

    <cache_dir>/manifest.json
    <cache_dir>/<field>/<field>.<frame:05d>.exr

A cache is identified by the hash of the parameters that affect the simulation result. Changing any of them, or
the frame_start, makes the cache stale. A matching cache can be resumed, including when the requested frame_end
extends past the manifest: existing frames stay and the bake continues from the first missing frame.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

CACHE_VERSION = 1
MANIFEST_NAME = "manifest.json"

BYTES_PER_PIXEL = {
    "rgba32float": 16, "rgba16float": 8, "rg32float": 8, "rg16float": 4,
    "mono32float": 4, "mono16float": 2, "rgba8fixed": 4,
}


def _canonical(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return repr(value)
        return float(f"{value:.9g}")
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def param_hash(params: dict) -> str:
    """Stable 16-hex-digit hash of simulation parameters (key order and float noise beyond 9 digits ignored)."""
    blob = json.dumps(_canonical(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class Manifest:
    sim: str
    params: dict
    fields: dict            # field name -> pixel format
    res: list               # [w, h] or [w, h, d]
    bounds: list            # world-space [xmin, ymin, xmax, ymax] (2D) or 6 values (3D)
    fps: float
    substeps: int
    frame_start: int
    frame_end: int          # inclusive
    param_hash: str = ""
    version: int = CACHE_VERSION
    complete: bool = False
    frames_written: int = 0
    td_version: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.param_hash:
            self.param_hash = param_hash(self.params)

    @property
    def dt(self) -> float:
        return 1.0 / (self.fps * self.substeps)

    @property
    def frame_count(self) -> int:
        return self.frame_end - self.frame_start + 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dt"] = self.dt
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def manifest_path(cache_dir) -> Path:
    return Path(cache_dir) / MANIFEST_NAME


def write_manifest(cache_dir, manifest: Manifest) -> Path:
    """Write atomically, so a crash mid-write never leaves a truncated manifest."""
    dest = manifest_path(cache_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".manifest.", dir=dest.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, dest)
    return dest


def read_manifest(cache_dir) -> Manifest | None:
    path = manifest_path(cache_dir)
    if not path.is_file():
        return None
    try:
        return Manifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def frame_path(cache_dir, field_name: str, frame: int) -> Path:
    return Path(cache_dir) / field_name / f"{field_name}.{int(frame):05d}.exr"


def frame_exists(cache_dir, manifest: Manifest, frame: int) -> bool:
    return all(frame_path(cache_dir, f, frame).is_file() for f in manifest.fields)


def resume_frame(cache_dir, manifest: Manifest) -> int:
    """First frame (from frame_start) that is missing any field; frame_end + 1 when all frames exist."""
    for frame in range(manifest.frame_start, manifest.frame_end + 1):
        if not frame_exists(cache_dir, manifest, frame):
            return frame
    return manifest.frame_end + 1


MISSING, STALE, PARTIAL, COMPLETE = "missing", "stale", "partial", "complete"

# Clock uniforms are the same for every publisher and change (or can change) every step. They are not part of a
# publisher's identity; dt/fps/seed are hashed with the sim. An expression is stored as text so a disc that follows
# sin(time * Discspeed) does not make the cache stale every frame. The Discspeed value itself is hashed with the
# owning COMP's custom parameters. A constant uniform is stored as its value.
CLOCK_UNIFORMS = frozenset({"uClock", "uClock2"})


def glsl_uniform_identity(blocks) -> list:
    """Publisher GLSL vec uniforms that belong in the cache hash.

    `blocks` is a sequence of `(name, components)` and each component is `(expression, value)`.
    Clock uniforms are dropped. Other expressions are kept as source text; constants are kept as values.
    """
    found = []
    for name, components in blocks:
        if not name or name in CLOCK_UNIFORMS:
            continue
        recorded = []
        for expr, value in components:
            recorded.append(str(expr) if expr else value)
        found.append([str(name), recorded])
    found.sort(key=lambda item: item[0])
    return found


def custom_par_identity(pars) -> dict:
    """Owning-COMP custom parameters for the cache hash. Pulses are skipped. An expression is stored as text
    so a time-varying parameter does not change the hash every frame; a constant is stored as its value."""
    out = {}
    for name, expr, value, style in pars:
        if str(style).lower() == "pulse":
            continue
        out[str(name)] = str(expr) if expr else value
    return out


def plan_bake(manifest: Manifest | None, params: dict, frame_start: int, frame_end: int, present: set[int]) -> dict:
    """What Bake should do with an existing cache.

    `present` is the set of frame numbers whose field files are all on disk. A longer Frameend with the same
    param hash, version, and frame_start is a resume that keeps those files and continues from the first gap
    (or from the old end, when the cached range is complete). A shorter Frameend stops at the request: if that
    prefix is already on disk the plan is complete, and the bake must not simulate past it. A different
    frame_start, param hash, or cache version is stale. No manifest is missing.
    """
    frame_start, frame_end = int(frame_start), int(frame_end)
    base = {"frame_start": frame_start, "frame_end": frame_end, "resume_frame": frame_start}
    if manifest is None:
        return {"action": "missing", **base}
    if manifest.version != CACHE_VERSION or manifest.param_hash != param_hash(params):
        return {"action": "stale", **base}
    if frame_start != int(manifest.frame_start):
        return {"action": "stale", **base}
    resume = frame_end + 1
    for frame in range(frame_start, frame_end + 1):
        if frame not in present:
            resume = frame
            break
    action = "complete" if resume > frame_end else "resume"
    return {"action": action, "frame_start": frame_start, "frame_end": frame_end, "resume_frame": resume}


def directory_plan(cache_dir, params: dict, frame_start: int, frame_end: int) -> dict:
    """plan_bake for a cache directory, using the files actually on disk."""
    manifest = read_manifest(cache_dir)
    present: set[int] = set()
    if manifest is not None:
        lo = min(int(manifest.frame_start), int(frame_start))
        hi = max(int(manifest.frame_end), int(frame_end))
        present = {frame for frame in range(lo, hi + 1) if frame_exists(cache_dir, manifest, frame)}
    return plan_bake(manifest, params, frame_start, frame_end, present)


def cache_status(cache_dir, params: dict, frame_start: int | None = None, frame_end: int | None = None) -> str:
    """missing: no manifest; stale: parameters, version, or frame_start changed;
    partial: same identity but at least one requested frame is missing (including a range extension);
    complete: every requested frame of every field is present. A requested end inside a longer cache is
    complete once that prefix exists."""
    manifest = read_manifest(cache_dir)
    if manifest is None:
        return MISSING
    start = int(manifest.frame_start) if frame_start is None else int(frame_start)
    end = int(manifest.frame_end) if frame_end is None else int(frame_end)
    action = directory_plan(cache_dir, params, start, end)["action"]
    return {"missing": MISSING, "stale": STALE, "resume": PARTIAL, "complete": COMPLETE}[action]


def clear_cache(cache_dir, manifest: Manifest | None = None) -> int:
    """Delete the manifest and frame files this module writes (nothing else). Returns the number of files removed."""
    root = Path(cache_dir)
    removed = 0
    if manifest is not None:
        fields = list(manifest.fields)
    elif root.is_dir():
        fields = [p.name for p in root.iterdir() if p.is_dir()]
    else:
        fields = []
    for name in fields:
        d = root / name
        if not d.is_dir():
            continue
        for f in d.glob(f"{name}.*.exr"):
            f.unlink()
            removed += 1
        if not any(d.iterdir()):
            d.rmdir()
    mp = manifest_path(root)
    if mp.is_file():
        mp.unlink()
        removed += 1
    return removed


def estimate_bytes(res, fields: dict, frames: int) -> int:
    """Uncompressed upper bound; EXR zip usually lands at 30-70% of this for smoke."""
    pixels = math.prod(int(r) for r in res)
    return pixels * sum(BYTES_PER_PIXEL.get(fmt, 16) for fmt in fields.values()) * int(frames)


def blend_frames(t: float, frame_start: int, frame_end: int) -> tuple[int, int, float]:
    """Fractional frame -> (frame a, frame b, weight of b), clamped to the cached range."""
    t = min(max(float(t), frame_start), frame_end)
    a = int(math.floor(t))
    b = min(a + 1, frame_end)
    return a, b, (t - a) if b != a else 0.0
