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

# Movie File In on this machine fails to open rgba16float EXRs when either edge is 128 or smaller
# (256x128 failed, 256x256 and up loaded). 3D atlases are padded out to this edge.
MIN_ATLAS_EDGE = 256

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
    # Empty for a single-sim cache. A group cache maps sim name -> {params, fields, res, bounds, extra}
    # and stores frames under <cache>/<sim>/<field>/.
    members: dict = field(default_factory=dict)

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


def frame_path(cache_dir, field_name: str, frame: int, sim: str | None = None) -> Path:
    """EXR path for one field. `sim` selects a member subfolder of a group cache."""
    root = Path(cache_dir) / sim if sim else Path(cache_dir)
    return root / field_name / f"{field_name}.{int(frame):05d}.exr"


def storage_of(manifest: Manifest | None, sim: str | None = None) -> dict:
    """Bake layout options. Missing storage is the phase 1-3 layout: every frame stores every field."""
    if manifest is None:
        return {}
    if sim and manifest.members:
        extra = (manifest.members.get(sim) or {}).get("extra") or {}
        if extra.get("storage"):
            return dict(extra["storage"])
    return dict((manifest.extra or {}).get("storage") or {})


def is_keyframe(manifest: Manifest, frame: int, sim: str | None = None) -> bool:
    """True when this frame stores a full resume state. Interval 0 or 1 keeps every frame."""
    interval = int(storage_of(manifest, sim).get("keyframe_interval") or 0)
    if interval <= 1:
        return True
    return (int(frame) - int(manifest.frame_start)) % interval == 0


def fields_for_frame(manifest: Manifest, frame: int, sim: str | None = None) -> list[str]:
    """Field names that must be on disk for this frame."""
    if sim and manifest.members:
        pool = list(manifest.members[sim]["fields"])
    else:
        pool = list(manifest.fields)
    storage = storage_of(manifest, sim)
    if is_keyframe(manifest, frame, sim):
        names = storage.get("keyframe_fields") or pool
    else:
        names = storage.get("playback_fields") or pool
    chosen = [name for name in names if name in pool]
    # A sim whose fields are not the fluid scalars (rigid state, particles) still stores its own
    # fields every frame. An empty intersection would otherwise mark the frame complete with nothing saved.
    return chosen or list(pool)


def resolve_frame(manifest: Manifest, field: str, frame: int, sim: str | None = None) -> int:
    """Latest frame at or before `frame` that actually stores `field` (a keyframe, when playback omits it)."""
    frame = int(frame)
    start = int(manifest.frame_start)
    cursor = frame
    while cursor >= start:
        if field in fields_for_frame(manifest, cursor, sim):
            return cursor
        cursor -= 1
    return frame


def resume_state_frame(manifest: Manifest, frame: int, sim: str | None = None) -> int | None:
    """Frame to load before simulating `frame`.

    That is the latest keyframe at or before `frame - 1`. None means the resume is the first frame and the
    caller loads zeros. When every frame is a keyframe this is `frame - 1`, the same resume the phase 1 baker uses.
    """
    prev = int(frame) - 1
    if prev < int(manifest.frame_start):
        return None
    cursor = prev
    while cursor > int(manifest.frame_start) and not is_keyframe(manifest, cursor, sim):
        cursor -= 1
    return cursor


def frame_bbox(manifest: Manifest, frame: int) -> list | None:
    boxes = (manifest.extra or {}).get("bboxes") or {}
    box = boxes.get(str(int(frame)))
    return list(box) if box is not None else None


def bbox_shape(bbox: list) -> list[int]:
    """Inclusive voxel box [x0, y0, z0, x1, y1, z1] -> [width, height, depth]."""
    return [int(bbox[3]) - int(bbox[0]) + 1, int(bbox[4]) - int(bbox[1]) + 1, int(bbox[5]) - int(bbox[2]) + 1]


def occupied_bbox(points, res, pad: int = 0) -> list[int]:
    """Inclusive voxel box around `(x, y, z)` points, padded and clamped to `res` = [w, h, d].

    An empty point list is the full domain, so a frame with no smoke still has a well-defined atlas.
    """
    w, h, d = (int(res[0]), int(res[1]), int(res[2]))
    pad = int(pad)
    if not points:
        return [0, 0, 0, w - 1, h - 1, d - 1]
    xs, ys, zs = zip(*((int(p[0]), int(p[1]), int(p[2])) for p in points))
    return [
        max(0, min(xs) - pad), max(0, min(ys) - pad), max(0, min(zs) - pad),
        min(w - 1, max(xs) + pad), min(h - 1, max(ys) + pad), min(d - 1, max(zs) + pad),
    ]


def _frame_slots(manifest: Manifest, frame: int | None = None) -> list[tuple[str | None, str]]:
    if manifest.members:
        slots = []
        for sim, info in manifest.members.items():
            names = info["fields"] if frame is None else fields_for_frame(manifest, frame, sim)
            slots.extend((sim, name) for name in names)
        return slots
    names = manifest.fields if frame is None else fields_for_frame(manifest, frame)
    return [(None, name) for name in names]


def frame_exists(cache_dir, manifest: Manifest, frame: int) -> bool:
    return all(frame_path(cache_dir, name, frame, sim).is_file() for sim, name in _frame_slots(manifest, frame))


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


def plan_bake(manifest: Manifest | None, params: dict, frame_start: int, frame_end: int, present: set[int],
              storage: dict | None = None) -> dict:
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
    # storage=None skips the check so a caller that does not know about keyframes still reads old caches.
    # An empty dict is the legacy layout and matches a manifest that never recorded storage.
    if storage is not None and ((manifest.extra or {}).get("storage") or {}) != storage:
        return {"action": "stale", **base}
    resume = frame_end + 1
    for frame in range(frame_start, frame_end + 1):
        if frame not in present:
            resume = frame
            break
    action = "complete" if resume > frame_end else "resume"
    return {"action": action, "frame_start": frame_start, "frame_end": frame_end, "resume_frame": resume}


def directory_plan(cache_dir, params: dict, frame_start: int, frame_end: int, storage: dict | None = None) -> dict:
    """plan_bake for a cache directory, using the files actually on disk."""
    manifest = read_manifest(cache_dir)
    present: set[int] = set()
    if manifest is not None:
        lo = min(int(manifest.frame_start), int(frame_start))
        hi = max(int(manifest.frame_end), int(frame_end))
        present = {frame for frame in range(lo, hi + 1) if frame_exists(cache_dir, manifest, frame)}
    return plan_bake(manifest, params, frame_start, frame_end, present, storage=storage)


def cache_status(cache_dir, params: dict, frame_start: int | None = None, frame_end: int | None = None,
                 storage: dict | None = None) -> str:
    """missing: no manifest; stale: parameters, version, or frame_start changed;
    partial: same identity but at least one requested frame is missing (including a range extension);
    complete: every requested frame of every field is present. A requested end inside a longer cache is
    complete once that prefix exists."""
    manifest = read_manifest(cache_dir)
    if manifest is None:
        return MISSING
    start = int(manifest.frame_start) if frame_start is None else int(frame_start)
    end = int(manifest.frame_end) if frame_end is None else int(frame_end)
    action = directory_plan(cache_dir, params, start, end, storage=storage)["action"]
    return {"missing": MISSING, "stale": STALE, "resume": PARTIAL, "complete": COMPLETE}[action]


def clear_cache(cache_dir, manifest: Manifest | None = None) -> int:
    """Delete the manifest and frame files this module writes (nothing else). Returns the number of files removed."""
    root = Path(cache_dir)
    removed = 0
    if manifest is not None and manifest.members:
        for sim, info in manifest.members.items():
            for name in info.get("fields", {}):
                d = root / sim / name
                if not d.is_dir():
                    continue
                for pattern in (f"{name}.*.exr", f"{name}.*.npy"):
                    for f in d.glob(pattern):
                        f.unlink()
                        removed += 1
                if not any(d.iterdir()):
                    d.rmdir()
            sim_dir = root / sim
            if sim_dir.is_dir() and not any(sim_dir.iterdir()):
                sim_dir.rmdir()
        mp = manifest_path(root)
        if mp.is_file():
            mp.unlink()
            removed += 1
        return removed
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
        for pattern in (f"{name}.*.exr", f"{name}.*.npy"):
            for f in d.glob(pattern):
                f.unlink()
                removed += 1
        if not any(d.iterdir()):
            d.rmdir()
    mp = manifest_path(root)
    if mp.is_file():
        mp.unlink()
        removed += 1
    return removed


def atlas_layout(res) -> dict:
    """Slice-atlas layout for a 3D volume stored as one 2D EXR.

    Slices are tiles of `volume[0]` by `volume[1]`. `cols * rows` is at least the depth, and both atlas
    edges are at least MIN_ATLAS_EDGE so a half-float EXR will open in Movie File In. Empty tiles (the
    padding past `depth`) are unused. The result is deterministic for a given resolution.
    """
    if len(res) < 3:
        raise ValueError("atlas_layout expects a 3D resolution")
    w, h, d = (max(1, int(res[0])), max(1, int(res[1])), max(1, int(res[2])))
    best = None
    for cols in range(1, d + 1):
        rows = math.ceil(d / cols)
        pc, pr = cols, rows
        while pc * w < MIN_ATLAS_EDGE:
            pc += 1
        while pr * h < MIN_ATLAS_EDGE:
            pr += 1
        waste = pc * pr - d
        aw, ah = pc * w, pr * h
        score = (waste, abs(aw - ah), aw * ah, pc)
        if best is None or score < best[0]:
            best = (score, pc, pr, aw, ah)
    _, cols, rows, aw, ah = best
    return {"cols": cols, "rows": rows, "width": aw, "height": ah, "volume": [w, h, d]}


def atlas_pixel(x: int, y: int, z: int, layout: dict) -> tuple[int, int]:
    """Volume voxel -> atlas pixel. `layout` is an atlas_layout dict."""
    w, h, d = layout["volume"]
    cols = int(layout["cols"])
    x, y, z = int(x), int(y), int(z)
    if not (0 <= x < w and 0 <= y < h and 0 <= z < d):
        raise ValueError(f"voxel {(x, y, z)} outside volume {layout['volume']}")
    return (z % cols) * w + x, (z // cols) * h + y


def atlas_voxel(px: int, py: int, layout: dict) -> tuple[int, int, int] | None:
    """Atlas pixel -> volume voxel, or None when the pixel is padding past the depth."""
    w, h, d = layout["volume"]
    cols = int(layout["cols"])
    px, py = int(px), int(py)
    if not (0 <= px < int(layout["width"]) and 0 <= py < int(layout["height"])):
        return None
    tx, x = divmod(px, w)
    ty, y = divmod(py, h)
    z = ty * cols + tx
    if z >= d or tx >= cols or ty >= int(layout["rows"]):
        return None
    return x, y, z


def stored_pixels(res) -> int:
    """Pixels actually written per frame. 3D fields are slice atlases, which can be larger than the voxel count."""
    dims = [int(r) for r in res]
    if len(dims) >= 3:
        layout = atlas_layout(dims[:3])
        return int(layout["width"]) * int(layout["height"])
    return math.prod(dims)


def _field_pixels(res, bbox) -> int:
    shape = bbox_shape(bbox) if bbox is not None else [int(v) for v in res]
    if len(shape) >= 3:
        return stored_pixels(shape[:3])
    return math.prod(int(v) for v in shape)


def estimate_bytes(res, fields: dict, frames: int, storage: dict | None = None, frame_start: int = 1,
                   bboxes: dict | None = None) -> int:
    """Uncompressed upper bound; EXR zip usually lands at 30-70% of this for smoke.

    A 3D `res` is stored as a 2D slice atlas (see atlas_layout), so the bound uses the atlas pixel count.
    When the slice grid is exact and no edge padding is required, that equals the voxel count.

    `storage` selects a reduced layout: keyframes store `keyframe_fields` (default: every field) and the
    frames between them store `playback_fields`. `bboxes` maps a frame number to an inclusive voxel box;
    fields listed in `storage["crop_fields"]` then use that box's atlas instead of the full grid. Without
    `storage` and `bboxes` the bound is the full layout, which is what phases 1-3 write.
    """
    frames = int(frames)
    if not storage and not bboxes:
        return stored_pixels(res) * sum(BYTES_PER_PIXEL.get(fmt, 16) for fmt in fields.values()) * frames
    storage = dict(storage or {})
    crop_fields = set(storage.get("crop_fields") or [])
    total = 0
    # A tiny stand-in manifest so fields_for_frame / is_keyframe see the same rules the baker uses.
    standin = Manifest(
        sim="", params={}, fields=dict(fields), res=list(res), bounds=[], fps=1, substeps=1,
        frame_start=int(frame_start), frame_end=int(frame_start) + frames - 1,
        extra={"storage": storage} if storage else {},
    )
    for index in range(frames):
        frame = int(frame_start) + index
        names = fields_for_frame(standin, frame)
        bbox = None
        if bboxes:
            bbox = bboxes.get(str(frame))
            if bbox is None:
                bbox = bboxes.get(frame)
        for name in names:
            fmt = fields.get(name, "rgba16float")
            use_box = bbox if name in crop_fields and not is_keyframe(standin, frame) else None
            total += _field_pixels(res, use_box) * BYTES_PER_PIXEL.get(fmt, 16)
    return int(total)


def blend_frames(t: float, frame_start: int, frame_end: int) -> tuple[int, int, float]:
    """Fractional frame -> (frame a, frame b, weight of b), clamped to the cached range."""
    t = min(max(float(t), frame_start), frame_end)
    a = int(math.floor(t))
    b = min(a + 1, frame_end)
    return a, b, (t - a) if b != a else 0.0
