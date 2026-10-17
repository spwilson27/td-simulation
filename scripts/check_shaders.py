"""Compile every GLSL shader offline with glslangValidator, without TouchDesigner.

    python scripts/check_shaders.py [paths...]      # default: every .glsl under shaders/ that has a main()

A bad shader in TD shows checkerboards, error dialogs, or can hang the GPU. This catches syntax, type and undeclared
identifier errors first. Each file is wrapped with stand-ins for the TD built-ins it relies on (sTD2DInputs,
uTDOutputInfo, TDOutputSwizzle, TDImageStoreOutput, mTDComputeOutputs, uTDPass, ...) and `#include <name>` is
resolved to shaders/**/<name minus _glsl>.glsl, the way the builder maps DAT names to files.

Stage: compute when the file declares `local_size_x`, otherwise pixel. 3D compute output (image3D) when it declares
`local_size_z` greater than 1 or has a `// td: output3d` line. Files containing `// td: skip-check` are skipped
(probes that are expected to fail). Exit status 1 when any shader fails.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHADERS = ROOT / "shaders"
MAX_INPUTS = 8

COMMON = f"""#version 460
#define TD_NUM_2D_INPUTS {MAX_INPUTS}
#define TD_NUM_3D_INPUTS {MAX_INPUTS}
#define TD_NUM_COLOR_BUFFERS 1
uniform sampler2D sTD2DInputs[TD_NUM_2D_INPUTS];
uniform sampler3D sTD3DInputs[TD_NUM_3D_INPUTS];
uniform sampler2DArray sTD2DArrayInputs[TD_NUM_2D_INPUTS];
struct TDTexInfo {{ vec4 res; vec4 depth; }};
uniform TDTexInfo uTD2DInfos[TD_NUM_2D_INPUTS];
uniform TDTexInfo uTD3DInfos[TD_NUM_3D_INPUTS];
uniform TDTexInfo uTDOutputInfo;
uniform int uTDPass;
uniform vec4 uTDCurrentDepth;
float TDSimplexNoise(vec2 v) {{ return 0.0; }}
float TDSimplexNoise(vec3 v) {{ return 0.0; }}
float TDSimplexNoise(vec4 v) {{ return 0.0; }}
float TDPerlinNoise(vec2 v) {{ return 0.0; }}
float TDPerlinNoise(vec3 v) {{ return 0.0; }}
vec3 TDHSVToRGB(vec3 c) {{ return c; }}
vec3 TDRGBToHSV(vec3 c) {{ return c; }}
"""

PIXEL = COMMON + """in vec3 vUV;
vec4 TDOutputSwizzle(vec4 c) { return c; }
uvec4 TDOutputSwizzle(uvec4 c) { return c; }
vec2 TDPointCoord() { return vec2(0.5); }
"""

VERTEX = COMMON + """
vec3 TDPos() { return vec3(0.0); }
vec3 TDNormal() { return vec3(0.0, 0.0, 1.0); }
vec3 TDTexCoord(uint layer) { return vec3(0.0); }
vec4 TDDeform(vec3 p) { return vec4(p, 1.0); }
vec4 TDWorldToProj(vec3 p) { return vec4(p, 1.0); }
vec4 TDWorldToProj(vec4 p) { return p; }
int TDInstanceID() { return 0; }
void TDWritePickingValues() {}
#define gl_VertexIndex 0
#define gl_InstanceIndex 0
"""

COMPUTE_2D = COMMON + """layout(rgba32f) uniform image2D mTDComputeOutputs[TD_NUM_COLOR_BUFFERS];
void TDImageStoreOutput(uint i, uvec3 c, vec4 v) { imageStore(mTDComputeOutputs[0], ivec2(c.xy), v); }
void TDImageStoreOutput(uint i, ivec3 c, vec4 v) { imageStore(mTDComputeOutputs[0], c.xy, v); }
vec4 TDImageLoadOutput(uint i, uvec3 c) { return imageLoad(mTDComputeOutputs[0], ivec2(c.xy)); }
vec4 TDImageLoadOutput(uint i, ivec3 c) { return imageLoad(mTDComputeOutputs[0], c.xy); }
vec4 TDOutputSwizzle(vec4 c) { return c; }
"""

COMPUTE_3D = COMMON + """layout(rgba32f) uniform image3D mTDComputeOutputs[TD_NUM_COLOR_BUFFERS];
void TDImageStoreOutput(uint i, uvec3 c, vec4 v) { imageStore(mTDComputeOutputs[0], ivec3(c), v); }
void TDImageStoreOutput(uint i, ivec3 c, vec4 v) { imageStore(mTDComputeOutputs[0], c, v); }
vec4 TDImageLoadOutput(uint i, uvec3 c) { return imageLoad(mTDComputeOutputs[0], ivec3(c)); }
vec4 TDImageLoadOutput(uint i, ivec3 c) { return imageLoad(mTDComputeOutputs[0], c); }
vec4 TDOutputSwizzle(vec4 c) { return c; }
"""

INCLUDE_RE = re.compile(r'^\s*#include\s*[<"]([^>"]+)[>"]', re.M)
EXTENSION_RE = re.compile(r"^\s*#extension[^\n]*\n", re.M)


def find_include(name: str) -> Path | None:
    stem = Path(name).name
    stem = stem[:-5] if stem.endswith("_glsl") else stem
    hits = sorted(SHADERS.rglob(f"{stem}.glsl"))
    return hits[0] if hits else None


def expand(src: str, seen: set[Path]) -> str:
    def repl(m):
        path = find_include(m.group(1))
        if path is None:
            raise FileNotFoundError(f"#include <{m.group(1)}>: no shaders/**/{m.group(1)}.glsl")
        if path in seen:
            return ""
        seen.add(path)
        return expand(path.read_text(encoding="utf-8"), seen)
    return INCLUDE_RE.sub(repl, src)


def stage_of(src: str) -> tuple[str, str]:
    if "// td: stage vertex" in src:
        return "vert", VERTEX
    if "local_size_x" not in src:
        return "frag", PIXEL
    z = re.search(r"local_size_z\s*=\s*(\d+)", src)
    is3d = "// td: output3d" in src or (z is not None and int(z.group(1)) > 1)
    return "comp", COMPUTE_3D if is3d else COMPUTE_2D


def check(path: Path, validator: str) -> tuple[bool, str]:
    src = path.read_text(encoding="utf-8")
    if "// td: skip-check" in src:
        return True, "skipped"
    try:
        body = expand(src, set())
    except FileNotFoundError as exc:
        return False, str(exc)
    stage, preamble = stage_of(body)
    extensions = "".join(EXTENSION_RE.findall(body))
    body = EXTENSION_RE.sub("", body)
    # #extension must come right after #version; keep line numbers of the user file readable via #line.
    full = preamble.replace("#version 460\n", "#version 460\n" + extensions, 1) + "#line 1\n" + body
    with tempfile.NamedTemporaryFile("w", suffix=f".{stage}", delete=False) as f:
        f.write(full)
        tmp = f.name
    try:
        res = subprocess.run([validator, "-S", stage, tmp], capture_output=True, text=True, timeout=60)
    finally:
        Path(tmp).unlink(missing_ok=True)
    out = (res.stdout + res.stderr).replace(tmp, str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))
    lines = [ln for ln in out.splitlines() if ln.strip() and not ln.strip().endswith(path.name)]
    return res.returncode == 0, "\n".join(lines)


def targets(args: list[str]) -> list[Path]:
    if args:
        return [Path(a).resolve() for a in args]
    return [p for p in sorted(SHADERS.rglob("*.glsl")) if re.search(r"\bvoid\s+main\s*\(", p.read_text(encoding="utf-8"))]


def main(argv: list[str]) -> int:
    validator = shutil.which("glslangValidator")
    if validator is None:
        print("glslangValidator not found (brew install glslang)", file=sys.stderr)
        return 2
    failed = 0
    for path in targets(argv):
        ok, log = check(path, validator)
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        if ok:
            print(f"ok    {rel}" + (" (skipped)" if log == "skipped" else ""))
        else:
            failed += 1
            print(f"FAIL  {rel}\n" + "\n".join("      " + ln for ln in log.splitlines()))
    print(f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
