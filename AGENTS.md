# AGENTS.md (td-simulations)

Rules and accumulated knowledge for AI agents working in this repo. Human docs: README.md.

## What this is
Real-time (>= 30 fps) and baked (cached) simulations for TouchDesigner 2025.32820 on macOS (Apple M4 Max, Vulkan over
Metal): smoke/fire, particles, destruction, liquids, cloth. Every simulation reads and writes a shared field bus so
simulations interact. Network is built from repo sources by `python/build_network.py`; shaders and extensions are Text
DATs file-synced to the repo.

## Commands
- Tests (no TD): `.venv/bin/python -m pytest -q` (must pass before every commit).
- Scratch TD: `.venv/bin/python scripts/td_scratch.py start|status|run <file|->|stop`. From Python: `attach()` then
  `td.run(code)` / `td.run_json(code)` (TD code prints `@@JSON@@<json>`).
- Build + save: `.venv/bin/python scripts/build_project.py [--no-save]` (saves sims.toe and tox/sims.tox).
- Shader compile check (no TD): `.venv/bin/python scripts/check_shaders.py [files]`.
- Checks: `scripts/bake_check.py`, `scripts/bench.py`, `scripts/sim_stats.py`, `scripts/probe_gpu.py`, and one
  `scripts/validate_phaseN.py` per phase (exit 0 = pass, writes `reports/phaseN.json`).

## Hard rules
- Only ever drive the scratch TD from `.scratch/state.json`. Never touch, save, close or kill any other TouchDesigner
  process (the user runs several, including one on port 8765). Only kill processes you started.
- Never `sleep` inside TD code (blocks the main thread). Wait from outside. Wrap TD calls in timeouts.
- Never print, log or commit tokens (`.scratch/connection.json`, the td_python_debug Token parameter).
- No machine-specific absolute paths in sources or docs. Paths resolve from `/sims` Root (default `project.folder`).
- Cache budget: at most 30 GB under `cache/` at any time; delete validation caches when a phase is validated
  (keep one small showcase cache per phase if under budget). `cache/` is gitignored; never commit caches.
- Do not commit or push unless the orchestrator asked you to. Stage explicit paths, never `git add -A` blindly.
- Keep phase 1 behaviour working: `scripts/bake_check.py --frames 40` must still print ALL OK after your changes.
- Compile shaders offline before TD sees them: `.venv/bin/python scripts/check_shaders.py [files]` (glslangValidator
  with stub TD built-ins; no TD needed). Fix every error there first; a broken shader in TD pops dialogs on the
  user's screen. Includes resolve `#include <name_glsl>` to `shaders/**/name.glsl`, so name DATs to match. If you
  rely on a TD built-in the stub lacks, add it to the preamble in `check_shaders.py` rather than skipping the file.

## TouchDesigner lessons (verified on this machine)
- GLSL TOP: max 3 wired inputs; further inputs via the `tops` parameter (absolute paths, appended in order). A GLSL TOP
  with zero inputs breaks `sTD2DInputs`; give merges a wired zero input. `TD_NUM_2D_INPUTS` works.
- GLSL TOP supports compute mode, `type` texture3d with custom depth, `outputaccess` readwrite, `npasses` (pass k reads
  pass k-1 through input 0; `uTDPass` set). `#include <datname>` resolves a sibling DAT. Compile errors via `op.errors()`.
- Supported: compute, 3D images (`TDImageStoreOutput`, `sTD3DInputs`, `texelFetch`), shared-memory int atomics.
  NOT supported: float image atomics (GL_EXT_shader_atomic_float); TD atomic counters did not compile. Accumulate by
  additive-blend rendering or gather, not atomics.
- Feedback TOP captures its target only on frames it cooks; cook it explicitly every step. Reset loops with a Switch
  (index 1 = init) set at frame start, not the Feedback reset pulse.
- Cook explicitly at frame start (clock Tick) for exactly one step per frame. Ops cooked once in a frame do not recook
  unless forced. Reading a solver output must not recook the solver: public outputs are Cache TOP snapshots made
  inactive after each step (see Fluid2D.Step).
- `TOP.save(path)` writes EXR; float32/16 round-trips are bit-exact. Movie File In cannot open EXRs only a few rows
  tall (fine >= 96 px). Movie File In loads synchronously on cook.
- Pure open domain boundaries were unstable on the collocated grid: sides are walls in the pressure solve, "open"
  sides get an absorbing band.
- Python in TD: 3.11, numpy 2.1 available, no scipy.
- Feedback TOP does not capture a 3D texture. With a 3D target it passes its wired input through and the value
  does not accumulate across cooks. Persistent 3D state is a pair of Cache TOPs. An inactive Cache TOP keeps the
  3D image; a second write requires `replace` on and `replaceindex` 0, otherwise the first frame sticks.
- 3D `numpyArray` shape is `(depth, height, width, channels)`, index `[z, y, x]`. `uTDOutputInfo.res` is the 2D
  layout even for a volume: xy = 1/width, 1/height and zw = width, height. Depth is `uTDOutputInfo.depth`
  (x = 1/depth, y = depth).
- A 2D TOP wired into a 3D compute shader is `sTD2DInputs`, not `sTD3DInputs`. `TD_NUM_3D_INPUTS` counts only
  volumes. `npasses` on a 3D compute TOP feeds the previous pass through input 0. Read-write storage does not
  survive from one cook to the next.
- Constant TOP's type menu lists `texture3d` but exposes no depth parameter. A one-line compute shader is the
  clear. `allowCooking` can be turned off on COMPs only, not on TOPs.
- rgba16float EXRs need both edges ≥ 256 to open in Movie File In (256×128 failed, 256×256 loaded). 3D caches
  are 2D slice atlases padded to that edge. `poptoSOP` turns a POP into polygons the SOP API can triangulate.
- A long `execute_python` (tens of seconds of cooks in one call) has killed the scratch TD. Split the work and
  wait outside. Do not `sleep` inside TD.
- A uniform expression that changes every step (`uClock` especially) dirties that 3D TOP. An inactive Cache TOP
  still recooks a dirty input, so the mesh jump flood ran every frame (~90 ms at 128³) while its zero-volume
  input carried the clock, and the sim stayed under 30 fps. Static volumes bind those uniforms to constants.
  The mesh SDF is voxelized once into `sdf_hold` and rigid-transformed after that.
- The same pull means a later script can recook the 3D solver and move the live snapshot off the frame just
  saved. A byte-exact replay compare has to run in the same `execute_python` as the `Tick` that saved it.
- Movie File In's default index is `me.time.frame`, which rewrites the frame number inside an EXR path.
  Cache-reader movies are created with `playmode` locked so they open the path they are given.
- A GLSL TOP vector-uniform sequence cannot be set to 0 blocks (the minimum is 1).

## Phase plan
1. Done: clock, bus, baker, cache reader, GPU probe, 2D smoke/fire.
2. Done: 3D bus, 128³ fluid, raymarched volume, analytic and mesh SDFs, damped multigrid pressure, slice-atlas caches.
3. GPU particles: millions, advected by bus velocity (2D+3D), SDF collisions, emitted from sims, two-way via
   additive-blend splats into bus sources/forces; particle caches.
4. High-res baked smoke/fire: 256^3+ bakes, real-time detail upres on playback, cache size reduction (field
   selection, half precision, crop, async save). Must fix the pressure solve first: the Phase 2 multigrid (coarse
   gain 0.05) and Jacobi 12 each remove only ~10-15% of divergence RMS per projection, which is not incompressible.
   Target ≥10× RMS reduction per projection at 256³ (e.g. a proper V-cycle with consistent restriction/prolongation
   or a MAC-staggered grid), measured as divergence reduction per GPU millisecond against Jacobi.
5. Destruction: Voronoi fracture in pure Python/numpy, Bullet Solver COMP if usable on macOS else own GPU rigid
   bodies, triggers, pieces as moving bus obstacles, debris particles and dust; baked many-piece version.
6. FLIP liquids (bake + real-time 2D preview): GPU binning + gather P2G, multigrid pressure, surface (marching cubes
   or screen-space), interaction with obstacles/destruction/smoke.
7. Cloth: XPBD on GPU, bus wind, SDF collision, tearing, optional two-way air coupling; baked high-res.
8. Integration: scene/preset system, cross-camp showcase scenes, performance pass, render/export outputs, docs.
