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

## Phase plan
1. Done: clock, bus, baker, cache reader, GPU probe, 2D smoke/fire.
2. 3D smoke/fire real-time: 3D bus, 128^3 solver, raymarched volume render, geometry -> 3D SDF obstacles, multigrid
   (or red-black GS + multigrid) pressure, 3D caches as half-float slice atlases.
3. GPU particles: millions, advected by bus velocity (2D+3D), SDF collisions, emitted from sims, two-way via
   additive-blend splats into bus sources/forces; particle caches.
4. High-res baked smoke/fire: 256^3+ bakes, real-time detail upres on playback, cache size reduction (field
   selection, half precision, crop, async save).
5. Destruction: Voronoi fracture in pure Python/numpy, Bullet Solver COMP if usable on macOS else own GPU rigid
   bodies, triggers, pieces as moving bus obstacles, debris particles and dust; baked many-piece version.
6. FLIP liquids (bake + real-time 2D preview): GPU binning + gather P2G, multigrid pressure, surface (marching cubes
   or screen-space), interaction with obstacles/destruction/smoke.
7. Cloth: XPBD on GPU, bus wind, SDF collision, tearing, optional two-way air coupling; baked high-res.
8. Integration: scene/preset system, cross-camp showcase scenes, performance pass, render/export outputs, docs.
