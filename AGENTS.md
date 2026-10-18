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
- Pure open domain boundaries were unstable on the 2D collocated grid: those sides stay walls in the pressure
  solve and get an absorbing band. In 3D an open side is a Dirichlet pressure boundary (ghost pressure 0).
  All-Neumann pressure is singular once combustion adds divergence, and that leftover is a residual floor.
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
- `numpyArray(delayed=False)` inside a per-step loop is a GPU sync. MGPCG was latency-bound when every CG
  iteration read three dots back to Python (~2 ms of GPU work reported as ~3 ms of wall time, and a 28 ms
  chain ran at ~41 ms/step). Keep iterative-solver scalars on the GPU: reduce to a 1×1 texture, compute
  alpha/beta in a tiny pass, and let the saxpy sample that texture. Read the residual ratio once per step.
- The same pull means a later script can recook the 3D solver and move the live snapshot off the frame just
  saved. A byte-exact replay compare has to run in the same `execute_python` as the `Tick` that saved it.
- Movie File In's default index is `me.time.frame`, which rewrites the frame number inside an EXR path.
  Cache-reader movies are created with `playmode` locked so they open the path they are given. Every Movie File In
  that loads a cache frame needs this, including the sims' `load_*` resume loaders. Unlocked resume loaders made
  3D bake resume flaky (pressure files differed after resume, depending on the timeline frame). `outputresolution`
  `useinput` on a loader with no input keeps a stale picture: set custom resolution to the state texture, index
  constant 0, and pulse Reload. An RGBA float EXR with either edge under 256 is misread (a 128² particle state
  did not round-trip; 256² did).
- A GLSL TOP vector-uniform sequence cannot be set to 0 blocks (the minimum is 1).
- A Geometry COMP is created with a torus POP whose render flag is on. Destroy that torus or every Render TOP that includes the geometry draws it on top of the particles.
- A million point sprites should be instanced from one point (`TDInstanceID` fetches the particle texel). Building a `toptoPOP` with one point per texel was several times slower. `TDPos()` is the vertex position; a raw `P` attribute is not. `gl_PointSize` is in pixels. With antialiasing off (`aa1`) and the point snapped to a pixel centre, additive blending (`source one`, `destination one`, depth off, order-independent) sums overlapping particles exactly.
- Sampling a fluid output recooks the solver when that output's input is dirty, which the clock does every step. An inactive Cache TOP does not stop it. Particle steps therefore bypass the bus branch they are not using, so a 2D step does not pull the 3D solver. Particle state is a locked Null TOP. A Feedback TOP in that graph captured one new frame and then held it.
- A size-1 Cache TOP records one image per TouchDesigner frame. Several `Tick`s inside one script leave `vel_out` / `scalars_out` on the first of those steps, so a scripted render has to copy the solved null with the Replace Pulse (`replacespulse`) after each tick and save the picture in a later call. The Active Pulse is deferred until the script returns. Leave `replace` off after the pulse or a later read records another solver step and the bake replay diverges.
- Particle sprites and the force splat have to take their world position from the state texture. `TDPos()` is only the single instanced point, which sits at the origin and lands on the clip edge. GLSL MAT sampler names are `samplerNname` / `samplerNtop`. Parameter expressions are Python, so a C ternary in a resolution or select expression does not cook.
- A Render TOP in order-independent transparency mode does not clear uncovered float pixels, so point sprites leave large negative alpha in the background. The beauty pass uses ordinary blending with background alpha 0, then a clamp, and `beauty` is that clamped picture (`alpha >= 0`).
- The baker's Group Sims list bakes every named sim into one cache (`<cache>/<sim>/<field>/`). Empty Group Sims is the single-sim layout from phases 1 and 2. Resume calls `SetInit` on every member. The cache reader's Member parameter picks which sim to play. A group bake does not listen to the clock: `PumpOnce` ticks, then saves. The timeline execute has to be off for that stretch, or it steps frames the baker never writes. The 2D force and source grids are saved with the frame and published again before the resumed fluid step, because that step runs before the particles rewrite them.
- A particle step has to integrate position and velocity from the same previous snapshot. Writing the new position snapshot before the velocity cook makes a resumed step (still reading the loaded frame) disagree with a continuous one. Leave those solved nulls locked afterwards so a save does not integrate a second time.
- The reaction a particle writes into the bus is `Couplingstrength * mass * drag * (v_p - v_f) / (density * cell volume)`, with `Particlemass` stored in nanograms. The 2D splat is one 32² workgroup of shared integer atomics, so a million light embers stay a small acceleration. Heat uses the same mass ratio. The bus merge is selected by a parameter, so the particle step has to cook it or the fluid never sees the new grid. While advecting on the bus, position and velocity are snapped to 1e-4 (velocity biased by one fluid bin) so a resumed sample that is one quantize step off stays on the same side of the boundary. Constant-velocity tracers are not snapped.
- A particle bake leaves `SetInit` on the Movie File In. Call `SetInit(None)` before the next live step or the integrator reads the cache. The same run can leave the cache reader's `Frame` expression and the clock seed off their fluid defaults. `bake_check` and the phase 2 resume match the committed sequence only after that expression is `op.simclock.par.Frame`, Interpolate is on, and the seed is 0.
- The 3D divergence and the pressure gradient have to be adjoints of the same matrix. Central differences over two cells compose to a wide stencil; Jacobi on the compact 7-point Laplacian then barely changes the divergence you measure. The matching pair is a face flux owned by the upstream cell (obstacle velocity on a solid face, 0 on a closed face) and a forward difference of pressure. An open face uses ghost pressure 0.
- Full-weighting restriction of that residual, trilinear prolongation at gain 1, and `busCell()` at each level (so the coarse unknown sees spacing 2h) is a real V-cycle. Gain 0.05 was compensating for the stencil mismatch. Four cycles with 14 weighted-Jacobi sweeps (omega 2/3) are the bake solve. Red-black Gauss-Seidel on the read-write multipass did not converge better. A manufactured zero-mean sine mode on the numpy reference drops about 4.1× per cycle with 4 sweeps and about 9.2× per cycle with 14 (about 8.6× with a solid). On the GPU the first cycle of the live fire does most of the reduction. The old ~0.011 RMS floor was the incompatible Neumann component, not a measurement cutoff. Do not drop projections whose incoming RMS is small.
- A cropped scalar Movie File In is left on custom resolution, and its volume decode is bypassed. The next non-cropped cache (phase 2 included) has to set that movie back to native and turn the bypass off, or the scalar replay reads the atlas instead of the volume.
- The baker only saves the last substep of an output frame. Saving the earlier substep as well used the same filename and then overwrote it.
- Bullet Solver COMP exists (`bulletsolverCOMP` / `bulletsolverCHOP`) and can be created under `/sims`, but `cook()` with `play` on and gravity set does not move actors (a probe actor stayed at ty 15). Turning the timeline on from inside `execute_python` wedged the scratch process. Destruction does not use Bullet. The rigid step is numpy in `python/rigidbody.py`: sequential impulses after integration, one contact per pair, glue as a velocity-averaged cluster, a sphere sweep for the projectile. Evidence is in `reports/phase5.json` (`bullet.moved` false).
- A 32-bit EXR saved from a script TOP is not returned by Movie File In (dummy image, no error). The baker writes a `.npy` beside that EXR via `ExportArray`, and `SetInit` loads the array. Half-float fluid atlases still round-trip as EXR. Group fluid saves must be the atlas (`StateTops`), not the live 3D `*_state` null: saving the null recooks the solver and the resume decoder cannot read it.
- Do not declare `in vec3 Cd` or `in vec3 N` in a GLSL MAT. TouchDesigner injects them; a second declaration fails compile and the viewer shows the red/blue checker. A Geometry COMP is still created with a torus POP; destroy it. Lock the piece script SOP after one cook or it rebuilds the mesh every frame.
- A destruction reset that restores the rigid snapshot has to clear the dust texture too. Glue `_emitted` lives on the glue objects; leaving it set makes the next bake skip the dust puff, and the smoke field then diverges while the body state still matches.
- Piece obstacles on the bus are oriented boxes (the piece's local AABB), not the exact convex SDF. The obstacle shader walks at most 8 bins. Do not put `uClock` on that shader.
- The realtime destruction scene is 64³, four multigrid V-cycles with 8 smooth sweeps, a 480×270 volume (6 ray steps, 2 shadow steps), and bounding-sphere piece contacts. A 128³ grid plus mesh contacts does not hold 30 fps once the pieces, the volume, and the debris are drawing. Four cycles at 64³ land the divergence under the phase 4 Jacobi residual and the frame near 60 fps. The numpy tests still use mesh contacts. A projectile only hits a shard whose centre enters the projectile radius, so the bounding spheres do not shatter the wall. The 1000-piece showcase uses that broadphase at 128³, with fluid keyframes every 30 frames and cropped scalars in between, so the cache stays under 3 GB.
- FLIP particle-to-grid is a bitonic sort of `(cell id, particle index)` followed by a per-cell gather. Float addition order is then the particle index order, so a resumed bake matches. Renormalising the free-surface residual onto the coarse grid amplifies the interface and the V-cycle diverges above 16³; leaving those weights un-normalised stays stable, and a short fine-grid Jacobi polish after the V-cycle is what keeps a tank from draining into a puddle. The liquid must not sample the obstacle merge it publishes into: that phi would become a solid on the next step. A time-varying emitter left on the bus makes a replay from a saved particle state disagree with the live step, because the force depends on the clock. A Cache TOP still stores only the first image of a TouchDesigner frame, so a picture saved after several scripted ticks has to be snapped in a later call, and the baker's preview is a locked null cooked inside the step.

## Phase plan
1. Done: clock, bus, baker, cache reader, GPU probe, 2D smoke/fire.
2. Done: 3D bus, 128³ fluid, raymarched volume, analytic and mesh SDFs, damped multigrid pressure, slice-atlas caches.
3. Done: GPU particles, 1024² state textures, bus advection in 2D and 3D, SDF collisions, field emission, additive instanced splats into bus forces and sources, EXR caches with resume and exact replay.
4. Done: adjoint 7-point pressure projection, Dirichlet pressure on open sides, and a 4-cycle V-cycle
   (full-weight restriction, trilinear prolongation, gain 1, weighted Jacobi). 256³ bakes with keyframe
   resume, scalar crop, and display-time upres. Realtime stays on Jacobi. The projection ratio is the RMS
   of div(u) − expansion on every step, with no incoming-RMS cutoff.
5. Done: Voronoi fracture (numpy, volume error ~0 at 1000 pieces), numpy rigid bodies (Bullet cooks but does not
   step on this Mac), breakable glue, pieces as moving bus obstacles, debris particles and dust. Realtime wall
   is 100 pieces plus 128³ Jacobi; the group bake is 96³ and byte-identical across cancel/resume via `.npy` body
   and particle state. Bus obstacles are the piece OBBs.
6. In progress: GPU FLIP on the 3D bus. The live sort is a stable 6-digit radix (cell id, then particle index); `SortAgrees` checks it against a full bitonic sort. Free-surface pressure is MGPCG (CG on the fine grid, one symmetric V-cycle per iteration, air Dirichlet, clamp-to-edge prolongation). Each step runs 8 warm-started iterations; alpha and beta stay in 1×1 textures, and the residual ratio is read once. A still 64³ tank holds its height over 120 steps (95th percentile 0.414 m to 0.415 m, divergence RMS about 7e-5, max speed about 0.0005 m/s). The dam toe stays within 22% of the Martin & Moyce Z(T) line. 64³ realtime still misses a 30 ms frame (79 steps against 187 frames in 3 s). The 48³ preset holds 124,250 live particles with the surface on. A synced splash step was sort 1.2 ms, P2G 1.8 ms, divergence 0.8 ms, MGPCG 13.1 ms (8 iterations), G2P 1.3 ms, render 0.7 ms. At a 33 fps cook rate both validation windows were 100 steps and 100 frames. The 180-frame showcase is 128³ with 2,018,940 live particles.
7. Cloth: XPBD on GPU, bus wind, SDF collision, tearing, optional two-way air coupling; baked high-res.
8. Integration: scene/preset system, cross-camp showcase scenes, performance pass, render/export outputs, docs.
   Phase 3 debt to clear here: the deterministic 2D particle splat (`shaders/particles/splat_sum.glsl`) sums all
   particles in ONE workgroup into a 32×32 grid (coarse coupling, ~45 fps with the 2D fluid); move it to a full-res
   multi-workgroup fixed-point sum (probe integer image atomics on r32i, else per-tile shared sums + a reduce pass).
   Particle pos/vel are snapped to 1e-4 under bus advection (`step.glsl`) to hide a sub-ulp difference in the first
   resumed fluid sample; a coupled group-bake resume still differs by one snap step in a few pvel values
   (validate_phase3 tolerates ≤ 2.5e-4 in pvel only). Find the real source and make coupled resume byte-exact.
   Fluid2D also snaps vel/pressure to 1e-5 and scalars to 1e-4 every step (`q_*` / `quantize.glsl`), added for the
   same reason; test removing it once the splat is deterministic (it costs three passes and kills smoke < 5e-5).
   Phase 4 debt: playback "upres" is mostly screen grain (it also speckles obstacles) and is validated on an early,
   small plume. Replace with real detail (wavelet turbulence / advected curl-noise in the volume, masked to smoke)
   and validate on a developed showcase frame. Multigrid at 256³ costs ~360 ms per projection (2.8 steps/s bake).
   Phase 6 debt: the raymarch is the particle phi at grid resolution, so the surface is faceted. A script TOP
   does not round-trip a 3D pressure volume through texelFetch; resume reloads a 2D pack. The bake preview
   is drawn from that step's phi, not from a GLSL TOP that may already have cooked. Two-way push on rigid
   pieces is not wired; point `extra_solids` at the destruction obstacle TOP for one-way coupling. APIC is
   in `python/flip.py` only.
