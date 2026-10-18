"""Fracture, rigid bodies, and the bus obstacles those pieces publish.

The numpy solver in rigidbody.py is the one that runs here. Bullet's result on this machine is recorded
in AGENTS.md. Body state is a 256² rgba32float EXR (four quadrants), which is also the baker field,
so a step ends by quantizing to float32 and the next step reads those bits.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

GRID_DEFAULT = 16
BIN_MAX = 8
BIN_STRIDE = 1 + (BIN_MAX + 3) // 4  # count texel plus 4 ids per texel


def _ensure_python_path(comp) -> None:
    path = str(Path(comp.parent.sims.par.Root.eval()) / "python")
    if path not in sys.path:
        sys.path.insert(0, path)


class Destruction:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        _ensure_python_path(ownerComp)
        self.world = None
        self.pieces = []
        self._snap = None
        self._use_snap = True
        self._mesh_gen = 0
        self._mesh_cache = []
        self._broke_frame = False
        self._spawns = []
        self._bursts = []
        self._armed = False

    # -- build ------------------------------------------------------------------------------------------------
    def _settings(self) -> dict:
        p = self.ownerComp.par
        return {
            "count": int(p.Pieces.eval()),
            "seed": int(p.Seed.eval()),
            "pattern": str(p.Pattern.eval()),
            "shape": str(p.Shape.eval()),
            "center": (float(p.Centerx.eval()), float(p.Centery.eval()), float(p.Centerz.eval())),
            "half": (float(p.Halfx.eval()), float(p.Halfy.eval()), float(p.Halfz.eval())),
            "sigma": float(p.Sigma.eval()),
            "glue": float(p.Glue.eval()),
            "gravity": float(p.Gravity.eval()),
            "iterations": int(p.Iterations.eval()),
            "positers": int(p.Positers.eval()),
            "mesh_contacts": bool(p.Meshcontacts.eval()),
            "projectile": bool(p.Projectile.eval()),
            "mode": str(p.Breakmode.eval()),
            "mesh": str(p.Meshpath.eval()),
        }

    def _solid(self):
        import fracture  # noqa: PLC0415

        s = self._settings()
        if s["shape"] == "sphere":
            return fracture.sphere_solid(s["center"], radius=float(min(s["half"])), subdiv=1)
        if s["shape"] == "mesh":
            return self._solid_from_sop(s["mesh"])
        return fracture.box_solid(s["center"], s["half"])

    def _solid_from_sop(self, path: str):
        import fracture  # noqa: PLC0415

        sop = op(path)  # noqa: F821
        verts = [(float(pt.x), float(pt.y), float(pt.z)) for pt in sop.points]
        faces = []
        for prim in sop.prims:
            faces.append(tuple(int(v.point.index) for v in prim))
        return fracture.solid_from_mesh(verts, faces)

    def Build(self) -> dict:
        import importlib  # noqa: PLC0415

        import fracture  # noqa: PLC0415
        import rigidbody  # noqa: PLC0415

        importlib.reload(fracture)
        importlib.reload(rigidbody)

        s = self._settings()
        solid = self._solid()
        impact = (s["center"][0] - s["half"][0], s["center"][1], s["center"][2])
        pieces = fracture.fracture(
            solid, s["count"], seed=s["seed"], pattern=s["pattern"], impact=impact, sigma=s["sigma"], density=200.0,
        )
        world = rigidbody.World(
            gravity=(0.0, -s["gravity"], 0.0), iterations=s["iterations"], position_iters=s["positers"],
            friction=0.7, restitution=0.05, sleep_after=8, mesh_contacts=s["mesh_contacts"],
        )
        for piece in pieces:
            world.add_piece(piece)
        for i, j, point, normal in fracture.neighboring_pairs(pieces):
            world.add_glue(i, j, point, normal, s["glue"])
        if s["projectile"]:
            world.add_sphere(
                (s["center"][0] - s["half"][0] - 0.55, s["center"][1], s["center"][2]),
                radius=0.07, mass=4.0, velocity=(7.0, 0.2, 0.0),
            )
        self.pieces = pieces
        self.world = world
        self._snap = world.snapshot()
        self._use_snap = True
        self._broke_frame = False
        self._spawns = []
        self._bursts = []
        self.ownerComp.par.Bodycount = len(world.bodies)
        self._cache_mesh()
        self._mesh_gen += 1
        self._upload()
        self._probe_flip()
        mesh = self.ownerComp.op("pieces_geo/mesh")
        if mesh is not None:
            mesh.lock = False
            mesh.cook(force=True)
            mesh.lock = True
        n_glue = sum(1 for g in world.glues if not g.broken)
        self.ownerComp.par.Status = f"built {len(pieces)} pieces, {n_glue} glues"
        return {"pieces": len(pieces), "glues": n_glue, "volume": float(sum(p.volume for p in pieces)),
                "source": float(solid.volume())}

    def _cache_mesh(self) -> None:
        cached = []
        for index, piece in enumerate(self.pieces):
            tris = piece.triangles - piece.com
            flags = piece.interior
            cached.append((index, np.asarray(tris, np.float32), np.asarray(flags)))
        self._mesh_cache = cached

    def FillMesh(self, sop) -> None:
        gen = int(self._mesh_gen)
        if sop.storage.get("gen") == gen and sop.numPoints > 0:
            return
        sop.clear()
        sop.pointAttribs.create("N")
        sop.pointAttribs.create("Cd")
        for index, tris, flags in self._mesh_cache:
            for tri, interior in zip(tris, flags):
                pts = []
                normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                length = float(np.linalg.norm(normal))
                normal = normal / length if length > 1e-8 else np.array([0.0, 1.0, 0.0])
                for vertex in tri:
                    pt = sop.appendPoint()
                    pt.P = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
                    pt.N = (float(normal[0]), float(normal[1]), float(normal[2]))
                    pt.Cd = (float(index), 1.0 if interior else 0.0, 0.0)
                    pts.append(pt)
                poly = sop.appendPoly(3, closed=True, addPoints=False)
                for k, pt in enumerate(pts):
                    poly[k].point = pt
        sop.storage["gen"] = gen

    def ApplyRealtime(self) -> None:
        """The scene the realtime measurement runs: 64³, four V-cycles, 480×270, sphere contacts.

        A 128³ grid with mesh contacts does not hold 30 fps once the pieces, the volume, and the
        debris are all drawing. Four V-cycles with 8 smooth sweeps fit at 64³ and bring the
        divergence under the phase 4 Jacobi residual. Piece contacts are bounding spheres; the
        mesh path stays in the unit tests. The picture is 480×270 with 6 ray steps.
        """
        bus = op.simbus3d  # noqa: F821
        bus.par.Resw = 64
        bus.par.Resh = 64
        bus.par.Resd = 64
        fluid = op("/sims/fluid3d")  # noqa: F821
        fluid.par.Solver = "multigrid"
        fluid.par.Vcycles = 4
        fluid.par.Smooth = 8
        fluid.par.Coarse = 16
        fluid.par.Coarsegain = 1.0
        fluid.par.Raysteps = 6
        fluid.par.Shadowsteps = 2
        fluid.par.Renderw = 480
        fluid.par.Renderh = 270
        fluid.par.Cookrender = True
        fluid.par.Maccormack = 0
        for name in ("render_out", "shadow"):
            node = fluid.op(name)
            if node is not None:
                node.bypass = False
        p = self.ownerComp.par
        p.Cookrender = True
        p.Renderw = 480
        p.Renderh = 270
        p.Iterations = 2
        p.Positers = 1
        p.Meshcontacts = False

    def Arm(self) -> dict:
        self.ApplyRealtime()
        info = self.Build()
        self._set_bypass(False)
        bus3 = op.simbus3d  # noqa: F821
        bus2 = op.simbus  # noqa: F821
        bus3.Publish("obstacles", self.ownerComp.op("obstacles3d"))
        bus3.Publish("sources", self.ownerComp.op("dust3d"))
        bus2.Publish("obstacles", self.ownerComp.op("obstacles2d"))
        parts = op("/sims/particles")  # noqa: F821
        parts.par.Bus = "3d"
        parts.par.Emitter = "tracer"
        parts.par.Velmode = "bus"
        parts.par.Resx = 256
        parts.par.Resy = 256
        parts.par.Couplingstrength = 0
        parts.par.Gravity = 2.0
        parts.par.Drag = 0.4
        parts.par.Lifemin = 1.5
        parts.par.Lifemax = 2.5
        parts.par.Cookrender = False
        clock = op.simclock  # noqa: F821
        clock.par.Sims = "/sims/destruction /sims/fluid3d /sims/particles"
        clock.par.Substeps = 1
        self._armed = True
        self._clear_bursts()
        return info

    def ArmMover(self) -> None:
        """One kinematic box with a constant velocity, for the obstacle-coupling check."""
        import importlib  # noqa: PLC0415

        import rigidbody  # noqa: PLC0415

        importlib.reload(rigidbody)
        world = rigidbody.World(gravity=(0.0, 0.0, 0.0), iterations=1, position_iters=1)
        world.add_box((0.0, 1.0, 0.0), (0.18, 0.18, 0.18), mass=1.0, velocity=(0.6, 0.0, 0.0), kind="kinematic")
        self.world = world
        self.pieces = []
        self._mesh_cache = []
        self._snap = world.snapshot()
        self._use_snap = True
        self.ownerComp.par.Bodycount = 1
        self.ownerComp.par.Pieces = 1
        self._upload()
        self._probe_flip()
        self._set_bypass(False)
        op.simbus3d.Publish("obstacles", self.ownerComp.op("obstacles3d"))  # noqa: F821
        self._armed = True

    def Release(self) -> None:
        bus3 = op.simbus3d  # noqa: F821
        bus2 = op.simbus  # noqa: F821
        for name, bus in (("obstacles3d", bus3), ("dust3d", bus3), ("obstacles2d", bus2)):
            top = self.ownerComp.op(name)
            if top is not None:
                bus.Unpublish(top.path)
        self._set_bypass(True)
        self._armed = False
        self.ownerComp.par.Status = "released"

    def _set_bypass(self, bypass: bool) -> None:
        for name in ("obstacles3d", "obstacles2d", "dust3d", "pieces_render", "composite"):
            node = self.ownerComp.op(name)
            if node is not None:
                node.bypass = bypass

    # -- textures ---------------------------------------------------------------------------------------------
    def _upload(self) -> None:
        world = self.world
        if world is None:
            return
        bodies = self.ownerComp.op("bodies")
        extents = self.ownerComp.op("extents")
        bins = self.ownerComp.op("bins")
        state = world.pack_state()
        ext = np.zeros_like(state)
        for i, body in enumerate(world.bodies):
            x, y = i % 128, i // 128
            if y >= 128:
                break
            if len(body.vertices):
                mn, mx = body.vertices.min(0), body.vertices.max(0)
                center = 0.5 * (mn + mx)
                half = np.maximum(0.5 * (mx - mn), 1e-4)
            else:
                center = np.zeros(3)
                half = np.array([body.radius, body.radius, body.radius])
            ext[y, x, :3] = half
            ext[y, x + 128, :3] = center
        self._copy(bodies, state)
        self._copy(extents, ext)
        self._copy(bins, self._bins())
        count = self.ownerComp.par.Pieces
        # The uniform is the piece count. Bodies include the projectile, so the shader cap is the body count.
        # uCount.x is read as Pieces; store the real body count on the same parameter only for the shader
        # by writing it into a constant channel of the bin header instead. The shader also clamps by id.

    def _copy(self, top, image: np.ndarray) -> None:
        top.par.resolutionw = int(image.shape[1])
        top.par.resolutionh = int(image.shape[0])
        top.lock = True
        top.copyNumpyArray(np.ascontiguousarray(image, dtype=np.float32))

    def _bins(self) -> np.ndarray:
        import numpy as np  # noqa: PLC0415

        grid = int(self.ownerComp.par.Grid.eval())
        bmin, bmax = self._domain()
        size = np.maximum(bmax - bmin, 1e-4)
        image = np.zeros((64, 256, 4), np.float32)
        counts = np.zeros((grid, grid, grid), np.int32)
        ids = np.full((grid, grid, grid, BIN_MAX), -1, np.int32)
        for i, body in enumerate(self.world.bodies):
            self.world._refresh_aabb(body)
            lo = np.clip(np.floor((body.aabb_min - bmin) / size * grid).astype(int), 0, grid - 1)
            hi = np.clip(np.floor((body.aabb_max - bmin) / size * grid).astype(int), 0, grid - 1)
            for z in range(int(lo[2]), int(hi[2]) + 1):
                for y in range(int(lo[1]), int(hi[1]) + 1):
                    for x in range(int(lo[0]), int(hi[0]) + 1):
                        slot = int(counts[x, y, z])
                        if slot < BIN_MAX:
                            ids[x, y, z, slot] = i
                        counts[x, y, z] = slot + 1
        # bin id is x + grid * (y + grid * z). Transpose so a ravel matches that order.
        packed = np.zeros((grid * grid * grid, BIN_STRIDE, 4), np.float32)
        cz = np.transpose(counts, (2, 1, 0)).ravel()
        iz = np.transpose(ids, (2, 1, 0, 3)).reshape(-1, BIN_MAX)
        packed[:, 0, 0] = np.minimum(cz, BIN_MAX)
        packed[:, 1, :4] = iz[:, :4]
        if BIN_MAX > 4:
            packed[:, 2, :4] = iz[:, 4:8]
        flat = image.reshape(-1, 4)
        block = packed.reshape(-1, 4)
        flat[:block.shape[0]] = block
        return image

    def _write_bin(self, image, start, count, id_row) -> None:
        h, w = image.shape[:2]

        def put(linear, value):
            if 0 <= linear < w * h:
                image[linear // w, linear % w] = value

        put(start, (count, 0, 0, 0))
        for i in range(0, BIN_MAX, 4):
            put(start + 1 + i // 4, (
                float(id_row[i]), float(id_row[i + 1]) if i + 1 < BIN_MAX else -1,
                float(id_row[i + 2]) if i + 2 < BIN_MAX else -1,
                float(id_row[i + 3]) if i + 3 < BIN_MAX else -1,
            ))

    def _domain(self):
        bus = op.simbus3d  # noqa: F821
        b = bus.Bounds
        return (np.array([b[0], b[1], b[2]], np.float64), np.array([b[3], b[4], b[5]], np.float64))

    def _probe_flip(self) -> None:
        """texelFetch and copyNumpyArray have disagreed on row 0. A box centre decides which way.

        Try both orientations. A probe that only flips when the current one misses will stick on the
        wrong row after the parameter was left set by an earlier scene.
        """
        if not self.world or not self.world.bodies:
            return
        body = self.world.bodies[0]
        node = self.ownerComp.op("obstacles3d")
        node.bypass = False
        best = None
        for flip in (False, True):
            self.ownerComp.par.Rowflip = flip
            node.cook(force=True)
            inside = self._sdf_at(body.pos)
            if inside is None:
                continue
            if best is None or inside < best[0]:
                best = (inside, flip)
        if best is None:
            return
        self.ownerComp.par.Rowflip = best[1]
        node.cook(force=True)
        if not self._armed:
            node.bypass = True
        self.ownerComp.par.Status = f"flip {int(best[1])} sdf {best[0]}"

    def _sdf_at(self, world_point) -> float | None:
        obs = self.ownerComp.op("obstacles3d")
        arr = obs.numpyArray(delayed=False)
        if arr is None:
            return None
        bus = op.simbus3d  # noqa: F821
        b, res = bus.Bounds, bus.Res
        uv = [(float(world_point[i]) - b[i]) / (b[i + 3] - b[i]) for i in range(3)]
        x = int(np.clip(round(uv[0] * res[0] - 0.5), 0, res[0] - 1))
        y = int(np.clip(round(uv[1] * res[1] - 0.5), 0, res[1] - 1))
        z = int(np.clip(round(uv[2] * res[2] - 0.5), 0, res[2] - 1))
        return float(arr[z, y, x, 0])

    def _clear_bursts(self) -> None:
        bursts = self.ownerComp.op("bursts")
        if bursts is not None:
            self._copy(bursts, np.zeros((64, 64, 4), np.float32))
        parts = op("/sims/particles")  # noqa: F821
        h = int(parts.par.Resy.eval())
        w = int(parts.par.Resx.eval())
        black = np.zeros((h, w, 4), np.float32)
        for name in ("burst_pos", "burst_vel"):
            top = parts.op(name)
            if top is not None:
                self._copy(top, black)

    # -- clock ------------------------------------------------------------------------------------------------
    def BeginStep(self, reset: bool) -> None:
        if self.world is None:
            self.Build()
        if reset and self._use_snap and self._snap is not None:
            self.world.restore(self._snap)
            self._broke_frame = False
            self._bursts = []
            self._spawns = []
            # The dust texture is not in the rigid snapshot. A second bake has to start from a
            # clear puff, or the smoke left by the previous run is still a source on frame 1.
            self._clear_bursts()
        show = bool(self.ownerComp.par.Cookrender.eval()) and self._armed
        render = self.ownerComp.op("pieces_render")
        comp = self.ownerComp.op("composite")
        if render is not None:
            render.bypass = not show
        if comp is not None:
            comp.bypass = not show

    def Step(self) -> None:
        if self.world is None:
            return
        p = self.ownerComp.par
        clock = op.simclock  # noqa: F821
        dt = float(clock.par.Dt.eval())
        self._triggers(dt)
        self._fluid_drag(dt)
        self.world.step(dt, quantize=True)
        self._upload()
        self._emit(bool(self._reset_this_step()))
        if p.Cookrender.eval() and self._armed:
            # The composite samples the volume. Cooking it here raymarches before the fluid
            # step, and the fluid step raymarches again. Pieces are drawn here; the composite
            # is drawn at the end of the fluid step.
            self.ownerComp.op("pieces_render").cook(force=True)

    def _reset_this_step(self) -> bool:
        # The particle shader reads its init texture on the reset tick and the live snapshot otherwise.
        return bool(self._use_snap) and int(op.simclock.par.Step.eval()) == 0  # noqa: F821

    def _triggers(self, dt: float) -> None:
        mode = str(self.ownerComp.par.Breakmode.eval())
        frame = int(op.simclock.par.Frame.eval())  # noqa: F821
        if mode == "frame" and frame >= int(self.ownerComp.par.Breakframe.eval()) and not self._broke_frame:
            self._broke_frame = True
            self.world.break_glue()
            self._boom()
        elif mode == "explosion" and frame >= int(self.ownerComp.par.Breakframe.eval()) and not self._broke_frame:
            self._broke_frame = True
            self.world.break_glue()
            self._boom()
        elif mode == "pressure":
            self._pressure_break()
        newly = [g for g in self.world.glues if g.broken and g.last_impulse > 0 and not getattr(g, "_emitted", False)]
        # Glue records the impulse that broke it. Emit once.
        for glue in self.world.glues:
            if glue.broken and not getattr(glue, "_emitted", False):
                glue._emitted = True
                a = self.world.bodies[glue.a]
                b = self.world.bodies[glue.b]
                point = 0.5 * (a.pos + b.pos)
                normal = b.pos - a.pos
                self._queue_break(point, normal, a.lin)

    def _boom(self) -> None:
        p = self.ownerComp.par
        origin = (float(p.Explodex.eval()), float(p.Explodey.eval()), float(p.Explodez.eval()))
        self.world.explosion(origin, float(p.Exploderadius.eval()), float(p.Explodestrength.eval()))
        self._queue_break(np.array(origin), np.array([0.0, 1.0, 0.0]), np.zeros(3))

    def _pressure_break(self) -> None:
        thresh = float(self.ownerComp.par.Pressurethresh.eval())
        if thresh <= 0 or self.world is None:
            return
        try:
            pressure = op("/sims/fluid3d/pressure_out").numpyArray(delayed=False)  # noqa: F821
        except Exception:
            return
        if pressure is None:
            return
        bus = op.simbus3d  # noqa: F821
        b, res = bus.Bounds, bus.Res
        for glue in self.world.glues:
            if glue.broken:
                continue
            point = 0.5 * (self.world.bodies[glue.a].pos + self.world.bodies[glue.b].pos)
            uv = [(float(point[i]) - b[i]) / (b[i + 3] - b[i]) for i in range(3)]
            if any(v < 0 or v > 1 for v in uv):
                continue
            x = int(np.clip(uv[0] * res[0], 0, res[0] - 1))
            y = int(np.clip(uv[1] * res[1], 0, res[1] - 1))
            z = int(np.clip(uv[2] * res[2], 0, res[2] - 1))
            if float(pressure[z, y, x, 0]) > thresh:
                glue.broken = True

    def _fluid_drag(self, dt: float) -> None:
        drag = float(self.ownerComp.par.Fluiddrag.eval())
        if drag <= 0 or self.world is None:
            return
        try:
            vel = op("/sims/fluid3d/vel_out").numpyArray(delayed=False)  # noqa: F821
        except Exception:
            return
        if vel is None:
            return
        bus = op.simbus3d  # noqa: F821
        b, res = bus.Bounds, bus.Res
        for body in self.world.bodies:
            if body.inv_mass <= 0 or body.mass > 0.4:
                continue
            uv = [(float(body.pos[i]) - b[i]) / (b[i + 3] - b[i]) for i in range(3)]
            if any(v < 0 or v > 1 for v in uv):
                continue
            x = int(np.clip(uv[0] * res[0], 0, res[0] - 1))
            y = int(np.clip(uv[1] * res[1], 0, res[1] - 1))
            z = int(np.clip(uv[2] * res[2], 0, res[2] - 1))
            fluid = np.asarray(vel[z, y, x, :3], np.float64)
            body.lin = body.lin + drag * (fluid - body.lin) * dt

    def _queue_break(self, point, normal, velocity) -> None:
        point = np.asarray(point, np.float64)
        normal = np.asarray(normal, np.float64)
        length = float(np.linalg.norm(normal))
        normal = normal / length if length > 1e-8 else np.array([0.0, 1.0, 0.0])
        self._bursts.append((point.copy(), 1.2))
        rng = np.random.Generator(np.random.PCG64(len(self._spawns) + 17))
        for k in range(12):
            jitter = rng.normal(size=3) * 0.02
            speed = 0.4 + 0.8 * float(rng.random())
            self._spawns.append((point + normal * 0.01 + jitter, np.asarray(velocity, np.float64) + normal * speed))
        self._write_dust()

    def _write_dust(self) -> None:
        image = np.zeros((64, 64, 4), np.float32)
        p = self.ownerComp.par
        # Strength decays so a break is a puff, not a permanent source.
        kept = []
        for point, strength in self._bursts[-int(p.Dustcount.eval()):]:
            strength *= 0.92
            if strength > 0.02:
                kept.append((point, strength))
        self._bursts = kept
        for i, (point, strength) in enumerate(kept[:64]):
            image[:, i, 0] = point[0]
            image[:, i, 1] = point[1]
            image[:, i, 2] = point[2]
            image[:, i, 3] = strength
        self._copy(self.ownerComp.op("bursts"), image)

    def _emit(self, reset: bool) -> None:
        parts = op("/sims/particles")  # noqa: F821
        h = int(parts.par.Resy.eval())
        w = int(parts.par.Resx.eval())
        pos = np.zeros((h, w, 4), np.float32)
        vel = np.zeros((h, w, 4), np.float32)
        if self._spawns:
            # Tracer init leaves every slot but 0 dead, so the first rows are free.
            n = min(len(self._spawns), h * w - 1)
            for i, (point, speed) in enumerate(self._spawns[:n]):
                x = (i + 1) % w
                y = (i + 1) // w
                if y >= h:
                    break
                pos[y, x, :3] = np.asarray(point, np.float32)
                pos[y, x, 3] = 1.4
                vel[y, x, :3] = np.asarray(speed, np.float32)
            self._spawns = []
        for name, image in (("burst_pos", pos), ("burst_vel", vel)):
            top = parts.op(name)
            if top is not None:
                self._copy(top, image)

    def _particle_life(self, reset: bool):
        parts = op("/sims/particles")  # noqa: F821
        top = parts.op("init_pvel") if reset else parts.op("pvel_out")
        try:
            arr = top.numpyArray(delayed=False)
        except Exception:
            arr = None
        if arr is None:
            return np.zeros((int(parts.par.Resy.eval()), int(parts.par.Resx.eval())), np.float32)
        return arr[..., 3]

    def _dead_slots(self, life, n):
        slots = []
        h, w = life.shape
        for y in range(h):
            row = life[y]
            for x in range(w):
                if row[x] <= 0.0 and not (x == 0 and y == 0):
                    slots.append((x, y))
                    if len(slots) >= n:
                        return slots
        return slots

    # -- baker ------------------------------------------------------------------------------------------------
    def StateTops(self) -> dict:
        return {"bodies": self.ownerComp.op("bodies")}

    def StateFormats(self) -> dict:
        return {"bodies": "rgba32float"}

    def ExportArray(self, field: str):
        """Float32 body state. Movie File In on this machine does not return a 32-bit EXR saved from a
        script TOP, so the baker also writes this array and SetInit prefers it."""
        if field != "bodies" or self.world is None:
            return None
        return self.world.pack_state()

    def SetInit(self, paths: dict | None = None) -> None:
        if self.world is None:
            self.Build()
        if not paths:
            self._use_snap = True
            return
        npy = Path(str(paths["bodies"]))
        if not npy.is_absolute():
            npy = Path(self.ownerComp.parent.sims.par.Root.eval()) / npy
        npy = npy.with_suffix(".npy")
        if npy.is_file():
            image = np.load(npy)
            self.world.load_state(image)
            self._use_snap = False
            self._upload()
            return
        load = self.ownerComp.op("load_bodies")
        load.par.file = str(paths["bodies"])
        load.par.playmode = "locked"
        load.par.index.mode = ParMode.CONSTANT  # noqa: F821
        load.par.index = 0
        load.par.outputresolution = "custom"
        load.par.resolutionw = 256
        load.par.resolutionh = 256
        load.par.reloadpulse.pulse()
        load.cook(force=True)
        image = np.array(load.numpyArray(delayed=False), dtype=np.float32, copy=True)
        self.world.load_state(image)
        self._use_snap = False
        self._upload()

    def Params(self) -> dict:
        p = self.ownerComp.par
        bus = op.simbus3d  # noqa: F821
        clock = op.simclock  # noqa: F821
        shaders = sorted((d.name, d.text) for d in self.ownerComp.findChildren(type=textDAT, depth=1))  # noqa: F821
        digest = hashlib.sha256(repr(shaders).encode("utf-8")).hexdigest()[:16]
        s = self._settings()
        s["gravity"] = float(p.Gravity.eval())
        return {
            "sim": "destruction",
            "settings": s,
            "res": list(bus.Res),
            "bounds": list(bus.Bounds),
            "fps": clock.par.Fps.eval(),
            "substeps": int(clock.par.Substeps.eval()),
            "seed": int(clock.par.Seed.eval()),
            "shaders": digest,
        }

    def SdfAt(self, point) -> float:
        node = self.ownerComp.op("obstacles3d")
        node.bypass = False
        node.cook(force=True)
        return self._sdf_at(point)
