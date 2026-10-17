"""Demo publishers for the 3D bus: a fire emitter, analytic primitives, and a mesh signed-distance field.

RebuildMesh reads a SOP, or a POP via poptoSOP, fan-triangulates it, and jump-floods a local SDF into sdf_hold.
The mesh obstacle publisher only samples that cache, with a rigid transform, so a moving mesh stays real-time.
"""

from __future__ import annotations

import hashlib


def _as_sop(host, geo):
    family = str(getattr(geo, "family", ""))
    if family == "POP":
        conv = host.op("pop_to_sop")
        conv.par.pop = geo.path
        conv.cook(force=True)
        return conv
    return geo


def polygons_of(host, geo) -> list:
    sop = _as_sop(host, geo)
    prims = sop.prims() if callable(sop.prims) else sop.prims
    polys = []
    for prim in prims:
        pts = []
        for vert in prim:
            pt = vert.point
            pts.append((float(pt.x), float(pt.y), float(pt.z)))
        if len(pts) >= 3:
            polys.append(pts)
    return polys


class Demo3D:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp

    def _mesh_op(self):
        path = self.ownerComp.par.Meshop.eval()
        geo = op(path)  # noqa: F821
        if geo is None:
            raise RuntimeError(f"mesh operator not found: {path}")
        return geo

    def Triangles(self, geo=None) -> list:
        import sdf3d  # noqa: PLC0415

        polys = polygons_of(self.ownerComp, geo if geo is not None else self._mesh_op())
        tris = []
        for poly in polys:
            tris.extend(sdf3d.triangulate(poly))
        return sdf3d.orient_outward(tris)

    def _set_uniform(self, top, name, values) -> None:
        count = int(top.seq.vec.numBlocks)
        for i in range(count):
            if getattr(top.par, f"vec{i}name").eval() != name:
                continue
            for axis, value in zip("xyzw", list(values) + [0, 0, 0, 0]):
                par = getattr(top.par, f"vec{i}value{axis}")
                par.expr = ""
                par.val = float(value)
            return
        raise KeyError(name)

    def RebuildMesh(self) -> dict:
        import numpy as np  # noqa: PLC0415

        import sdf3d  # noqa: PLC0415

        tris = self.Triangles()
        n = len(tris)
        arr = np.zeros((max(n, 1), 4, 4), np.float32)
        if n:
            for i, (a, b, c) in enumerate(tris):
                arr[i, 0, :3] = a
                arr[i, 1, :3] = b
                arr[i, 2, :3] = c
                arr[i, 3, :3] = sdf3d.triangle_normal(a, b, c)
            stacked = np.array(tris, np.float32).reshape(-1, 3)
            pad = 0.25
            mn = stacked.min(axis=0) - pad
            mx = stacked.max(axis=0) + pad
        else:
            mn = np.array([-0.5, -0.5, -0.5], np.float32)
            mx = np.array([0.5, 0.5, 0.5], np.float32)
        tex = self.ownerComp.op("triangles")
        tex.par.outputresolution = "custom"
        tex.par.resolutionw = 4
        tex.par.resolutionh = max(n, 1)
        tex.par.format = "rgba32float"
        tex.lock = True
        tex.copyNumpyArray(arr)
        p = self.ownerComp.par
        p.Sdfminx, p.Sdfminy, p.Sdfminz = (float(v) for v in mn)
        p.Sdfmaxx, p.Sdfmaxy, p.Sdfmaxz = (float(v) for v in mx)
        p.Tricount = n
        p.Sdfflip = 1
        p.Meshhash = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
        depth = max(int(op.simbus3d.par.Resw.eval()), int(op.simbus3d.par.Resh.eval()), int(op.simbus3d.par.Resd.eval()))  # noqa: F821
        sdf = self.ownerComp.op("mesh_sdf")
        final = self.ownerComp.op("mesh_final")
        for top in (sdf, final):
            self._set_uniform(top, "uBMin", (float(mn[0]), float(mn[1]), float(mn[2]), 0))
            self._set_uniform(top, "uBMax", (float(mx[0]), float(mx[1]), float(mx[2]), 0))
        import math  # noqa: PLC0415

        sdf.par.npasses = 2 + int(math.ceil(math.log2(max(depth, 2))))
        sdf.cook(force=True)
        final.cook(force=True)
        volume = final.numpyArray(delayed=False)
        if volume is not None and float(volume[..., 0].min()) > 1.0 and n:
            # Script TOP rows and texelFetch disagreed; flip and rebuild once.
            tex.copyNumpyArray(arr[::-1].copy())
            sdf.cook(force=True)
            final.cook(force=True)
            volume = final.numpyArray(delayed=False)
        cz, cy, cx = (s // 2 for s in volume.shape[:3])
        if float(volume[cz, cy, cx, 0]) > 0.0:
            p.Sdfflip = -1
            final.cook(force=True)
        hold = self.ownerComp.op("sdf_hold")
        hold.par.active = True
        hold.cook(force=True)
        hold.par.active = False
        return {"triangles": n, "flip": int(p.Sdfflip.eval()), "min": float(final.numpyArray(delayed=False)[..., 0].min())}
