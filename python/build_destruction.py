"""Destruction network: fracture pieces, a rigid step, bus obstacles, dust, and a lit mesh.

Built from python/build_network.py. Nothing here is published or stepped until Destruction.Arm().
"""

from __future__ import annotations

import td

import build_fluid3d as bf


P = "parent().par."
CLOCK = ("op.simclock.par.Dt", "op.simclock.par.Time", "op.simclock.par.Step", "op.simclock.par.Frame")


def build_destruction(sims, bn):
    comp = bn.place(sims.create(td.baseCOMP, "destruction"), 10, 0)
    bn.par(comp, "Sim", "int", "Pieces", 120, minmax=(1, 4096))
    bn.par(comp, "Sim", "int", "Seed", 1)
    bn.par(comp, "Sim", "menu", "Pattern", "clustered", menu=["uniform", "clustered", "radial"])
    bn.par(comp, "Sim", "menu", "Shape", "box", menu=["box", "sphere", "mesh"])
    for name, default in (("Centerx", 0.15), ("Centery", 0.55), ("Centerz", 0.0),
                          ("Halfx", 0.1), ("Halfy", 0.5), ("Halfz", 0.4)):
        bn.par(comp, "Sim", "float", name, default)
    bn.par(comp, "Sim", "str", "Meshpath", "/sims/demo3d/mesh_geo/sphere")
    bn.par(comp, "Sim", "float", "Sigma", 0.18)
    bn.par(comp, "Sim", "float", "Glue", 0.35, label="Glue Threshold")
    bn.par(comp, "Sim", "float", "Gravity", 9.81)
    bn.par(comp, "Sim", "int", "Iterations", 12, minmax=(1, 64))
    bn.par(comp, "Sim", "int", "Positers", 6, minmax=(1, 16), label="Position Iters")
    bn.par(comp, "Sim", "toggle", "Meshcontacts", False, label="Mesh Contacts")
    bn.par(comp, "Sim", "toggle", "Projectile", True)
    bn.par(comp, "Sim", "menu", "Breakmode", "impact", menu=["impact", "frame", "explosion", "pressure"])
    bn.par(comp, "Sim", "int", "Breakframe", 8)
    for name, default in (("Explodex", 0.15), ("Explodey", 0.7), ("Explodez", 0.0),
                          ("Exploderadius", 0.55), ("Explodestrength", 2.5)):
        bn.par(comp, "Sim", "float", name, default)
    bn.par(comp, "Sim", "float", "Pressurethresh", 1.5, label="Pressure Threshold")
    bn.par(comp, "Sim", "float", "Fluiddrag", 0.0)
    bn.par(comp, "Sim", "int", "Debrisn", 12, minmax=(1, 256), label="Debris Per Break")
    bn.par(comp, "Sim", "float", "Dustsigma", 0.08)
    bn.par(comp, "Sim", "float", "Dustheat", 2.0)
    bn.par(comp, "Sim", "int", "Dustcount", 48, minmax=(1, 64))
    bn.par(comp, "Sim", "toggle", "Cookrender", False, label="Cook Render")
    bn.par(comp, "Sim", "int", "Renderw", 960)
    bn.par(comp, "Sim", "int", "Renderh", 540)
    for name, default in (("Camaz", 0.6), ("Camel", 0.35), ("Camdist", 3.4), ("Camfov", 32.0),
                          ("Targetx", 0.1), ("Targety", 0.7), ("Targetz", 0.0)):
        bn.par(comp, "Sim", "float", name, default)
    bn.par(comp, "Sim", "int", "Bodycount", 0, readonly=True)
    bn.par(comp, "Sim", "int", "Grid", 16, minmax=(4, 16))
    bn.par(comp, "Sim", "toggle", "Rowflip", False, label="Row Flip")
    bn.par(comp, "Sim", "pulse", "Arm")
    bn.par(comp, "Sim", "pulse", "Release")
    bn.par(comp, "Sim", "str", "Status", "idle", readonly=True)

    bn.extension(comp, "destruction_ext", "python/destruction.py", "Destruction")
    bn.pulse_callbacks(comp, {"Arm": "o.Arm()", "Release": "o.Release()"})
    src = {
        "obstacles": bn.shader(comp, "obstacles_glsl", "shaders/destruction/obstacles.glsl", -1, 0),
        "obstacles2d": bn.shader(comp, "obstacles2d_glsl", "shaders/destruction/obstacles2d.glsl", -1, 1),
        "dust": bn.shader(comp, "dust_glsl", "shaders/destruction/dust.glsl", -1, 2),
        "piece_vert": bn.shader(comp, "piece_vert_glsl", "shaders/destruction/piece_vert.glsl", -1, 3),
        "piece_frag": bn.shader(comp, "piece_frag_glsl", "shaders/destruction/piece_frag.glsl", -1, 4),
        "composite": bn.shader(comp, "composite_glsl", "shaders/destruction/composite.glsl", -1, 5),
    }
    bn.shader(comp, "bus3d_glsl", "shaders/common/bus3d.glsl", -1, 6)
    bn.shader(comp, "bus_glsl", "shaders/common/bus.glsl", -1, 7)

    def table(name, w, h):
        top = bn.place(comp.create(td.scriptTOP, name), 0, 0 if name == "bodies" else 1)
        top.par.outputresolution = "custom"
        top.par.resolutionw = w
        top.par.resolutionh = h
        top.par.format = "rgba32float"
        top.lock = True
        return top

    bodies = table("bodies", 256, 256)
    extents = table("extents", 256, 256)
    bins = table("bins", 256, 64)
    bursts = table("bursts", 64, 64)
    state_uniforms = [("uCount", (P + "Bodycount", P + "Grid", 3, P + "Rowflip"))]
    obs = bf.glsl3d(bn, comp, "obstacles3d", src["obstacles"], [bodies, extents, bins], "rgba32float",
                    state_uniforms, col=2, row=0)
    obs.bypass = True
    dust = bf.glsl3d(bn, comp, "dust3d", src["dust"], [bursts], "rgba32float",
                     [("uDust", (P + "Dustsigma", P + "Dustcount", P + "Dustheat", 0))], col=2, row=1)
    dust.bypass = True
    obs2 = bn.glsl(comp, "obstacles2d", src["obstacles2d"], [bodies, extents, bins], "rgba32float",
                   state_uniforms + [("uSlice", (0.0, "op.simbus3d.par.Boundsz0", "op.simbus3d.par.Boundsz1", 0))],
                   col=2, row=2)
    obs2.bypass = True

    cam_pars = (P + "Camaz", P + "Camel", P + "Camdist", P + "Camfov")
    target = (P + "Targetx", P + "Targety", P + "Targetz", 0)
    mat = bn.place(comp.create(td.glslMAT, "piece_mat"), 4, 2)
    mat.par.glslversion = "glsl460"
    mat.par.vdat = src["piece_vert"]
    mat.par.pdat = src["piece_frag"]
    mat.par.depthtest = True
    mat.par.depthwriting = True
    mat.seq.sampler.numBlocks = 1
    mat.par.sampler0name = "sState"
    mat.par.sampler0top = bodies
    mat.par.sampler0filter = "nearest"
    bf.set_uniforms(mat, [
        ("uCount", (P + "Bodycount", P + "Grid", 3, P + "Rowflip")),
        ("uCam", cam_pars),
        ("uTarget", target),
        ("uRes", (P + "Renderw", P + "Renderh", 0, 0)),
        ("uLight", (0.35, 0.85, 0.28, 0)),
    ])
    geo = bn.place(comp.create(td.geometryCOMP, "pieces_geo"), 4, 0)
    torus = geo.op("torus1")
    if torus is not None:
        torus.destroy()
    geo.par.material = mat
    mesh = geo.create(td.scriptSOP, "mesh")
    mesh.render = True
    mesh.display = True
    dat = bn.place(comp.create(td.textDAT, "mesh_callbacks"), 4, 3)
    dat.text = (
        "def onCook(scriptOp):\n"
        "    scriptOp.parent(2).ext.Destruction.FillMesh(scriptOp)\n"
        "    return\n"
    )
    mesh.par.callbacks = dat
    cam = bn.place(comp.create(td.cameraCOMP, "cam"), 5, 2)
    cam.par.tz = 5
    render = bn.place(comp.create(td.renderTOP, "pieces_render"), 5, 0)
    render.par.camera = cam
    render.par.geometry = geo
    render.par.outputresolution = "custom"
    render.par.resolutionw.expr = P + "Renderw"
    render.par.resolutionh.expr = P + "Renderh"
    render.par.format = "rgba16float"
    render.par.antialias = "aa1"
    render.par.dither = False
    # Order-independent transparency does not clear uncovered pixels, so the background alpha
    # stayed 0. Ordinary blending with an opaque black clear fills the frame.
    render.par.transparency = "blend"
    render.par.bgcolora = 1
    render.par.bgcolorr = 0
    render.par.bgcolorg = 0
    render.par.bgcolorb = 0
    render.bypass = True
    vol = bn.select(comp, "vol_in", "op('/sims/fluid3d').path + '/render_out'", 6, 0)
    comp_glsl = bn.glsl(comp, "composite", src["composite"], [render, vol], "rgba16float",
                        [("uComp", (0.35, 0.0, 0.02, 0))], col=6, row=1)
    # bn.glsl forces the bus resolution. The picture should match the piece render.
    comp_glsl.par.outputresolution = "custom"
    comp_glsl.par.resolutionw.expr = P + "Renderw"
    comp_glsl.par.resolutionh.expr = P + "Renderh"
    comp_glsl.bypass = True
    bn.null(comp, "picture_out", comp_glsl, 7, 1)
    load = bn.place(comp.create(td.moviefileinTOP, "load_bodies"), 0, 3)
    load.par.playmode = "locked"
    load.par.index = 0
    return comp
