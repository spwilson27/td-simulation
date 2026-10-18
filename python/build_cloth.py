"""GPU XPBD cloth under /sims/cloth. Substeps are a fixed TOP chain; colours are passes inside each one."""

from __future__ import annotations

import td

import build_fluid3d as b3

MAX_SUBSTEPS = 16
SELF_ITERS = 4
SOLVE_PASSES = 21
P = "parent().par."
BMIN = ("op.simbus3d.par.Boundsx0", "op.simbus3d.par.Boundsy0", "op.simbus3d.par.Boundsz0", 0)
BMAX = ("op.simbus3d.par.Boundsx1", "op.simbus3d.par.Boundsy1", "op.simbus3d.par.Boundsz1", 0)
DT = "op.simclock.par.Dt / max(parent().par.Substeps, 1)"
DX = "parent().par.Sizex / max(parent().par.Resx - 1, 1)"
DY = "parent().par.Sizey / max(parent().par.Resy - 1, 1)"
SORT = ("max(1, (int(parent().par.Resx * parent().par.Resy).bit_length() - 1)"
        " * int(parent().par.Resx * parent().par.Resy).bit_length() // 2)")
CELL = ("(op.simbus3d.par.Boundsx1 - op.simbus3d.par.Boundsx0) / max(op.simbus3d.par.Resw, 1)")
SCALE = f"parent().par.Couplegain / (1.2 * ({CELL}) * ({CELL}) * ({CELL}))"


def glsl2d(comp, name, shader, inputs, fmt, uniforms, col, row, npasses=None, res=None):
    top = comp.create(td.glslTOP, name)
    top.nodeX, top.nodeY = col * 200, -row * 150
    top.par.glslversion = "glsl460"
    top.par.mode = "compute"
    top.par.computedat = shader
    top.par.format = fmt
    top.par.outputresolution = "custom"
    top.par.resolutionw.expr = P + "Resx" if res is None else res[0]
    top.par.resolutionh.expr = P + "Resy" if res is None else res[1]
    top.par.filtertype = "nearest"
    top.par.inputfiltertype = "nearest"
    top.par.outputaccess = "readwrite" if npasses is not None else "writeonly"
    b3.wire(top, inputs)
    b3.set_uniforms(top, uniforms)
    b3._undock(top)
    if npasses is not None:
        if isinstance(npasses, str):
            top.par.npasses.expr = npasses
        else:
            top.par.npasses = int(npasses)
    return top


def build_cloth(sims, bn):
    f = bn.place(sims.create(td.baseCOMP, "cloth"), 14, 0)
    bn.par(f, "Sim", "menu", "Preset", "realtime", menu=["realtime", "hang", "drape", "showcase"])
    bn.par(f, "Sim", "pulse", "Applypreset", label="Apply Preset")
    bn.par(f, "Sim", "int", "Resx", 64, minmax=(8, 256))
    bn.par(f, "Sim", "int", "Resy", 64, minmax=(8, 256))
    bn.par(f, "Sim", "int", "Substeps", 4, minmax=(1, MAX_SUBSTEPS))
    bn.par(f, "Sim", "int", "Selfiters", 2, minmax=(0, SELF_ITERS), label="Self Iterations")
    bn.par(f, "Sim", "menu", "Plane", "xy", menu=["xy", "xz"])
    bn.par(f, "Sim", "menu", "Pin", "left", menu=["left", "corners", "none", "top"])
    bn.par(f, "Sim", "toggle", "Collide", True)
    bn.par(f, "Sim", "toggle", "Couple", False, label="Two-Way Force")
    bn.par(f, "Sim", "toggle", "Cookrender", False, label="Cook Render")
    bn.par(f, "Sim", "int", "Hashres", 32, minmax=(8, 96), label="Hash Resolution")
    for name, default in (("Originx", -0.85), ("Originy", 1.55), ("Originz", 0.35),
                          ("Sizex", 0.95), ("Sizey", 0.75)):
        bn.par(f, "Layout", "float", name, default)
    for name, default in (("Attachx", 0.0), ("Attachy", 0.0), ("Attachz", 0.0), ("Attachyaw", 0.0)):
        bn.par(f, "Layout", "float", name, default)
    bn.par(f, "Physics", "float", "Density", 0.25)
    bn.par(f, "Physics", "float", "Stretch", 2.0e-5, label="Stretch Compliance")
    bn.par(f, "Physics", "float", "Shear", 2.0e-5, label="Shear Compliance")
    bn.par(f, "Physics", "float", "Bend", 5.0e-4, label="Bend Compliance")
    bn.par(f, "Physics", "float", "Damping", 0.6)
    bn.par(f, "Physics", "float", "Gravityx", 0.0)
    bn.par(f, "Physics", "float", "Gravityy", -9.81)
    bn.par(f, "Physics", "float", "Gravityz", 0.0)
    bn.par(f, "Physics", "float", "Thickness", 0.008)
    bn.par(f, "Physics", "float", "Friction", 0.4)
    bn.par(f, "Physics", "float", "Ground", 0.0)
    bn.par(f, "Physics", "float", "Maxspeed", 8.0)
    bn.par(f, "Physics", "float", "Tear", 0.35, label="Tear Strain")
    bn.par(f, "Physics", "float", "Air", 1.2, label="Air Density")
    bn.par(f, "Physics", "float", "Drag", 0.8)
    bn.par(f, "Physics", "float", "Couplegain", 0.02, label="Force Gain")
    for name, default in (("Spherx", 0.0), ("Sphery", 0.36), ("Spherez", 0.7), ("Spherer", 0.22),
                          ("Sphervx", 0.0), ("Sphervy", 0.0), ("Sphervz", 0.0)):
        bn.par(f, "Obstacle", "float", name, default)
    bn.par(f, "Look", "int", "Renderw", 480)
    bn.par(f, "Look", "int", "Renderh", 270)
    for name, default in (("Camaz", 0.9), ("Camel", 0.28), ("Camdist", 2.8), ("Camfov", 38.0),
                          ("Targetx", -0.2), ("Targety", 1.1), ("Targetz", 0.2)):
        bn.par(f, "Look", "float", name, default)
    bn.par(f, "Look", "rgb", "Front", [0.62, 0.22, 0.16])
    bn.par(f, "Look", "rgb", "Back", [0.18, 0.28, 0.55])
    bn.extension(f, "cloth_ext", "python/cloth.py", "Cloth")
    bn.pulse_callbacks(f, {"Applypreset": "o.ApplyPreset(o.par.Preset.eval())"})

    names = (
        "cloth_common", "init", "solve", "velocity", "keys", "sort", "bins", "self", "pack_bins",
        "aero", "force", "jet", "wind", "sphere", "cloth_vert", "cloth_frag", "composite",
    )
    src = {}
    for i, name in enumerate(names):
        src[name] = bn.shader(f, name + "_glsl", f"shaders/cloth/{name}.glsl", -1, i)

    init_u = [
        ("uOrigin", (P + "Originx", P + "Originy", P + "Originz", 0)),
        ("uSize", (P + "Sizex", P + "Sizey", 0, 0)),
        ("uPin", (P + "Pin.menuIndex", 0, 0, 0)),
        ("uMass", (P + "Density", 0, 0, 0)),
        ("uPlane", (P + "Plane.menuIndex", 0, 0, 0)),
    ]
    solve_u = [
        ("uDyn", (DT, P + "Damping", P + "Maxspeed", P + "Collide")),
        ("uGrav", (P + "Gravityx", P + "Gravityy", P + "Gravityz", P + "Air")),
        ("uAero", (P + "Drag", 12.0, P + "Friction", P + "Thickness")),
        ("uComp", (P + "Stretch", P + "Shear", P + "Bend", P + "Tear")),
        ("uLen", (DX, DY, 0, 0)),
        ("uGround", (P + "Ground", 0, 0, 0)),
        ("uHash", (P + "Hashres", 0, 0, 0)),
        ("uAttach", (P + "Attachx", P + "Attachy", P + "Attachz", P + "Attachyaw")),
        ("uOrigin", (P + "Originx", P + "Originy", P + "Originz", 0)),
        ("uPin", (P + "Pin.menuIndex", 0, 0, 0)),
        ("uBMin", BMIN),
        ("uBMax", BMAX),
    ]
    vel_u = [
        ("uDyn", (DT, P + "Damping", P + "Maxspeed", 0)),
        ("uLen", (DX, DY, 0, 0)),
        ("uTear", (P + "Tear", 0, 0, 0)),
        ("uMode", (0, 0, 0, 0)),
    ]
    hash_u = [("uBMin", BMIN), ("uBMax", BMAX), ("uHash", (P + "Hashres", 0, 0, 0))]
    part = (P + "Resx", P + "Resy", P + "Resx * " + P + "Resy", 0)

    init = glsl2d(f, "init_pos", src["init"], [], "rgba32float", init_u, 0, 0)
    glsl2d(f, "rest", src["init"], [], "rgba32float", init_u, 0, 1)
    vel0 = bn.place(f.create(td.constantTOP, "init_vel"), 0, 2)
    vel0.par.outputresolution = "custom"
    vel0.par.resolutionw.expr = P + "Resx"
    vel0.par.resolutionh.expr = P + "Resy"
    vel0.par.format = "rgba32float"
    vel0.par.colorr = vel0.par.colorg = vel0.par.colorb = vel0.par.alpha = 0
    for i, field in enumerate(("pos", "vel")):
        src_top = bn.select(f, "src_" + field, f"op('init_{field}')", 1, i)
        src_top.par.top.expr = ""
        src_top.par.top = f.op("init_" + field).path

    wind = b3.glsl3d(bn, f, "uniform_vel", src["wind"], [], "rgba32float",
                     [("uWind", (0, 0, 0, 0))], col=0, row=4, include_bus=False)
    for axis in ("resolutionw", "resolutionh", "customdepth"):
        getattr(wind.par, axis).expr = "8"
    wind_src = bn.select(f, "wind_src", "op('uniform_vel')", 1, 4)
    wind_src.par.top.expr = ""
    wind_src.par.top = wind.path
    solid = bn.select(f, "solid_src", "op.simbus3d.path + '/obstacles'", 1, 5)
    sphere = b3.glsl3d(bn, f, "sphere_obs", src["sphere"], [], "rgba32float",
                       [("uBMin", BMIN), ("uBMax", BMAX),
                        ("uCenter", (P + "Spherx", P + "Sphery", P + "Spherez", P + "Spherer")),
                        ("uMotion", (P + "Sphervx", P + "Sphervy", P + "Sphervz", 0)),
                        ("uTime", ("op.simclock.par.Time", 0, 0, 0))],
                       col=0, row=6, include_bus=False)

    keys0 = glsl2d(f, "keys0", src["keys"], [f.op("src_pos")], "rgba32float", hash_u, 2, 0)
    sort0 = glsl2d(f, "sort0", src["sort"], [keys0], "rgba32float", [("uUnused", (0, 0, 0, 0))], 3, 0, npasses=SORT)
    bins0 = b3.glsl3d(bn, f, "bins0", src["bins"], [sort0], "rgba32float", [("uPart", part)],
                      col=4, row=0, include_bus=False)
    for axis in ("resolutionw", "resolutionh", "customdepth"):
        getattr(bins0.par, axis).expr = P + "Hashres"

    prev_pos = f.op("src_pos")
    prev_vel = f.op("src_vel")
    for k in range(MAX_SUBSTEPS):
        start = bn.null(f, f"start_{k}", prev_pos, 5, k)
        solve = glsl2d(
            f, f"solve_{k}", src["solve"],
            [prev_pos, prev_vel, f.op("rest"), start, keys0, wind_src, solid, bins0],
            "rgba32float", solve_u, 6, k, npasses=SOLVE_PASSES)
        vel = glsl2d(f, f"vel_{k}", src["velocity"], [solve, start, prev_vel], "rgba32float", vel_u, 7, k)
        prev_pos, prev_vel = solve, vel

    pos_solved = bn.select(f, "pos_solved", f.op("solve_0").path, 8, 0)
    vel_solved = bn.select(f, "vel_solved", f.op("vel_0").path, 8, 1)
    pos_solved.par.top.expr = ""
    vel_solved.par.top.expr = ""
    keys1 = glsl2d(f, "keys1", src["keys"], [pos_solved], "rgba32float", hash_u, 9, 0)
    sort1 = glsl2d(f, "sort1", src["sort"], [keys1], "rgba32float", [("uUnused", (0, 0, 0, 0))], 10, 0, npasses=SORT)
    bins1 = b3.glsl3d(bn, f, "bins1", src["bins"], [sort1], "rgba32float", [("uPart", part)],
                      col=11, row=0, include_bus=False)
    for axis in ("resolutionw", "resolutionh", "customdepth"):
        getattr(bins1.par, axis).expr = P + "Hashres"
    self_u = hash_u + [("uAero", (P + "Drag", 12.0, P + "Friction", P + "Thickness"))]
    prev = pos_solved
    for k in range(SELF_ITERS):
        prev = glsl2d(f, f"self_{k}", src["self"], [prev, sort1, bins1], "rgba32float", self_u, 12, k)
    pos_final = bn.select(f, "pos_final", prev.path, 13, 0)
    pos_final.par.top.expr = ""
    # Frame-average velocity from the positions at the start and end of the step. The last
    # substep's velocity depends on that substep's internal path and a reloaded state can
    # take a different path to the same positions; this does not.
    final_u = [
        ("uDyn", ("op.simclock.par.Dt", P + "Damping", P + "Maxspeed", 0)),
        ("uLen", (DX, DY, 0, 0)),
        ("uTear", (P + "Tear", 0, 0, 0)),
        ("uMode", (0, 0, 0, 0)),
    ]
    vel_final = glsl2d(f, "vel_final", src["velocity"], [pos_final, f.op("src_pos"), f.op("src_vel")],
                       "rgba32float", final_u, 13, 1)
    bn.null(f, "pos_state", pos_final, 14, 0)
    bn.null(f, "vel_state", vel_final, 14, 1)
    for i, field in enumerate(("pos", "vel")):
        out = bn.null(f, field + "_out", f.op(field + "_state"), 15, i)
        out.lock = True

    aero = glsl2d(f, "aero", src["aero"], [f.op("pos_out"), f.op("vel_out"), wind_src], "rgba32float",
                  [("uBMin", BMIN), ("uBMax", BMAX),
                   ("uGrav", (P + "Gravityx", P + "Gravityy", P + "Gravityz", P + "Air")),
                   ("uAero", (P + "Drag", 12.0, P + "Friction", P + "Thickness")),
                   ("uLen", (DX, DY, 0, 0))], 16, 0)
    pack = glsl2d(f, "bins_pack", src["pack_bins"], [bins1], "rgba32float",
                  [("uHash", (P + "Hashres", 0, 0, 0))], 16, 1,
                  res=(P + "Hashres", P + "Hashres * " + P + "Hashres"))
    for name, node in (("aero_out", aero), ("bins_out", pack), ("keys_out", sort1)):
        locked = bn.null(f, name, node, 17, 0)
        locked.lock = True
    jet_in = bn.constant(f, "jet_in", "rgba32float", (0, 0, 0, 1), 17, 1)
    b3.glsl3d(bn, f, "jet", src["jet"], [jet_in], "rgba32float",
              [("uBMin", BMIN), ("uBMax", BMAX),
               ("uBox0", (0.15, 1.2, -0.55, 8.0)),
               ("uBox1", (0.85, 1.75, -0.2, 0.0))],
              col=18, row=1, include_bus=False)
    force = b3.glsl3d(bn, f, "cloth_force", src["force"],
                      [f.op("bins_out"), f.op("keys_out"), f.op("aero_out"), f.op("pos_out")],
                      "rgba32float",
                      [("uBMin", BMIN), ("uBMax", BMAX), ("uHash", (P + "Hashres", 0, 0, 0)),
                       ("uPart", part), ("uScale", (SCALE, 0, 0, 0))],
                      col=18, row=0, include_bus=False)

    cam_pars = (P + "Camaz", P + "Camel", P + "Camdist", P + "Camfov")
    target = (P + "Targetx", P + "Targety", P + "Targetz", 0)
    mat = bn.place(f.create(td.glslMAT, "cloth_mat"), 19, 2)
    mat.par.glslversion = "glsl460"
    mat.par.vdat = src["cloth_vert"]
    mat.par.pdat = src["cloth_frag"]
    mat.par.depthtest = True
    mat.par.depthwriting = True
    try:
        mat.par.cullface = "none"
    except Exception:
        pass
    mat.seq.sampler.numBlocks = 1
    mat.par.sampler0name = "sPos"
    mat.par.sampler0top = f.op("pos_out")
    mat.par.sampler0filter = "nearest"
    b3.set_uniforms(mat, [
        ("uCam", cam_pars),
        ("uTarget", target),
        ("uRes", (P + "Renderw", P + "Renderh", 0, 0)),
        ("uFront", (P + "Frontr", P + "Frontg", P + "Frontb", 1)),
        ("uBack", (P + "Backr", P + "Backg", P + "Backb", 1)),
        ("uLight", (0.35, 0.9, 0.28, 0)),
    ])
    geo = bn.place(f.create(td.geometryCOMP, "cloth_geo"), 19, 0)
    torus = geo.op("torus1")
    if torus is not None:
        torus.destroy()
    geo.par.material = mat
    mesh = geo.create(td.scriptSOP, "mesh")
    mesh.render = True
    mesh.display = True
    dat = bn.place(f.create(td.textDAT, "mesh_callbacks"), 19, 3)
    dat.text = (
        "def onCook(scriptOp):\n"
        "    scriptOp.parent(2).ext.Cloth.FillMesh(scriptOp)\n"
        "    return\n"
    )
    mesh.par.callbacks = dat
    cam = bn.place(f.create(td.cameraCOMP, "cam"), 20, 2)
    cam.par.tz = 4
    render = bn.place(f.create(td.renderTOP, "cloth_render"), 20, 0)
    render.par.camera = cam
    render.par.geometry = geo
    render.par.outputresolution = "custom"
    render.par.resolutionw.expr = P + "Renderw"
    render.par.resolutionh.expr = P + "Renderh"
    render.par.format = "rgba16float"
    render.par.antialias = "aa1"
    render.par.dither = False
    render.par.transparency = "blend"
    render.par.bgcolorr = 0
    render.par.bgcolorg = 0
    render.par.bgcolorb = 0
    render.par.bgcolora = 0
    plate = bn.place(f.create(td.constantTOP, "plate"), 20, 1)
    plate.par.outputresolution = "custom"
    plate.par.resolutionw.expr = P + "Renderw"
    plate.par.resolutionh.expr = P + "Renderh"
    plate.par.format = "rgba16float"
    plate.par.colorr, plate.par.colorg, plate.par.colorb, plate.par.alpha = 0.045, 0.05, 0.06, 1
    bg = bn.select(f, "bg_src", plate.path, 21, 1)
    bg.par.top.expr = ""
    bg.par.top = plate.path
    comp = bn.glsl(f, "composite", src["composite"], [render, bg], "rgba16float", col=21, row=0)
    comp.par.outputresolution = "custom"
    comp.par.resolutionw.expr = P + "Renderw"
    comp.par.resolutionh.expr = P + "Renderh"
    bn.null(f, "picture_out", comp, 22, 0)
    return f
