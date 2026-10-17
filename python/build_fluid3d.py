"""3D bus, fluid solver, and demo. Called from build_network.build with that module as `bn`."""

from __future__ import annotations

import td

LEVELS = 4


def bus_uniforms():
    return [
        ("uBMin", ("op.simbus3d.par.Boundsx0", "op.simbus3d.par.Boundsy0", "op.simbus3d.par.Boundsz0", 0)),
        ("uBMax", ("op.simbus3d.par.Boundsx1", "op.simbus3d.par.Boundsy1", "op.simbus3d.par.Boundsz1", 0)),
        ("uWalls", ("op.simbus3d.par.Wallleft", "op.simbus3d.par.Wallright", "op.simbus3d.par.Wallbottom", "op.simbus3d.par.Walltop")),
        ("uWallsZ", ("op.simbus3d.par.Wallfront", "op.simbus3d.par.Wallback", 0, 0)),
        ("uSponge", ("op.simbus3d.par.Spongewidth", "op.simbus3d.par.Spongerate", 0, 0)),
        ("uClock", ("op.simclock.par.Dt", "op.simclock.par.Time", "op.simclock.par.Step", "op.simclock.par.Frame")),
        ("uClock2", ("op.simclock.par.Seed", 0, 0, 0)),
    ]


def set_uniforms(top, uniforms):
    # GLSL TOP vector sequences cannot be empty. A static volume passes no uniforms.
    if not uniforms:
        uniforms = (("uUnused", (0, 0, 0, 0)),)
    top.seq.vec.numBlocks = len(uniforms)
    for i, (uname, values) in enumerate(uniforms):
        getattr(top.par, f"vec{i}name").val = uname
        vals = list(values) + [0] * (4 - len(list(values)))
        for axis, value in zip("xyzw", vals):
            par = getattr(top.par, f"vec{i}value{axis}")
            if isinstance(value, str):
                par.expr = value
            else:
                par.expr = ""
                par.val = value


def wire(top, inputs):
    top.setInputs(list(inputs[:3]))
    top.par.tops = " ".join(src.path for src in inputs[3:])


def _undock(top):
    for dock in list(top.docked):
        dock.destroy()


def glsl3d(bn, comp, name, shader, inputs, fmt, uniforms=(), col=0, row=0, npasses=None, filtering="nearest", level=0, include_bus=True):
    top = bn.place(comp.create(td.glslTOP, name), col, row)
    top.par.glslversion = "glsl460"
    top.par.mode = "compute"
    top.par.computedat = shader
    top.par.type = "texture3d"
    top.par.depth = "custom"
    top.par.format = fmt
    top.par.outputresolution = "custom"
    div = 1 << level
    top.par.resolutionw.expr = f"max(4, int(op.simbus3d.par.Resw / {div}))"
    top.par.resolutionh.expr = f"max(4, int(op.simbus3d.par.Resh / {div}))"
    top.par.customdepth.expr = f"max(4, int(op.simbus3d.par.Resd / {div}))"
    top.par.inputfiltertype = filtering
    top.par.filtertype = filtering
    top.par.outputaccess = "readwrite" if npasses is not None else "writeonly"
    wire(top, inputs)
    set_uniforms(top, (bus_uniforms() if include_bus else []) + list(uniforms))
    if npasses is not None:
        if isinstance(npasses, str):
            top.par.npasses.expr = npasses
        else:
            top.par.npasses = int(npasses)
    _undock(top)
    return top


def glsl_pixel(bn, comp, name, shader, inputs, fmt, uniforms, res_w, res_h, col, row, filtering="linear"):
    top = bn.place(comp.create(td.glslTOP, name), col, row)
    top.par.glslversion = "glsl460"
    top.par.pixeldat = shader
    top.par.format = fmt
    top.par.outputresolution = "custom"
    top.par.resolutionw.expr = res_w
    top.par.resolutionh.expr = res_h
    top.par.inputfiltertype = filtering
    top.par.filtertype = filtering
    wire(top, inputs)
    set_uniforms(top, uniforms)
    _undock(top)
    return top


def hold(bn, comp, name, src, fmt, col, row):
    cache = bn.place(comp.create(td.cacheTOP, name), col, row)
    src.outputConnectors[0].connect(cache.inputConnectors[0])
    cache.par.active = False
    cache.par.alwayscook = False
    cache.par.cachesize = 1
    cache.par.outputindex = 0
    cache.par.replace = True
    cache.par.replaceindex = 0
    cache.par.format = fmt
    cache.par.outputresolution = "useinput"
    cache.par.inputfiltertype = "nearest"
    cache.par.filtertype = "nearest"
    return cache


def select(bn, comp, name, path, col, row):
    sel = bn.place(comp.create(td.selectTOP, name), col, row)
    if isinstance(path, str) and path.startswith("op."):
        sel.par.top.expr = path
    else:
        sel.par.top = path
    return sel


def build_bus3d(sims, bn):
    b = bn.place(sims.create(td.baseCOMP, "bus3d"), 6, 0)
    b.par.opshortcut = "simbus3d"
    for name, default in (("Resw", 128), ("Resh", 128), ("Resd", 128)):
        bn.par(b, "Grid", "int", name, default, minmax=(32, 512))
    for name, default in (("Boundsx0", -1.0), ("Boundsy0", 0.0), ("Boundsz0", -1.0),
                          ("Boundsx1", 1.0), ("Boundsy1", 2.0), ("Boundsz1", 1.0)):
        bn.par(b, "Grid", "float", name, default)
    for name, default in (("Wallleft", True), ("Wallright", True), ("Wallbottom", True), ("Walltop", False),
                          ("Wallfront", True), ("Wallback", True)):
        bn.par(b, "Grid", "toggle", name, default)
    bn.par(b, "Grid", "float", "Spongewidth", 0.35, label="Open Side Band Width")
    bn.par(b, "Grid", "float", "Spongerate", 4.0, label="Open Side Fade Rate")
    bn.par(b, "Fields", "str", "Fieldsource", "/sims/fluid3d")
    bn.extension(b, "bus3d_ext", "python/simbus.py", "FieldBus3D")
    bn.shader(b, "bus3d_glsl", "shaders/common/bus3d.glsl", -1, 2)
    empty_src = bn.shader(b, "empty_src", "shaders/common/empty3d.glsl", -1, 3)
    zero_src = bn.shader(b, "zero_src", "shaders/common/zero3d.glsl", -1, 4)
    sum_src = bn.shader(b, "sum_src", "shaders/common/sum3d.glsl", -1, 5)
    union_src = bn.shader(b, "union_src", "shaders/common/sdf_union3d.glsl", -1, 6)
    table = bn.place(b.create(td.tableDAT, "publishers"), -2, 2)
    table.clear()
    table.appendRow(["kind", "path"])
    zero = glsl3d(bn, b, "zero", zero_src, [], "rgba32float", col=0, row=0)
    empty = glsl3d(bn, b, "empty_obstacles", empty_src, [], "rgba32float", col=0, row=1)
    for i, kind in enumerate(("sources", "forces")):
        merged = glsl3d(bn, b, f"{kind}_merge", sum_src, [zero], "rgba32float", col=1, row=i)
        bn.null(b, kind, merged, 2, i)
    merged = glsl3d(bn, b, "obstacles_merge", union_src, [empty], "rgba32float", col=1, row=2)
    bn.null(b, "obstacles", merged, 2, 2)
    for i, field in enumerate(("vel", "scalars", "pressure")):
        select(bn, b, field, f"op.simbus3d.par.Fieldsource + '/{field}_out'", 3, i)
    return b


def _level_zero(bn, comp, src, level, col):
    return glsl3d(bn, comp, f"zero_l{level}", src, [], "mono32float", col=col, row=8 + level, level=level)


def _vcycle(bn, comp, prefix, shaders, pin, div, obs, zeros, col):
    """One multigrid V-cycle. `pin` is the fine pressure warm start. Returns the fine pressure."""
    pre_n = "parent().par.Smooth"
    post_n = "parent().par.Smooth"
    coarse_n = "parent().par.Coarse"
    pre = []
    rhs = [div]
    for level in range(LEVELS):
        pressure_in = pin if level == 0 else zeros[level]
        passes = coarse_n if level == LEVELS - 1 else pre_n
        sm = glsl3d(bn, comp, f"{prefix}_pre{level}", shaders["mg_smooth"], [pressure_in, rhs[level], obs],
                    "mono32float", col=col, row=level, npasses=passes, level=level)
        pre.append(sm)
        if level < LEVELS - 1:
            rhs.append(glsl3d(bn, comp, f"{prefix}_rhs{level + 1}", shaders["mg_restrict"], [sm, rhs[level], obs],
                              "mono32float", col=col + 1, row=level, level=level + 1))
    correction = pre[-1]
    for level in range(LEVELS - 2, -1, -1):
        prolonged = glsl3d(bn, comp, f"{prefix}_pro{level}", shaders["mg_prolong"], [pre[level], correction],
                           "mono32float", [("uScale", ("parent().par.Coarsegain", 0, 0, 0))], col=col + 2, row=level, filtering="linear", level=level)
        correction = glsl3d(bn, comp, f"{prefix}_post{level}", shaders["mg_smooth"], [prolonged, rhs[level], obs],
                            "mono32float", col=col + 3, row=level, npasses=post_n, level=level)
    return correction


def build_fluid3d(sims, bn):
    f = bn.place(sims.create(td.baseCOMP, "fluid3d"), 7, 0)
    bn.par(f, "Sim", "menu", "Preset", "realtime", menu=["preview", "realtime", "bake"])
    bn.par(f, "Sim", "pulse", "Applypreset", label="Apply Preset")
    bn.par(f, "Sim", "menu", "Solver", "jacobi", menu=["multigrid", "jacobi"])
    bn.par(f, "Sim", "int", "Vcycles", 1, minmax=(1, 2))
    bn.par(f, "Sim", "int", "Smooth", 8, label="Smooth Iterations", minmax=(1, 40))
    bn.par(f, "Sim", "int", "Coarse", 20, label="Coarse Iterations", minmax=(1, 80))
    bn.par(f, "Sim", "float", "Coarsegain", 0.05, label="Coarse Gain", minmax=(-0.25, 1))
    bn.par(f, "Sim", "int", "Jacobi", 16, minmax=(1, 200))
    bn.par(f, "Sim", "toggle", "Cookrender", True, label="Cook Render")
    bn.par(f, "Sim", "int", "Raysteps", 40, minmax=(8, 96))
    bn.par(f, "Sim", "int", "Shadowsteps", 10, minmax=(4, 32))
    bn.par(f, "Sim", "int", "Renderw", 1280, label="Render Width", minmax=(16, 3840))
    bn.par(f, "Sim", "int", "Renderh", 720, label="Render Height", minmax=(16, 2160))
    physics = [("Velocitydecay", 0.4), ("Smokedecay", 0.35), ("Fueldecay", 0.4), ("Maccormack", 0.4),
               ("Burnrate", 5.0), ("Burnheat", 3.5), ("Burnsmoke", 0.5), ("Ignition", 0.12), ("Cooling", 0.9),
               ("Expansion", 0.08), ("Maxtemp", 6.0), ("Buoyancy", 1.6), ("Weight", 0.2), ("Vorticity", 1.2),
               ("Windx", 0.0), ("Windy", 0.0), ("Windz", 0.0), ("Winddrag", 0.0), ("Maxspeed", 6.0)]
    for name, default in physics:
        bn.par(f, "Physics", "float", name, default)
    for name, default in (("Smokedensity", 4.5), ("Fireintensity", 3.5), ("Whitehot", 2.8), ("Obstacleshade", 1.0),
                          ("Background", 0.012)):
        bn.par(f, "Look", "float", name, default)
    bn.par(f, "Look", "rgb", "Smokecolor", [0.45, 0.48, 0.55])
    for name, default in (("Camaz", 0.7), ("Camel", 0.38), ("Camdist", 4.2), ("Camfov", 38.0)):
        bn.par(f, "Camera", "float", name, default)
    for name, default in (("Targetx", 0.0), ("Targety", 0.95), ("Targetz", 0.0)):
        bn.par(f, "Camera", "float", name, default)
    for name, default in (("Lightx", 0.35), ("Lighty", 1.0), ("Lightz", 0.25)):
        bn.par(f, "Camera", "float", name, default)
    bn.extension(f, "fluid3d_ext", "python/fluid3d.py", "Fluid3D")
    bn.pulse_callbacks(f, {"Applypreset": "o.ApplyPreset(o.par.Preset.eval())"})
    bn.shader(f, "bus3d_glsl", "shaders/common/bus3d.glsl", -1, 0)
    names = ("advect", "maccormack", "combust", "forces", "divergence", "project", "jacobi", "mg_smooth",
             "mg_restrict", "mg_prolong", "residual", "shadow", "render_volume")
    src = {n: bn.shader(f, n + "_src", f"shaders/fluid3d/{n}.glsl", -1, i + 1) for i, n in enumerate(names)}
    bn.shader(f, "zero_src", "shaders/common/zero3d.glsl", -1, 20)
    bn.shader(f, "atlas_encode_src", "shaders/common/atlas_encode.glsl", -1, 21)
    bn.shader(f, "atlas_decode_src", "shaders/common/atlas_decode.glsl", -1, 22)

    P = "parent().par."
    sels = {k: select(bn, f, "in_" + k, f"op.simbus3d.path + '/{k}'", 0, i) for i, k in enumerate(("sources", "forces", "obstacles"))}
    obs = sels["obstacles"]
    zeros = {}
    for i, (field, fmt) in enumerate(STATE_FIELDS()):
        zero = glsl3d(bn, f, "zero_" + field, f.op("zero_src"), [], fmt, col=0, row=4 + i)
        load = bn.place(f.create(td.moviefileinTOP, "load_" + field), 1, 4 + i)
        decode = glsl3d(bn, f, "decode_" + field, f.op("atlas_decode_src"), [load], fmt, uniforms=atlas_uniforms(),
                        col=2, row=4 + i, filtering="nearest", include_bus=False)
        init = bn.switch(f, "init_" + field, [zero, decode], 3, 4 + i)
        select(bn, f, "src_" + field, init.path, 4, 4 + i)
        zeros[field] = zero
    vel = f.op("src_vel")
    sc = f.op("src_scalars")
    pr = f.op("src_pressure")
    vdec = (P + "Velocitydecay",) * 3 + (0,)
    sdec = (P + "Smokedecay", 0, P + "Fueldecay", 0)
    mac = (P + "Maccormack",)
    adv_vel = glsl3d(bn, f, "advect_vel", src["advect"], [vel, vel, obs], "rgba32float",
                     [("uDecay", vdec), ("uAdvect", (1,))], 5, 0, filtering="linear")
    mac_vel = glsl3d(bn, f, "mac_vel", src["maccormack"], [vel, adv_vel, vel, obs], "rgba32float",
                     [("uDecay", vdec), ("uMac", mac)], 6, 0, filtering="linear")
    adv_sc = glsl3d(bn, f, "advect_scalars", src["advect"], [sc, vel, obs], "rgba32float",
                    [("uDecay", sdec), ("uAdvect", (0,))], 5, 1, filtering="linear")
    mac_sc = glsl3d(bn, f, "mac_scalars", src["maccormack"], [sc, adv_sc, vel, obs], "rgba32float",
                    [("uDecay", sdec), ("uMac", mac)], 6, 1, filtering="linear")
    combust = glsl3d(bn, f, "combust", src["combust"], [mac_sc, sels["sources"], obs], "rgba16float",
                     [("uBurn", (P + "Burnrate", P + "Burnheat", P + "Burnsmoke", P + "Ignition")),
                      ("uThermo", (P + "Cooling", P + "Expansion", P + "Maxtemp"))], 7, 1, filtering="linear")
    forced = glsl3d(bn, f, "forces", src["forces"], [mac_vel, combust, sels["forces"], obs], "rgba32float",
                    [("uForce", (P + "Buoyancy", P + "Weight", P + "Vorticity")),
                     ("uWind", (P + "Windx", P + "Windy", P + "Windz", P + "Winddrag"))], 8, 0, filtering="linear")
    div = glsl3d(bn, f, "divergence", src["divergence"], [forced, obs, combust], "mono32float", col=9, row=0)
    level_zero_src = f.op("zero_src")
    level_zeros = [None] + [_level_zero(bn, f, level_zero_src, level, 10) for level in range(1, LEVELS)]
    mg = _vcycle(bn, f, "a", src, pr, div, obs, level_zeros, 11)
    mg2 = _vcycle(bn, f, "b", src, mg, div, obs, level_zeros, 15)
    for child in f.children:
        if child.name.startswith("b_"):
            child.bypass = True
    jac = glsl3d(bn, f, "jacobi", src["jacobi"], [pr, div, obs], "mono32float", npasses="parent().par.Jacobi", col=19, row=0)
    jac.bypass = True
    pressure = bn.switch(f, "pressure_sel", [jac, mg, mg2], 20, 0)
    pressure.par.index = 1
    proj = glsl3d(bn, f, "project", src["project"], [forced, pressure, obs], "rgba16float",
                  [("uLimit", (P + "Maxspeed",))], 21, 0)
    states = {"vel": bn.null(f, "vel_state", proj, 22, 0),
              "scalars": bn.null(f, "scalars_state", combust, 22, 1),
              "pressure": bn.null(f, "pressure_state", pressure, 22, 2)}
    formats = dict(STATE_FIELDS())
    for i, field in enumerate(("vel", "scalars", "pressure")):
        h0 = hold(bn, f, f"hold_{field}_0", states[field], formats[field], 23, i)
        hold(bn, f, f"hold_{field}_1", states[field], formats[field], 24, i)
        out = select(bn, f, field + "_out", h0.path, 25, i)
        glsl_pixel(bn, f, field + "_atlas", f.op("atlas_encode_src"), [out], formats[field], atlas_uniforms(),
                   "op('/sims/fluid3d').AtlasWidth()", "op('/sims/fluid3d').AtlasHeight()", 26, i, filtering="nearest")
    glsl3d(bn, f, "div_check", src["divergence"], [f.op("vel_out"), obs, f.op("scalars_out")], "mono32float", col=27, row=0)
    glsl3d(bn, f, "residual_pre", src["residual"], [pr, div, obs], "mono32float", col=27, row=1)
    glsl3d(bn, f, "residual_mg", src["residual"], [mg, div, obs], "mono32float", col=27, row=2)
    glsl3d(bn, f, "residual_jac", src["residual"], [jac, div, obs], "mono32float", col=27, row=3)
    shadow = glsl3d(bn, f, "shadow", src["shadow"], [f.op("scalars_out")], "mono16float",
                    [("uLight", (P + "Lightx", P + "Lighty", P + "Lightz", 0)),
                     ("uLook", (P + "Smokedensity", P + "Shadowsteps", 0, 0))], 28, 1, filtering="linear")
    render_uniforms = [
        ("uBMin", ("op.simbus3d.par.Boundsx0", "op.simbus3d.par.Boundsy0", "op.simbus3d.par.Boundsz0", 0)),
        ("uBMax", ("op.simbus3d.par.Boundsx1", "op.simbus3d.par.Boundsy1", "op.simbus3d.par.Boundsz1", 0)),
        ("uClock", ("op.simclock.par.Dt", "op.simclock.par.Time", "op.simclock.par.Step", "op.simclock.par.Frame")),
        ("uClock2", ("op.simclock.par.Seed", 0, 0, 0)),
        ("uCam", (P + "Camaz", P + "Camel", P + "Camdist", P + "Camfov")),
        ("uTarget", (P + "Targetx", P + "Targety", P + "Targetz", 0)),
        ("uLight", (P + "Lightx", P + "Lighty", P + "Lightz", 0)),
        ("uLook", (P + "Smokedensity", P + "Fireintensity", P + "Whitehot", P + "Obstacleshade")),
        ("uSmoke", (P + "Smokecolorr", P + "Smokecolorg", P + "Smokecolorb", P + "Background")),
        ("uRay", (P + "Raysteps", 0, 0, 0)),
    ]
    render = glsl_pixel(bn, f, "render", src["render_volume"], [f.op("scalars_out"), obs, shadow], "rgba16float",
                        render_uniforms, P + "Renderw", P + "Renderh", 29, 1)
    bn.null(f, "render_out", render, 30, 1)
    return f


def STATE_FIELDS():
    return (("vel", "rgba16float"), ("scalars", "rgba16float"), ("pressure", "mono32float"))


def atlas_uniforms():
    return [
        ("uAtlas", ("op('/sims/fluid3d').AtlasCols()", "op('/sims/fluid3d').AtlasRows()",
                    "op.simbus3d.par.Resw", "op.simbus3d.par.Resh")),
        ("uAtlas2", ("op.simbus3d.par.Resd", 0, 0, 0)),
    ]


def build_demo3d(sims, bn):
    d = bn.place(sims.create(td.baseCOMP, "demo3d"), 8, 0)
    for name, value in (("Emitterx", 0.0), ("Emittery", 0.18), ("Emitterz", 0.0), ("Emitterradius", 0.22),
                        ("Smoke", 0.55), ("Heat", 4.2), ("Fuel", 7.0), ("Flicker", 0.8), ("Noisefreq", 2.4),
                        ("Lift", 3.2)):
        bn.par(d, "Emitter", "float", name, value)
    bn.par(d, "Sphere", "toggle", "Sphereenable", True, label="Enable")
    for name, value in (("Spherex", 0.0), ("Spherey", 0.78), ("Spherez", 0.0), ("Sphereradius", 0.18),
                        ("Sphereamp", 0.1), ("Spherespeed", 0.7)):
        bn.par(d, "Sphere", "float", name, value)
    bn.par(d, "Box", "toggle", "Boxenable", True, label="Enable")
    for name, value in (("Boxx", -0.38), ("Boxy", 0.72), ("Boxz", 0.05), ("Boxhx", 0.14), ("Boxhy", 0.22),
                        ("Boxhz", 0.14), ("Boxamp", 0.18), ("Boxspeed", 0.45)):
        bn.par(d, "Box", "float", name, value)
    bn.par(d, "Capsule", "toggle", "Capsuleenable", True, label="Enable")
    for name, value in (("Capsulex", 0.58), ("Capsuley", 0.7), ("Capsulez", 0.28), ("Capsuleradius", 0.09),
                        ("Capsulehy", 0.28), ("Capsuleamp", 0.12), ("Capsulespeed", 0.6)):
        bn.par(d, "Capsule", "float", name, value)
    bn.par(d, "Mesh", "str", "Meshop", "/sims/demo3d/mesh_geo/sphere")
    bn.par(d, "Mesh", "toggle", "Meshenable", True, label="Enable")
    for name, value in (("Meshx", 0.16), ("Meshy", 1.22), ("Meshz", 0.02), ("Meshscale", 1.0),
                        ("Meshamp", 0.08), ("Meshspeed", 0.4), ("Meshspin", 0.45)):
        bn.par(d, "Mesh", "float", name, value)
    for name, value in (("Sdfminx", -1.0), ("Sdfminy", -1.0), ("Sdfminz", -1.0),
                        ("Sdfmaxx", 1.0), ("Sdfmaxy", 1.0), ("Sdfmaxz", 1.0)):
        bn.par(d, "Mesh", "float", name, value)
    bn.par(d, "Mesh", "int", "Tricount", 0)
    bn.par(d, "Mesh", "int", "Sdfflip", 1)
    bn.par(d, "Mesh", "str", "Meshhash", "")
    bn.par(d, "Mesh", "pulse", "Rebuild", label="Rebuild Mesh")
    bn.extension(d, "demo3d_ext", "python/demo3d.py", "Demo3D")
    bn.pulse_callbacks(d, {"Rebuild": "o.RebuildMesh()"})
    bn.shader(d, "bus3d_glsl", "shaders/common/bus3d.glsl", -1, 0)
    splat = bn.shader(d, "splat_src", "shaders/fluid3d/splat.glsl", -1, 1)
    prim = bn.shader(d, "prim_src", "shaders/fluid3d/primitive.glsl", -1, 2)
    bn.shader(d, "mesh_sdf_src", "shaders/fluid3d/mesh_sdf.glsl", -1, 3)
    bn.shader(d, "mesh_final_src", "shaders/fluid3d/mesh_finalize.glsl", -1, 4)
    bn.shader(d, "mesh_sample_src", "shaders/fluid3d/mesh_sample.glsl", -1, 5)
    bn.shader(d, "zero_src", "shaders/common/zero3d.glsl", -1, 6)
    geo = bn.place(d.create(td.geometryCOMP, "mesh_geo"), 0, 6)
    sphere = geo.create(td.sphereSOP, "sphere")
    sphere.par.type = "poly"
    sphere.par.rows = 6
    sphere.par.cols = 8
    sphere.par.radx = 0.22
    sphere.par.rady = 0.22
    sphere.par.radz = 0.22
    bn.place(d.create(td.boxPOP, "box_pop"), 1, 6)
    bn.place(d.create(td.poptoSOP, "pop_to_sop"), 2, 6)
    tris = bn.place(d.create(td.scriptTOP, "triangles"), 3, 6)
    tris.par.outputresolution = "custom"
    tris.par.resolutionw = 4
    tris.par.resolutionh = 1
    tris.par.format = "rgba32float"
    P = "parent().par."
    t = "op.simclock.par.Time"
    pos = (P + "Emitterx", P + "Emittery", P + "Emitterz", P + "Emitterradius")
    amount = (P + "Smoke", P + "Heat", P + "Fuel", 0)
    glsl3d(bn, d, "emitter_sources", splat, [], "rgba32float",
           [("uSplatPos", pos), ("uSplatAmount", amount), ("uSplatVel", (0, 0, 0, 0)),
            ("uSplatNoise", (P + "Flicker", P + "Noisefreq", 0, 0))], 0, 0)
    glsl3d(bn, d, "emitter_forces", splat, [], "rgba32float",
           [("uSplatPos", pos), ("uSplatAmount", amount), ("uSplatVel", (0, P + "Lift", 0, 1)),
            ("uSplatNoise", (P + "Flicker", P + "Noisefreq", 0, 0))], 0, 1)
    sx = f"{P}Spherex + {P}Sphereamp * math.sin({t} * {P}Spherespeed)"
    sv = f"{P}Sphereamp * {P}Spherespeed * math.cos({t} * {P}Spherespeed)"
    glsl3d(bn, d, "sphere", prim, [], "rgba32float",
           [("uKind", (0, P + "Sphereradius", P + "Sphereenable", 0)),
            ("uCenter", (sx, P + "Spherey", P + "Spherez", 0)),
            ("uHalf", (1, 1, 1, 0)),
            ("uVel", (sv, 0, 0, 0))], 1, 0)
    bz = f"{P}Boxz + {P}Boxamp * math.sin({t} * {P}Boxspeed + 1.3)"
    bv = f"{P}Boxamp * {P}Boxspeed * math.cos({t} * {P}Boxspeed + 1.3)"
    glsl3d(bn, d, "box", prim, [], "rgba32float",
           [("uKind", (1, 0, P + "Boxenable", 0)),
            ("uCenter", (P + "Boxx", P + "Boxy", bz, 0)),
            ("uHalf", (P + "Boxhx", P + "Boxhy", P + "Boxhz", 0)),
            ("uVel", (0, 0, bv, 0))], 1, 1)
    cy = f"{P}Capsuley + {P}Capsuleamp * math.sin({t} * {P}Capsulespeed)"
    cv = f"{P}Capsuleamp * {P}Capsulespeed * math.cos({t} * {P}Capsulespeed)"
    glsl3d(bn, d, "capsule", prim, [], "rgba32float",
           [("uKind", (2, P + "Capsuleradius", P + "Capsuleenable", 0)),
            ("uCenter", (P + "Capsulex", cy, P + "Capsulez", 0)),
            ("uHalf", (0, P + "Capsulehy", 0, 0)),
            ("uVel", (0, cv, 0, 0))], 1, 2)
    # No bus/clock uniforms. A uClock expression dirties this chain every step, and the inactive sdf_hold
    # still recooks a dirty input, which re-runs the jump flood (~90 ms at 128^3) on every frame.
    zero = glsl3d(bn, d, "sdf_zero", d.op("zero_src"), [], "rgba32float", col=2, row=0, include_bus=False)
    info = ("parent().par.Tricount", "parent().par.Sdfflip", 0, 0)
    # Constants, not clock expressions. bus3d.glsl declares these uniforms; leaving them unset warns,
    # and binding uClock to the clock would dirty the jump flood on every frame.
    local = [("uBMin", (-0.5, -0.5, -0.5, 0)), ("uBMax", (0.5, 0.5, 0.5, 0)),
             ("uWalls", (0, 0, 0, 0)), ("uWallsZ", (0, 0, 0, 0)), ("uSponge", (0, 0, 0, 0)),
             ("uClock", (0, 0, 0, 0)), ("uClock2", (0, 0, 0, 0)), ("uInfo", info)]
    sdf = glsl3d(bn, d, "mesh_sdf", d.op("mesh_sdf_src"), [zero, tris], "rgba32float",
                 local, 2, 1, npasses=9, include_bus=False)
    final = glsl3d(bn, d, "mesh_final", d.op("mesh_final_src"), [sdf], "rgba32float", local, 2, 2, include_bus=False)
    cached = hold(bn, d, "sdf_hold", final, "rgba32float", 2, 3)
    mx = f"{P}Meshx + {P}Meshamp * math.sin({t} * {P}Meshspeed)"
    mv = f"{P}Meshamp * {P}Meshspeed * math.cos({t} * {P}Meshspeed)"
    glsl3d(bn, d, "mesh", d.op("mesh_sample_src"), [cached], "rgba32float",
           [("uCenter", (mx, P + "Meshy", P + "Meshz", 0)),
            ("uYaw", (f"{t} * {P}Meshspin", P + "Meshspin", P + "Meshscale", P + "Meshenable")),
            ("uVel", (mv, 0, 0, 0)),
            ("uSdf0", (P + "Sdfminx", P + "Sdfminy", P + "Sdfminz", 0)),
            ("uSdf1", (P + "Sdfmaxx", P + "Sdfmaxy", P + "Sdfmaxz", 0))], 2, 4, filtering="linear")
    return d
