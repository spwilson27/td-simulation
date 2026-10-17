"""GPU particles under /sims/particles, plus the /sims/demo_particles presets.

State is a pair of 2D float textures (position+age, velocity+life) kept in Feedback TOPs, the same
clock contract as the fluids. Two-way coupling is an additive point splat into the bus: this GPU has
no float image atomics. A Geometry COMP's default torus is destroyed; otherwise every Render TOP draws it.
"""

from __future__ import annotations

import td

P = "parent().par."
CLOCK = ("op.simclock.par.Dt", "op.simclock.par.Time", "op.simclock.par.Step", "op.simclock.par.Frame")
CLOCK2 = ("op.simclock.par.Seed", 0, 0, 0)
BOUNDS2 = ("op.simbus.par.Boundsx0", "op.simbus.par.Boundsy0", "op.simbus.par.Boundsx1", "op.simbus.par.Boundsy1")
BMIN = ("op.simbus3d.par.Boundsx0", "op.simbus3d.par.Boundsy0", "op.simbus3d.par.Boundsz0", 0)
BMAX = ("op.simbus3d.par.Boundsx1", "op.simbus3d.par.Boundsy1", "op.simbus3d.par.Boundsz1", 0)
CELL2 = "((op.simbus.par.Boundsx1 - op.simbus.par.Boundsx0) / op.simbus.par.Resw)"
CELL3 = "((op.simbus3d.par.Boundsx1 - op.simbus3d.par.Boundsx0) / op.simbus3d.par.Resw)"


def set_uniforms(top, uniforms):
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


def undock(top):
    for dock in list(top.docked):
        dock.destroy()


def particle_glsl(bn, comp, name, shader, inputs, uniforms, col, row, res_w=None, res_h=None):
    top = bn.place(comp.create(td.glslTOP, name), col, row)
    top.par.glslversion = "glsl460"
    top.par.pixeldat = shader
    top.par.format = "rgba32float"
    top.par.outputresolution = "custom"
    top.par.resolutionw.expr = res_w or (P + "Resx")
    top.par.resolutionh.expr = res_h or (P + "Resy")
    top.par.inputfiltertype = "nearest"
    top.par.filtertype = "nearest"
    top.par.inputextenduv = "zero"
    wire(top, inputs)
    set_uniforms(top, uniforms)
    undock(top)
    return top


def _state_uniforms(write, cell):
    return [
        ("uWrite", (write, 0, 0, 0)),
        ("uPhys", (P + "Drag", P + "Gravity", P + "Buoyancy", P + "Turbulence")),
        ("uTurb", (P + "Turbfreq", 0, 0, 0)),
        ("uField", (P + "Velmode.menuIndex", 0, 0, 0)),
        ("uAnalytic", (P + "Analyticx", P + "Analyticy", P + "Analyticz", P + "Analyticw")),
        ("uEmit", (P + "Emitter.menuIndex", P + "Threshold", 0, 0)),
        ("uEmitPos", (P + "Emitx", P + "Emity", P + "Emitz", P + "Emitradius")),
        ("uEmitBox", (P + "Boxhx", P + "Boxhy", P + "Boxhz", 0)),
        ("uEmitVel", (P + "Emitvelx", P + "Emitvely", P + "Emitvelz", P + "Emitvar")),
        ("uLife", (P + "Lifemin", P + "Lifemax", 0, 0)),
        ("uCollide", (P + "Radius", P + "Restitution", P + "Friction", cell)),
        ("uLimit", (P + "Maxspeed", 0, 0, 0)),
        ("uBounds", BOUNDS2),
        ("uBMin", BMIN),
        ("uBMax", BMAX),
        ("uSpawn", (P + "Spawnx", P + "Spawny", P + "Spawnz", 0)),
        ("uSpawnVel", (P + "Spawnvx", P + "Spawnvy", P + "Spawnvz", 0)),
        ("uClock", CLOCK),
        ("uClock2", CLOCK2),
    ]


def _deposit_uniforms(cell):
    return [
        ("uCouple", (P + "Couplingstrength", P + "Srcheat", 0, 0)),
        ("uPhys", (P + "Drag", 0, 0, 0)),
        ("uField", (P + "Velmode.menuIndex", 0, 0, 0)),
        ("uAnalytic", (P + "Analyticx", P + "Analyticy", P + "Analyticz", P + "Analyticw")),
        ("uBounds", BOUNDS2),
        ("uBMin", BMIN),
        ("uBMax", BMAX),
        ("uScale", (cell, P + "Particlemass", P + "Density", 0)),
    ]


def _sampler(mat, index, name, top):
    mat.par[f"sampler{index}name"] = name
    mat.par[f"sampler{index}top"] = top
    mat.par[f"sampler{index}filter"] = "nearest"


def _destroy_torus(geo):
    torus = geo.op("torus1")
    if torus is not None:
        torus.destroy()


def _render(bn, comp, name, geo, cam, col, row, res_w, res_h, transparency="orderind"):
    top = bn.place(comp.create(td.renderTOP, name), col, row)
    top.par.camera = cam
    top.par.geometry = geo
    top.par.outputresolution = "custom"
    top.par.resolutionw.expr = res_w
    top.par.resolutionh.expr = res_h
    top.par.format = "rgba32float"
    top.par.antialias = "aa1"
    top.par.dither = False
    top.par.transparency = transparency
    top.par.parmcolorspace = "passthrough"
    top.par.premultrgbbyalpha = False
    top.par.bgcolorr = 0
    top.par.bgcolorg = 0
    top.par.bgcolorb = 0
    top.par.bgcolora = 0
    return top


def _point_geo(bn, comp, name, material, col, row):
    geo = bn.place(comp.create(td.geometryCOMP, name), col, row)
    _destroy_torus(geo)
    geo.par.material = material
    geo.par.instancing = True
    geo.par.instancecountmode = "manual"
    geo.par.numinstances.expr = "parent().par.Resx * parent().par.Resy"
    one = geo.create(td.pointgeneratorPOP, "one")
    one.par.numpoints = 1
    one.par.createpointprim = True
    one.render = True
    one.display = True
    return geo


def build_particles(sims, bn):
    f = bn.place(sims.create(td.baseCOMP, "particles"), 9, 0)
    bn.par(f, "Sim", "int", "Resx", 1024, minmax=(64, 2048))
    bn.par(f, "Sim", "int", "Resy", 1024, minmax=(64, 2048))
    bn.par(f, "Sim", "menu", "Bus", "2d", menu=["2d", "3d"])
    bn.par(f, "Sim", "menu", "Velmode", "bus", menu=["bus", "constant", "rotation"])
    bn.par(f, "Sim", "menu", "Emitter", "sphere", menu=["point", "sphere", "box", "fire", "smoke", "tracer"])
    bn.par(f, "Sim", "toggle", "Cookrender", False, label="Cook Render")
    bn.par(f, "Sim", "menu", "Rendermode", "points", menu=["points", "composite", "emissive"])
    for name, default in (("Drag", 2.0), ("Gravity", 0.4), ("Buoyancy", 0.3), ("Turbulence", 0.2), ("Turbfreq", 3.0),
                          ("Maxspeed", 30.0), ("Radius", 0.0), ("Restitution", 0.3), ("Friction", 0.4),
                          ("Couplingstrength", 1.0), ("Particlemass", 1.0), ("Density", 1.0),
                          ("Srcheat", 0.0), ("Smokefrac", 0.2), ("Lifemin", 0.8), ("Lifemax", 1.6),
                          ("Threshold", 0.05)):
        created = bn.par(f, "Physics", "float", name, default)
        if name in ("Couplingstrength", "Particlemass", "Srcheat", "Density"):
            created.normMax = 100.0
            created.clampMin = False
            created.clampMax = False
            created.val = default
    for name, default in (("Emitx", 0.0), ("Emity", 0.25), ("Emitz", 0.0), ("Emitradius", 0.15),
                          ("Boxhx", 0.1), ("Boxhy", 0.1), ("Boxhz", 0.1),
                          ("Emitvelx", 0.0), ("Emitvely", 0.4), ("Emitvelz", 0.0), ("Emitvar", 0.3),
                          ("Analyticx", 1.0), ("Analyticy", 0.0), ("Analyticz", 0.0), ("Analyticw", 1.0),
                          ("Spawnx", -0.4), ("Spawny", 0.6), ("Spawnz", 0.0),
                          ("Spawnvx", 1.0), ("Spawnvy", 0.25), ("Spawnvz", 0.0)):
        bn.par(f, "Emitter", "float", name, default)
    bn.par(f, "Look", "float", "Pointsize", 4.0)
    bn.par(f, "Look", "float", "Colorscale", 1.3)
    bn.par(f, "Look", "int", "Renderw", 1280)
    bn.par(f, "Look", "int", "Renderh", 720)
    for name, default in (("Camaz", 0.7), ("Camel", 0.38), ("Camdist", 4.2), ("Camfov", 38.0),
                          ("Targetx", 0.0), ("Targety", 0.95), ("Targetz", 0.0)):
        bn.par(f, "Look", "float", name, default)
    for name, default in (("Atlasw", 1024), ("Atlash", 2048), ("Atlascols", 8), ("Atlasrows", 16)):
        bn.par(f, "Atlas", "int", name, default)
    bn.extension(f, "particles_ext", "python/particles.py", "Particles")

    names = (
        "uniforms", "hash", "spawn", "init", "step", "integrate2d", "integrate3d", "deposit2d", "deposit3d",
        "splat_vert", "splat_frag", "beauty_vert", "beauty_frag", "split_forces", "split_sources", "extract3d",
        "emissive", "opaque", "clamp_rgb", "splat_sum",
    )
    src = {}
    for i, name in enumerate(names):
        src[name] = bn.shader(f, name + "_glsl", f"shaders/particles/{name}.glsl", -1, i)
    bn.shader(f, "atlas_decode_glsl", "shaders/common/atlas_decode.glsl", -1, len(names))

    res = (P + "Resx", P + "Resy")
    dummy = bn.place(f.create(td.constantTOP, "dummy"), 0, 6)
    dummy.par.outputresolution = "custom"
    dummy.par.resolutionw.expr = res[0]
    dummy.par.resolutionh.expr = res[1]
    dummy.par.format = "rgba32float"

    loops = {}
    for i, field in enumerate(("pos", "pvel")):
        row = i
        shader_top = particle_glsl(bn, f, "spawn_" + field, src["init"], [dummy], _state_uniforms(i, CELL2), 0, row)
        load = bn.place(f.create(td.moviefileinTOP, "load_" + field), 1, row)
        load.par.playmode = "locked"
        load.par.index = 0
        choose = bn.switch(f, "init_" + field, [shader_top, load], 2, row)
        src_sel = bn.select(f, "src_" + field, choose.path, 3, row)
        src_sel.par.top.expr = ""
        src_sel.par.top = choose.path
        loops[field] = src_sel

    bus2 = {k: bn.select(f, "bus2_" + k, f"op.simbus.path + '/{k}'", 5, i) for i, k in enumerate(("vel", "scalars", "obstacles"))}
    bus3 = {k: bn.select(f, "bus3_" + k, f"op.simbus3d.path + '/{k}'", 5, 3 + i) for i, k in enumerate(("vel", "scalars", "obstacles"))}

    def integrate(dim, write, inputs, cell, col, row):
        shader = src["integrate2d"] if dim == "2d" else src["integrate3d"]
        return particle_glsl(bn, f, f"step_{dim}_{'pos' if write == 0 else 'vel'}", shader, inputs,
                             _state_uniforms(write, cell), col, row)

    ins2 = [loops["pos"], loops["pvel"], bus2["vel"], bus2["scalars"], bus2["obstacles"]]
    ins3 = [loops["pos"], loops["pvel"], bus3["vel"], bus3["scalars"], bus3["obstacles"]]
    pos2 = integrate("2d", 0, ins2, CELL2, 6, 0)
    vel2 = integrate("2d", 1, ins2, CELL2, 6, 1)
    pos3 = integrate("3d", 0, ins3, CELL3, 7, 0)
    vel3 = integrate("3d", 1, ins3, CELL3, 7, 1)
    pick_pos = bn.switch(f, "pick_pos", [pos2, pos3], 8, 0)
    pick_vel = bn.switch(f, "pick_vel", [vel2, vel3], 8, 1)
    pick_pos.par.index.expr = P + "Bus.menuIndex"
    pick_vel.par.index.expr = P + "Bus.menuIndex"
    bn.null(f, "pos_state", pick_pos, 9, 0)
    bn.null(f, "pvel_state", pick_vel, 9, 1)
    for i, field in enumerate(("pos", "pvel")):
        mem = bn.null(f, field + "_out", f.op(field + "_state"), 10, i)
        mem.lock = True

    dep2 = particle_glsl(bn, f, "deposit_2d", src["deposit2d"], [f.op("pos_out"), f.op("pvel_out"), bus2["vel"]],
                         _deposit_uniforms(CELL2), 11, 0)
    dep3 = particle_glsl(bn, f, "deposit_3d", src["deposit3d"], [f.op("pos_out"), f.op("pvel_out"), bus3["vel"]],
                         _deposit_uniforms(CELL3), 11, 1)
    pick_dep = bn.switch(f, "pick_deposit", [dep2, dep3], 12, 0)
    pick_dep.par.index.expr = P + "Bus.menuIndex"
    bn.null(f, "deposit", pick_dep, 13, 0)

    cam = bn.place(f.create(td.cameraCOMP, "cam"), 14, 3)
    cam.par.projection = "ortho"
    cam.par.orthowidth = 2
    cam.par.tz = 5

    splat_mat = bn.place(f.create(td.glslMAT, "splat_mat"), 14, 4)
    splat_mat.par.glslversion = "glsl460"
    splat_mat.par.vdat = src["splat_vert"]
    splat_mat.par.pdat = src["splat_frag"]
    splat_mat.par.depthtest = False
    splat_mat.par.depthwriting = False
    splat_mat.par.blending = True
    splat_mat.par.blendop = "add"
    splat_mat.par.srcblend = "one"
    splat_mat.par.destblend = "one"
    splat_mat.par.blendopa = "add"
    splat_mat.par.srcblenda = "one"
    splat_mat.par.destblenda = "one"
    splat_mat.seq.sampler.numBlocks = 2
    _sampler(splat_mat, 0, "sReact", f.op("deposit"))
    _sampler(splat_mat, 1, "sState", f.op("pos_out"))
    set_uniforms(splat_mat, [
        ("uMode", (P + "Bus.menuIndex", 0, 0, 0)),
        ("uBounds", BOUNDS2),
        ("uBMin", BMIN),
        ("uBMax", BMAX),
        ("uAtlas", (P + "Atlascols", P + "Atlasrows", "op.simbus3d.par.Resw", "op.simbus3d.par.Resh")),
        ("uAtlas2", (P + "Atlasw", P + "Atlash", "op.simbus3d.par.Resd", 0)),
        ("uRes", ("op.simbus.par.Resw", "op.simbus.par.Resh", 0, 0)),
    ])
    splat_w = "op.simbus.par.Resw if parent().par.Bus.menuIndex == 0 else parent().par.Atlasw"
    splat_h = "op.simbus.par.Resh if parent().par.Bus.menuIndex == 0 else parent().par.Atlash"
    splat_geo = _point_geo(bn, f, "splat_geo", splat_mat, 14, 0)
    _render(bn, f, "splat_render", splat_geo, cam, 15, 0, splat_w, splat_h)
    # 2D coupling is summed in integer shared memory so the bake is byte-stable. The point splat stays for 3D.
    splat_sum = bn.place(f.create(td.glslTOP, "splat_sum"), 15, 5)
    splat_sum.par.mode = "compute"
    splat_sum.par.computedat = src["splat_sum"]
    splat_sum.par.glslversion = "glsl460"
    splat_sum.par.format = "rgba32float"
    splat_sum.par.outputresolution = "custom"
    # One workgroup. A bus-sized dispatch races and the coupled bake stops being byte-stable.
    splat_sum.par.resolutionw = 32
    splat_sum.par.resolutionh = 32
    splat_sum.par.outputaccess = "writeonly"
    splat_sum.par.inputfiltertype = "nearest"
    splat_sum.par.filtertype = "nearest"
    wire(splat_sum, [f.op("deposit"), f.op("pos_out")])
    set_uniforms(splat_sum, [
        ("uMode", (0, 0, 0, 0)),
        ("uBounds", BOUNDS2),
        ("uBMin", BMIN),
        ("uBMax", BMAX),
        ("uAtlas", (P + "Atlascols", P + "Atlasrows", "op.simbus3d.par.Resw", "op.simbus3d.par.Resh")),
        ("uAtlas2", (P + "Atlasw", P + "Atlash", "op.simbus3d.par.Resd", 0)),
        ("uRes", ("op.simbus.par.Resw", "op.simbus.par.Resh", 0, 0)),
        ("uQuant", (1000000.0, 0, 0, 0)),
    ])
    undock(splat_sum)

    forces = particle_glsl(bn, f, "forces_2d", src["split_forces"], [splat_sum], [("uUnused", (0, 0, 0, 0))],
                           16, 0, "me.inputs[0].width", "me.inputs[0].height")
    sources = particle_glsl(bn, f, "sources_2d", src["split_sources"], [splat_sum],
                            [("uSrc", (P + "Smokefrac", 0, 0, 0))], 16, 1, "me.inputs[0].width", "me.inputs[0].height")
    for split in (forces, sources):
        split.par.outputresolution = "useinput"

    vol = bn.place(f.create(td.glslTOP, "splat_vol"), 17, 0)
    vol.par.mode = "compute"
    vol.par.computedat = f.op("atlas_decode_glsl")
    vol.par.glslversion = "glsl460"
    vol.par.type = "texture3d"
    vol.par.depth = "custom"
    vol.par.format = "rgba32float"
    vol.par.outputresolution = "custom"
    vol.par.resolutionw.expr = "op.simbus3d.par.Resw"
    vol.par.resolutionh.expr = "op.simbus3d.par.Resh"
    vol.par.customdepth.expr = "op.simbus3d.par.Resd"
    vol.par.outputaccess = "writeonly"
    vol.par.inputfiltertype = "nearest"
    vol.par.filtertype = "nearest"
    f.op("splat_render").outputConnectors[0].connect(vol.inputConnectors[0])
    set_uniforms(vol, [
        ("uAtlas", (P + "Atlascols", P + "Atlasrows", "op.simbus3d.par.Resw", "op.simbus3d.par.Resh")),
        ("uAtlas2", ("op.simbus3d.par.Resd", 0, 0, 0)),
    ])
    undock(vol)

    def extract(name, kind, row):
        top = bn.place(f.create(td.glslTOP, name), 18, row)
        top.par.mode = "compute"
        top.par.computedat = f.op("extract3d_glsl")
        top.par.glslversion = "glsl460"
        top.par.type = "texture3d"
        top.par.depth = "custom"
        top.par.format = "rgba32float"
        top.par.outputresolution = "custom"
        top.par.resolutionw.expr = "op.simbus3d.par.Resw"
        top.par.resolutionh.expr = "op.simbus3d.par.Resh"
        top.par.customdepth.expr = "op.simbus3d.par.Resd"
        top.par.outputaccess = "writeonly"
        top.par.inputfiltertype = "nearest"
        top.par.filtertype = "nearest"
        vol.outputConnectors[0].connect(top.inputConnectors[0])
        set_uniforms(top, [("uKind", (kind, 0, 0, 0)), ("uSrc", (P + "Smokefrac", 0, 0, 0))])
        undock(top)
        return top

    extract("forces_3d", 0, 0)
    extract("sources_3d", 1, 1)

    beauty_mat = bn.place(f.create(td.glslMAT, "beauty_mat"), 14, 5)
    beauty_mat.par.glslversion = "glsl460"
    beauty_mat.par.vdat = src["beauty_vert"]
    beauty_mat.par.pdat = src["beauty_frag"]
    beauty_mat.par.depthtest = False
    beauty_mat.par.depthwriting = False
    beauty_mat.par.blending = True
    beauty_mat.par.blendop = "add"
    beauty_mat.par.srcblend = "one"
    beauty_mat.par.destblend = "one"
    beauty_mat.seq.sampler.numBlocks = 2
    _sampler(beauty_mat, 0, "sState", f.op("pos_out"))
    _sampler(beauty_mat, 1, "sVel", f.op("pvel_out"))
    set_uniforms(beauty_mat, [
        ("uMode", (P + "Bus.menuIndex", 0, 0, 0)),
        ("uPoint", (P + "Pointsize", 0, 0, 0)),
        ("uLook", (P + "Colorscale", 0, 0, 0)),
        ("uBounds", BOUNDS2),
        ("uBMin", BMIN),
        ("uBMax", BMAX),
        ("uCam", (P + "Camaz", P + "Camel", P + "Camdist", P + "Camfov")),
        ("uTarget", (P + "Targetx", P + "Targety", P + "Targetz", 0)),
        ("uRes", (P + "Renderw", P + "Renderh", 0, 0)),
    ])
    beauty_geo = _point_geo(bn, f, "beauty_geo", beauty_mat, 14, 1)
    # Blend mode clears the target to the background. Order-independent drawing left uncovered
    # float pixels uninitialised, which showed up as large negative alpha.
    beauty_render = _render(bn, f, "beauty_render", beauty_geo, cam, 15, 1, P + "Renderw", P + "Renderh",
                            transparency="blend")
    bn.null(f, "render_out", beauty_render, 16, 2)

    fluid = bn.select(f, "fluid_in",
                      "'/sims/fluid2d/render_out' if parent().par.Bus.menuIndex == 0 else '/sims/fluid3d/render_out'",
                      15, 2)
    fit = bn.place(f.create(td.fitTOP, "fluid_fit"), 16, 3)
    fluid.outputConnectors[0].connect(fit.inputConnectors[0])
    fit.par.outputresolution = "custom"
    fit.par.resolutionw.expr = P + "Renderw"
    fit.par.resolutionh.expr = P + "Renderh"
    added = bn.place(f.create(td.addTOP, "composite"), 17, 3)
    fit.outputConnectors[0].connect(added.inputConnectors[0])
    clamped = bn.place(f.create(td.glslTOP, "beauty_clamp"), 16, 2)
    clamped.par.glslversion = "glsl460"
    clamped.par.pixeldat = src["clamp_rgb"]
    clamped.par.format = "rgba16float"
    clamped.par.outputresolution = "useinput"
    beauty_render.outputConnectors[0].connect(clamped.inputConnectors[0])
    bn.null(f, "beauty", clamped, 17, 2)
    undock(clamped)
    clamped.outputConnectors[0].connect(added.inputConnectors[1])
    bn.null(f, "composite_out", added, 18, 3)
    picture = bn.place(f.create(td.glslTOP, "picture"), 19, 3)
    picture.par.glslversion = "glsl460"
    picture.par.pixeldat = src["opaque"]
    picture.par.format = "rgba8fixed"
    picture.par.outputresolution = "useinput"
    added.outputConnectors[0].connect(picture.inputConnectors[0])
    undock(picture)
    bn.null(f, "picture_out", picture, 20, 3)

    black = bn.place(f.create(td.constantTOP, "emissive_dummy"), 15, 4)
    black.par.format = "rgba16float"
    emit = bn.place(f.create(td.glslTOP, "emissive"), 16, 4)
    emit.par.glslversion = "glsl460"
    emit.par.pixeldat = src["emissive"]
    emit.par.format = "rgba16float"
    emit.par.outputresolution = "custom"
    emit.par.resolutionw.expr = P + "Renderw"
    emit.par.resolutionh.expr = P + "Renderh"
    wire(emit, [black, f.op("sources_3d")])
    set_uniforms(emit, [
        ("uBMin", BMIN),
        ("uBMax", BMAX),
        ("uCam", (P + "Camaz", P + "Camel", P + "Camdist", P + "Camfov")),
        ("uTarget", (P + "Targetx", P + "Targety", P + "Targetz", 0)),
        ("uGain", (P + "Colorscale", 0, 0, 0)),
    ])
    undock(emit)
    glow = bn.place(f.create(td.addTOP, "emissive_add"), 17, 4)
    fit.outputConnectors[0].connect(glow.inputConnectors[0])
    emit.outputConnectors[0].connect(glow.inputConnectors[1])
    bn.null(f, "emissive_out", glow, 18, 4)
    return f


def build_demo_particles(sims, bn):
    d = bn.place(sims.create(td.baseCOMP, "demo_particles"), 10, 0)
    bn.par(d, "Demo", "menu", "Demo", "embers2d", menu=["embers2d", "sparks3d"])
    bn.par(d, "Demo", "pulse", "Apply", label="Apply Demo")
    bn.extension(d, "demo_particles_ext", "python/demo_particles.py", "DemoParticles")
    bn.pulse_callbacks(d, {"Apply": "o.ApplyDemo(o.par.Demo.eval())"})
    return d
