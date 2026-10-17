"""Build the /sims network inside TouchDesigner from this repo's sources. Run by scripts/build_project.py.

    /sims               Root parameter (repo folder, default project.folder), parent shortcut "sims"
      clock             SimClockExt, global shortcut op.simclock; Execute DAT ticks it at frame start
      bus               FieldBus, global shortcut op.simbus; merges publishers, routes the active fluid fields
      fluid2d           Fluid2D smoke/fire solver
      demo              publishers: emitter (sources + forces) and a moving disc obstacle
      baker             Baker (writes cache/fluid2d by default)
      cache_reader      CacheReader (plays a cache back; can be the bus field source)
      particles         GPU particles (position/velocity textures). Not in the clock until a demo enables them.
      demo_particles    Presets: 2D embers and 3D sparks.

Every shader and extension is a Text DAT synced to its file in the repo, so edits on disk show up in TD.
"""

from __future__ import annotations

import td

CLOCK_UNIFORMS = ("op.simclock.par.Dt", "op.simclock.par.Time", "op.simclock.par.Step", "op.simclock.par.Frame")
CLOCK2_UNIFORMS = ("op.simclock.par.Seed", 0, 0, 0)  # uClock2.x; yzw unused. See shaders/common/bus.glsl.
BOUNDS_UNIFORMS = ("op.simbus.par.Boundsx0", "op.simbus.par.Boundsy0", "op.simbus.par.Boundsx1", "op.simbus.par.Boundsy1")
WALL_UNIFORMS = ("op.simbus.par.Wallleft", "op.simbus.par.Wallright", "op.simbus.par.Wallbottom", "op.simbus.par.Walltop")
SPONGE_UNIFORMS = ("op.simbus.par.Spongewidth", "op.simbus.par.Spongerate")


# -- generic helpers ------------------------------------------------------------------------------------------
def place(o, col, row):
    o.nodeX, o.nodeY = col * 220, -row * 160
    return o


def par(comp, page, kind, name, default=None, label=None, size=1, menu=None, minmax=None, readonly=False):
    pg = next((p for p in comp.customPages if p.name == page), None) or comp.appendCustomPage(page)
    label = label or name
    if kind == "float":
        group = pg.appendFloat(name, label=label, size=size)
    elif kind == "int":
        group = pg.appendInt(name, label=label, size=size)
    elif kind == "toggle":
        group = pg.appendToggle(name, label=label)
    elif kind == "pulse":
        group = pg.appendPulse(name, label=label)
    elif kind == "str":
        group = pg.appendStr(name, label=label)
    elif kind == "folder":
        group = pg.appendFolder(name, label=label)
    elif kind == "menu":
        group = pg.appendMenu(name, label=label)
    elif kind == "rgb":
        group = pg.appendRGB(name, label=label)
    else:
        raise ValueError(kind)
    pars = list(group)
    if kind == "menu":
        pars[0].menuNames = menu
        pars[0].menuLabels = menu
    defaults = default if isinstance(default, (list, tuple)) else [default] * len(pars)
    for p, d in zip(pars, defaults):
        if minmax:
            p.normMin, p.normMax = minmax
        if d is not None:
            p.default = d
            p.val = d
        p.readOnly = readonly
    return pars[0]


def synced_text(comp, name, rel, col=0, row=0, dat_type=None):
    d = comp.create(dat_type or td.textDAT, name)
    d.par.file.expr = f"parent.sims.par.Root.eval() + '/{rel}'"
    d.par.syncfile = True
    d.par.loadonstartpulse.pulse()
    return place(d, col, row)


def extension(comp, dat_name, rel, cls):
    synced_text(comp, dat_name, rel, -2, 0)
    comp.par.ext0object = f"op('./{dat_name}').module.{cls}(me)"
    comp.par.ext0name = cls
    comp.par.ext0promote = True


def pulse_callbacks(comp, mapping):
    """Parameter Execute DAT calling owner methods on pulses: {par name: python statement using `o`}."""
    pe = place(comp.create(td.parameterexecuteDAT, "pulses"), -2, 1)
    pe.par.op = comp.path
    pe.par.pars = " ".join(mapping)
    pe.par.valuechange = False
    pe.par.onpulse = True
    body = "\n".join(f"    if par.name == {name!r}:\n        {stmt}" for name, stmt in mapping.items())
    pe.text = "def onPulse(par):\n    o = par.owner\n" + body + "\n    return\n"
    return pe


def bus_res(top):
    top.par.outputresolution = "custom"
    top.par.resolutionw.expr = "op.simbus.par.Resw"
    top.par.resolutionh.expr = "op.simbus.par.Resh"
    return top


def glsl(comp, name, shader_dat, inputs, fmt, uniforms=(), col=0, row=0, npasses=None, extend="zero"):
    g = place(comp.create(td.glslTOP, name), col, row)
    g.par.glslversion = "glsl460"
    g.par.pixeldat = shader_dat
    g.par.format = fmt
    g.par.inputextenduv = extend
    g.par.filtertype = "linear"
    g.par.inputfiltertype = "linear"
    bus_res(g)
    g.setInputs(list(inputs[:3]))   # the GLSL TOP takes 3 wired inputs; `tops` appends the rest in order
    g.par.tops = " ".join(src.path for src in inputs[3:])
    all_uniforms = [("uBounds", BOUNDS_UNIFORMS), ("uClock", CLOCK_UNIFORMS), ("uClock2", CLOCK2_UNIFORMS),
                    ("uWalls", WALL_UNIFORMS), ("uSponge", SPONGE_UNIFORMS)] + list(uniforms)
    g.seq.vec.numBlocks = len(all_uniforms)
    for i, (uname, values) in enumerate(all_uniforms):
        getattr(g.par, f"vec{i}name").val = uname
        for axis, v in zip("xyzw", list(values) + [0] * (4 - len(values))):
            p = getattr(g.par, f"vec{i}value{axis}")
            if isinstance(v, str):
                p.expr = v
            else:
                p.val = v
    if npasses is not None:
        if isinstance(npasses, str):
            g.par.npasses.expr = npasses
        else:
            g.par.npasses = npasses
    for dock in g.docked:
        dock.destroy()
    return g


def null(comp, name, src, col, row):
    n = place(comp.create(td.nullTOP, name), col, row)
    src.outputConnectors[0].connect(n.inputConnectors[0])
    return n


def hold(comp, name, src, fmt, col, row):
    """Cache TOP that Fluid2D.Step snapshots into. Inactive afterwards, so later reads do not recook the solver."""
    c = place(comp.create(td.cacheTOP, name), col, row)
    src.outputConnectors[0].connect(c.inputConnectors[0])
    c.par.active = False
    c.par.alwayscook = False
    c.par.cachesize = 1
    c.par.outputindex = 0
    c.par.format = fmt
    c.par.outputresolution = "useinput"
    c.par.inputfiltertype = "nearest"
    c.par.filtertype = "nearest"
    return c


def constant(comp, name, fmt, rgba, col, row):
    c = bus_res(place(comp.create(td.constantTOP, name), col, row))
    c.par.format = fmt
    c.par.colorr, c.par.colorg, c.par.colorb = rgba[:3]
    c.par.alpha = rgba[3]
    return c


def select(comp, name, expr, col, row):
    s = place(comp.create(td.selectTOP, name), col, row)
    s.par.top.expr = expr
    return s


def switch(comp, name, inputs, col, row):
    s = place(comp.create(td.switchTOP, name), col, row)
    for i, src in enumerate(inputs):
        src.outputConnectors[0].connect(s.inputConnectors[i])
    return s


def shader(comp, name, rel, col=-1, row=0):
    return synced_text(comp, name, rel, col, row)


# -- components -----------------------------------------------------------------------------------------------
def build_clock(sims):
    c = place(sims.create(td.baseCOMP, "clock"), 0, 0)
    c.par.opshortcut = "simclock"
    par(c, "Clock", "float", "Fps", 60.0, minmax=(1, 120))
    par(c, "Clock", "int", "Substeps", 1, minmax=(1, 16))
    par(c, "Clock", "int", "Startframe", 0)
    par(c, "Clock", "menu", "Mode", "realtime", menu=["realtime", "bake"])
    par(c, "Clock", "int", "Seed", 0)
    par(c, "Clock", "toggle", "Running", True)
    par(c, "Clock", "pulse", "Resetpulse", label="Reset")
    par(c, "Clock", "str", "Sims", "/sims/fluid2d")
    par(c, "Clock", "str", "Listeners", "")
    for name, kind in (("Step", "int"), ("Frame", "int"), ("Substep", "int"), ("Time", "float"), ("Dt", "float")):
        par(c, "Output", kind, name, 0, readonly=True)
    c.par.Dt = 1 / 60
    extension(c, "clock_ext", "python/simclock.py", "SimClockExt")
    tick = place(c.create(td.executeDAT, "tick"), 0, 1)
    tick.text = "def onFrameStart(frame):\n    op.simclock.Tick()\n    return\n"
    tick.par.framestart = True
    tick.par.active = True
    pulse_callbacks(c, {"Resetpulse": "o.RequestReset()"})
    return c


def build_bus(sims):
    b = place(sims.create(td.baseCOMP, "bus"), 1, 0)
    b.par.opshortcut = "simbus"
    par(b, "Grid", "int", "Resw", 1024, minmax=(64, 4096))
    par(b, "Grid", "int", "Resh", 1024, minmax=(64, 4096))
    for name, d in (("Boundsx0", -1.0), ("Boundsy0", 0.0), ("Boundsx1", 1.0), ("Boundsy1", 2.0)):
        par(b, "Grid", "float", name, d)
    for name, d in (("Wallleft", True), ("Wallright", True), ("Wallbottom", True), ("Walltop", False)):
        par(b, "Grid", "toggle", name, d)
    par(b, "Grid", "float", "Spongewidth", 0.35, label="Open Side Band Width")
    par(b, "Grid", "float", "Spongerate", 4.0, label="Open Side Fade Rate")
    par(b, "Fields", "str", "Fieldsource", "/sims/fluid2d")
    extension(b, "bus_ext", "python/simbus.py", "FieldBus")
    shader(b, "bus_glsl", "shaders/common/bus.glsl", -1, 2)
    sum_src = shader(b, "sum_src", "shaders/common/sum.glsl", -1, 3)
    union_src = shader(b, "union_src", "shaders/common/sdf_union.glsl", -1, 4)
    empty_src = shader(b, "empty_src", "shaders/common/empty_obstacles.glsl", -1, 5)
    t = place(b.create(td.tableDAT, "publishers"), -2, 2)
    t.clear()
    t.appendRow(["kind", "path"])

    zero = constant(b, "zero", "rgba32float", (0, 0, 0, 0), 0, 0)
    empty = glsl(b, "empty_obstacles", empty_src, [], "rgba32float", col=0, row=2)
    for i, kind in enumerate(("sources", "forces")):
        m = glsl(b, f"{kind}_merge", sum_src, [zero], "rgba32float", col=1, row=i)
        null(b, kind, m, 2, i)
    m = glsl(b, "obstacles_merge", union_src, [empty], "rgba32float", col=1, row=2)
    null(b, "obstacles", m, 2, 2)
    for i, field in enumerate(("vel", "scalars", "pressure")):
        select(b, field, f"op.simbus.par.Fieldsource.eval() + '/{field}_out'", 3, i)
    return b


def build_fluid2d(sims):
    f = place(sims.create(td.baseCOMP, "fluid2d"), 2, 0)
    par(f, "Sim", "menu", "Preset", "realtime", menu=["preview", "realtime", "bake"])
    par(f, "Sim", "pulse", "Applypreset", label="Apply Preset")
    par(f, "Sim", "int", "Jacobi", 30, minmax=(1, 200))
    par(f, "Sim", "toggle", "Cookrender", True, label="Cook Render")
    physics = [("Velocitydecay", 0.0), ("Smokedecay", 0.25), ("Fueldecay", 0.5), ("Maccormack", 1.0), ("Burnrate", 4.0),
               ("Burnheat", 2.5), ("Burnsmoke", 0.35), ("Ignition", 0.15), ("Cooling", 1.5), ("Expansion", 0.08),
               ("Maxtemp", 6.0), ("Buoyancy", 1.4), ("Weight", 0.1), ("Vorticity", 2.5), ("Windx", 0.0),
               ("Windy", 0.0), ("Winddrag", 0.0), ("Maxspeed", 20.0)]
    for name, d in physics:
        par(f, "Physics", "float", name, d)
    for name, d in (("Smokedensity", 1.8), ("Fireintensity", 1.4), ("Whitehot", 2.0), ("Obstacleopacity", 0.8),
                    ("Background", 0.02)):
        par(f, "Look", "float", name, d)
    par(f, "Look", "rgb", "Smokecolor", [0.55, 0.55, 0.6])
    extension(f, "fluid2d_ext", "python/fluid2d.py", "Fluid2D")
    pulse_callbacks(f, {"Applypreset": "o.ApplyPreset(o.par.Preset.eval())"})

    src = {n: shader(f, n + "_src", f"shaders/fluid2d/{n}.glsl", -1, i + 3) for i, n in enumerate(
        ("advect", "maccormack", "combust", "vorticity", "divergence", "jacobi", "project", "render_fire", "quantize"))}
    shader(f, "bus_glsl", "shaders/common/bus.glsl", -1, 2)

    P = "parent().par."
    sel = {k: select(f, "in_" + k, f"op.simbus.path + '/{k}'", 0, i) for i, k in enumerate(("sources", "forces", "obstacles"))}
    obs = sel["obstacles"]
    loops = {}
    for i, (field, fmt) in enumerate((("vel", "rg32float"), ("scalars", "rgba16float"), ("pressure", "mono32float"))):
        row = 4 + i
        zero = constant(f, "zero_" + field, fmt, (0, 0, 0, 0), 0, row)
        load = place(f.create(td.moviefileinTOP, "load_" + field), 1, row)
        load.par.playmode = "locked"
        load.par.index = 0
        init = switch(f, "init_" + field, [zero, load], 2, row)
        fb = place(f.create(td.feedbackTOP, "fb_" + field), 3, row)
        init.outputConnectors[0].connect(fb.inputConnectors[0])
        fb.par.top = field + "_state"
        loops[field] = switch(f, "loop_" + field, [fb, init], 4, row)

    vel, sc, pr = loops["vel"], loops["scalars"], loops["pressure"]
    vdec = (P + "Velocitydecay", P + "Velocitydecay", 0, 0)
    sdec = (P + "Smokedecay", 0, P + "Fueldecay", 0)
    mac = (P + "Maccormack",)
    adv_vel = glsl(f, "advect_vel", src["advect"], [vel, vel, obs], "rg32float", [("uDecay", vdec), ("uAdvect", (1,))], 5, 0)
    mac_vel = glsl(f, "mac_vel", src["maccormack"], [vel, adv_vel, vel, obs], "rg32float", [("uDecay", vdec), ("uMac", mac)], 6, 0)
    adv_sc = glsl(f, "advect_scalars", src["advect"], [sc, vel, obs], "rgba32float", [("uDecay", sdec), ("uAdvect", (0,))], 5, 1)
    mac_sc = glsl(f, "mac_scalars", src["maccormack"], [sc, adv_sc, vel, obs], "rgba32float", [("uDecay", sdec), ("uMac", mac)], 6, 1)
    combust = glsl(f, "combust", src["combust"], [mac_sc, sel["sources"], obs], "rgba16float",
                   [("uBurn", (P + "Burnrate", P + "Burnheat", P + "Burnsmoke", P + "Ignition")),
                    ("uThermo", (P + "Cooling", P + "Expansion", P + "Maxtemp"))], 7, 1)
    forces = glsl(f, "forces", src["vorticity"], [mac_vel, combust, sel["forces"], obs], "rg32float",
                  [("uForce", (P + "Buoyancy", P + "Weight", P + "Vorticity")),
                   ("uWind", (P + "Windx", P + "Windy", P + "Winddrag"))], 8, 0)
    div = glsl(f, "divergence", src["divergence"], [forces, obs, combust], "mono32float", col=9, row=0)
    jac = glsl(f, "jacobi", src["jacobi"], [pr, div, obs], "mono32float", npasses="parent().par.Jacobi", col=10, row=1)
    proj = glsl(f, "project", src["project"], [forces, jac, obs], "rg32float", [("uLimit", (P + "Maxspeed",))], 11, 0)
    qvel = glsl(f, "q_vel", src["quantize"], [proj], "rg32float", [("uStep", (1.0e5,))], 11, 3)
    qsc = glsl(f, "q_scalars", src["quantize"], [combust], "rgba16float", [("uStep", (1.0e4,))], 11, 4)
    qpr = glsl(f, "q_pressure", src["quantize"], [jac], "mono32float", [("uStep", (1.0e5,))], 11, 5)
    vel_state = null(f, "vel_state", qvel, 12, 0)
    sc_state = null(f, "scalars_state", qsc, 12, 1)
    pr_state = null(f, "pressure_state", qpr, 12, 2)
    hold(f, "vel_out", vel_state, "rg32float", 13, 0)
    sc_out = hold(f, "scalars_out", sc_state, "rgba16float", 13, 1)
    hold(f, "pressure_out", pr_state, "mono32float", 13, 2)
    render = glsl(f, "render", src["render_fire"], [sc_out, obs], "rgba16float",
                  [("uLook", (P + "Smokedensity", P + "Fireintensity", P + "Whitehot", P + "Obstacleopacity")),
                   ("uSmokeColor", (P + "Smokecolorr", P + "Smokecolorg", P + "Smokecolorb", P + "Background"))], 14, 1)
    null(f, "render_out", render, 15, 1)
    return f


def build_demo(sims):
    d = place(sims.create(td.baseCOMP, "demo"), 3, 0)
    for name, v in (("Emitterx", 0.0), ("Emittery", 0.18), ("Emitterradius", 0.12), ("Smoke", 0.15), ("Heat", 2.0),
                    ("Fuel", 6.0), ("Flicker", 0.6), ("Noisefreq", 6.0), ("Lift", 3.0)):
        par(d, "Emitter", "float", name, v)
    par(d, "Disc", "toggle", "Discenable", True, label="Enable")
    for name, v in (("Discradius", 0.16), ("Discamplitude", 0.45), ("Discspeed", 0.7), ("Discy", 1.1)):
        par(d, "Disc", "float", name, v)
    shader(d, "bus_glsl", "shaders/common/bus.glsl", -1, 2)
    splat = shader(d, "splat_src", "shaders/fluid2d/splat.glsl", -1, 3)
    disc_src = shader(d, "disc_src", "shaders/fluid2d/obstacle_disc.glsl", -1, 4)
    P = "parent().par."
    pos = (P + "Emitterx", P + "Emittery", P + "Emitterradius", P + "Flicker")
    amount = (P + "Smoke", P + "Heat", P + "Fuel", 0)
    glsl(d, "emitter_sources", splat, [], "rgba32float",
         [("uSplatPos", pos), ("uSplatAmount", amount), ("uSplatVel", (0, 0, 0, P + "Noisefreq"))], 0, 0)
    glsl(d, "emitter_forces", splat, [], "rgba32float",
         [("uSplatPos", pos), ("uSplatAmount", amount), ("uSplatVel", (0, P + "Lift", 1, P + "Noisefreq"))], 0, 1)
    t = "op.simclock.par.Time"
    glsl(d, "disc", disc_src, [], "rgba32float",
         [("uDisc", (f"{P}Discamplitude * math.sin({t} * {P}Discspeed)", P + "Discy", f"{P}Discradius * {P}Discenable")),
          ("uDiscVel", (f"{P}Discamplitude * {P}Discspeed * math.cos({t} * {P}Discspeed)", 0))], 0, 2)
    return d


def build_baker(sims):
    b = place(sims.create(td.baseCOMP, "baker"), 4, 0)
    par(b, "Bake", "str", "Sim", "/sims/fluid2d")
    par(b, "Bake", "str", "Groupsims", "", label="Group Sims")
    par(b, "Bake", "str", "Cachedir", "cache/fluid2d")
    par(b, "Bake", "int", "Framestart", 1)
    par(b, "Bake", "int", "Frameend", 120)
    par(b, "Bake", "int", "Warmup", 0)
    par(b, "Bake", "toggle", "Resume", True)
    par(b, "Bake", "toggle", "Asyncsave", False, label="Async Save")
    par(b, "Bake", "toggle", "Pauseafter", True, label="Pause After Bake")
    par(b, "Bake", "pulse", "Bake")
    par(b, "Bake", "pulse", "Cancel")
    par(b, "Bake", "pulse", "Clearcache", label="Clear Cache")
    par(b, "Bake", "str", "Status", "idle", readonly=True)
    par(b, "Bake", "float", "Progress", 0.0, readonly=True)
    par(b, "Bake", "float", "Eta", 0.0, label="ETA (s)", readonly=True)
    extension(b, "baker_ext", "python/baker.py", "Baker")
    pulse_callbacks(b, {"Bake": "o.Bake()", "Cancel": "o.Cancel()", "Clearcache": "o.ClearCache()"})
    return b


def build_cache_reader(sims):
    r = place(sims.create(td.baseCOMP, "cache_reader"), 5, 0)
    par(r, "Cache", "str", "Cachedir", "cache/fluid2d")
    par(r, "Cache", "str", "Member", "")
    fr = par(r, "Cache", "float", "Frame", 1.0)
    fr.expr = "op.simclock.par.Frame"
    par(r, "Cache", "toggle", "Interpolate", True)
    par(r, "Cache", "pulse", "Reload")
    par(r, "Cache", "float", "Gain", [1.0, 1.0, 1.0, 0.0], size=4)
    par(r, "Cache", "str", "Status", "", readonly=True)
    par(r, "Cache", "int", "Framestart", 0, readonly=True)
    par(r, "Cache", "int", "Frameend", 0, readonly=True)
    extension(r, "reader_ext", "python/cache_reader.py", "CacheReader")
    pulse_callbacks(r, {"Reload": "o.Reload()"})
    shader(r, "bus_glsl", "shaders/common/bus.glsl", -1, 2)
    scale_src = shader(r, "scale_src", "shaders/common/scale.glsl", -1, 3)
    render_src = shader(r, "render_src", "shaders/fluid2d/render_fire.glsl", -1, 4)
    shader(r, "atlas_decode_src", "shaders/common/atlas_decode.glsl", -1, 5)
    for i, field in enumerate(("vel", "scalars", "pressure")):
        ab = []
        for j, which in enumerate("ab"):
            m = place(r.create(td.moviefileinTOP, f"{field}_{which}"), 0, i * 2 + j)
            m.par.playmode = "locked"  # use Path() exactly; Index me.time.frame would rewrite the frame number
            m.par.file.expr = f"parent().ext.CacheReader.Path({field!r}, {j})"
            ab.append(m)
        x = place(r.create(td.crossTOP, f"{field}_x"), 1, i * 2)
        ab[0].outputConnectors[0].connect(x.inputConnectors[0])
        ab[1].outputConnectors[0].connect(x.inputConnectors[1])
        x.par.cross.expr = "parent().ext.CacheReader.Weight()"
        null(r, f"{field}_out", x, 2, i * 2)
    sc_out = r.op("scalars_out")
    P = "parent().par."
    glsl(r, "sources", scale_src, [sc_out], "rgba32float", [("uGain", (P + "Gain1", P + "Gain2", P + "Gain3", P + "Gain4"))], 3, 2)
    null(r, "sources_out", r.op("sources"), 4, 2)
    obs = select(r, "in_obstacles", "op.simbus.path + '/obstacles'", 2, 7)
    f = "op('/sims/fluid2d').par."
    render = glsl(r, "render", render_src, [sc_out, obs], "rgba16float",
                  [("uLook", (f + "Smokedensity", f + "Fireintensity", f + "Whitehot", f + "Obstacleopacity")),
                   ("uSmokeColor", (f + "Smokecolorr", f + "Smokecolorg", f + "Smokecolorb", f + "Background"))], 3, 4)
    null(r, "render_out", render, 4, 4)
    return r


def build(parent_path: str = "/") -> dict:
    root = td.op(parent_path)
    old = root.op("sims")
    if old is not None:
        old.destroy()
    sims = root.create(td.baseCOMP, "sims")
    sims.par.parentshortcut = "sims"
    p = par(sims, "Sims", "folder", "Root", "")
    p.expr = "project.folder"
    init = place(sims.create(td.executeDAT, "init_path"), 0, 2)
    init.text = ("import sys\n\ndef onStart():\n    path = parent.sims.par.Root.eval() + '/python'\n"
                 "    if path not in sys.path:\n        sys.path.insert(0, path)\n    return\n")
    init.par.start = True
    init.par.active = True
    import sys
    path = sims.par.Root.eval() + "/python"
    if path not in sys.path:
        sys.path.insert(0, path)

    build_clock(sims)
    build_bus(sims)
    build_fluid2d(sims)
    build_demo(sims)
    import build_fluid3d  # noqa: PLC0415
    import importlib  # noqa: PLC0415
    import inspect  # noqa: PLC0415
    importlib.reload(build_fluid3d)
    me = sys.modules.get(__name__) or inspect.getmodule(build)
    build_fluid3d.build_bus3d(sims, me)
    build_fluid3d.build_fluid3d(sims, me)
    build_fluid3d.build_demo3d(sims, me)
    import build_particles  # noqa: PLC0415
    importlib.reload(build_particles)
    build_particles.build_particles(sims, me)
    build_particles.build_demo_particles(sims, me)
    build_baker(sims)
    build_cache_reader(sims)
    for comp in sims.children:
        if comp.isCOMP:
            comp.par.reinitextensions.pulse()
    return {"built": [c.path for c in sims.children]}


def register_demo() -> None:
    bus = td.op.simbus
    bus.Publish("sources", "/sims/demo/emitter_sources")
    bus.Publish("forces", "/sims/demo/emitter_forces")
    bus.Publish("obstacles", "/sims/demo/disc")
    bus3 = td.op.simbus3d
    bus3.Publish("sources", "/sims/demo3d/emitter_sources")
    bus3.Publish("forces", "/sims/demo3d/emitter_forces")
    bus3.Publish("obstacles", "/sims/demo3d/sphere")
    bus3.Publish("obstacles", "/sims/demo3d/box")
    bus3.Publish("obstacles", "/sims/demo3d/capsule")
    bus3.Publish("obstacles", "/sims/demo3d/mesh")
    td.op("/sims/demo3d").RebuildMesh()
