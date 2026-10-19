"""GPU FLIP network under /sims/liquid. Grid resolution is the 3D bus."""

from __future__ import annotations

import td

import build_fluid3d as b3

LEVELS = 5
# Fixed CG iterations. The chain is unrolled so one cook runs the whole solve.
# Keep python/liquid.py PCG_STEPS equal to this.
PCG_STEPS = 8
P = "parent().par."
BMIN = ("op.simbus3d.par.Boundsx0", "op.simbus3d.par.Boundsy0", "op.simbus3d.par.Boundsz0", 0)
BMAX = ("op.simbus3d.par.Boundsx1", "op.simbus3d.par.Boundsy1", "op.simbus3d.par.Boundsz1", 0)
GRID = ("op.simbus3d.par.Resw", "op.simbus3d.par.Resh", "op.simbus3d.par.Resd", 0)
CLOCK = ("op.simclock.par.Dt", "op.simclock.par.Time", "op.simclock.par.Step", "op.simclock.par.Frame")


def glsl2d(comp, name, shader, inputs, fmt, uniforms, col, row, npasses=None):
    top = comp.create(td.glslTOP, name)
    top.nodeX, top.nodeY = col * 200, -row * 150
    top.par.glslversion = "glsl460"
    top.par.mode = "compute"
    top.par.computedat = shader
    top.par.format = fmt
    top.par.outputresolution = "custom"
    top.par.resolutionw.expr = P + "Resx"
    top.par.resolutionh.expr = P + "Resy"
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


def build_liquid(sims, bn):
    f = bn.place(sims.create(td.baseCOMP, "liquid"), 12, 0)
    bn.par(f, "Sim", "menu", "Preset", "realtime", menu=["realtime", "tank", "bake"])
    bn.par(f, "Sim", "pulse", "Applypreset", label="Apply Preset")
    bn.par(f, "Sim", "menu", "Mode", "dam", menu=["dam", "tank", "transfer"])
    bn.par(f, "Sim", "int", "Resx", 256, minmax=(64, 2048))
    bn.par(f, "Sim", "int", "Resy", 256, minmax=(64, 2048))
    bn.par(f, "Sim", "int", "Sortpasses", 136, minmax=(1, 400))
    bn.par(f, "Sim", "toggle", "Project", True, label="Pressure Projection")
    bn.par(f, "Sim", "toggle", "Cookrender", False, label="Cook Render")
    bn.par(f, "Sim", "int", "Smooth", 6, minmax=(1, 40))
    bn.par(f, "Sim", "int", "Coarse", 24, minmax=(1, 80))
    bn.par(f, "Sim", "int", "Vcycles", 2, minmax=(1, 2))
    bn.par(f, "Sim", "int", "Polish", 8, minmax=(0, 80), label="Fine Jacobi Polish")
    bn.par(f, "Sim", "int", "Pcgiters", 6, minmax=(1, 12), label="MGPCG Iterations")
    bn.par(f, "Sim", "int", "Pcgsmooth", 1, minmax=(1, 8), label="PCG Smooth")
    bn.par(f, "Sim", "int", "Pcgcoarse", 12, minmax=(4, 40), label="PCG Coarse")
    for name, default in (("Boxx0", 0.02), ("Boxy0", 0.02), ("Boxz0", 0.02),
                          ("Boxx1", 0.55), ("Boxy1", 0.70), ("Boxz1", 0.98)):
        bn.par(f, "Seed", "float", name, default)
    bn.par(f, "Seed", "toggle", "Boxenable", True, label="Custom Seed Box")
    bn.par(f, "Physics", "float", "Spacing", 1.0 / 64.0)
    bn.par(f, "Physics", "float", "Density", 1000.0)
    bn.par(f, "Physics", "float", "Flip", 0.9, minmax=(0.0, 1.0))
    bn.par(f, "Physics", "float", "Threshfrac", 0.2)
    bn.par(f, "Physics", "float", "Gravityx", 0.0)
    bn.par(f, "Physics", "float", "Gravityy", -9.81)
    bn.par(f, "Physics", "float", "Gravityz", 0.0)
    bn.par(f, "Physics", "float", "Maxspeed", 8.0)
    bn.par(f, "Physics", "float", "Transferspeed", 1.0)
    bn.par(f, "Obstacle", "float", "Spherx", 0.72)
    bn.par(f, "Obstacle", "float", "Sphery", 0.28)
    bn.par(f, "Obstacle", "float", "Spherez", 0.50)
    bn.par(f, "Obstacle", "float", "Spherer", 0.0)
    bn.par(f, "Obstacle", "float", "Pathampx", 0.0)
    bn.par(f, "Obstacle", "float", "Pathampz", 0.0)
    bn.par(f, "Look", "int", "Renderw", 640)
    bn.par(f, "Look", "int", "Renderh", 360)
    bn.par(f, "Look", "int", "Raysteps", 32, minmax=(12, 48))
    for name, default in (("Camaz", 0.6), ("Camel", 0.35), ("Camdist", 2.4), ("Camfov", 32.0),
                          ("Targetx", 0.5), ("Targety", 0.35), ("Targetz", 0.5)):
        bn.par(f, "Look", "float", name, default)
    bn.extension(f, "liquid_ext", "python/liquid.py", "Liquid")
    bn.pulse_callbacks(f, {"Applypreset": "o.ApplyPreset(o.par.Preset.eval())"})

    names = (
        "liquid_common", "seed", "keys", "bitonic", "bins", "p2g", "vel_from_mass", "classify", "forces",
        "extrapolate", "divergence", "smooth", "restrict", "restrict_type", "prolong", "project", "g2p",
        "advect", "phi", "sphere", "solids", "render", "preview", "apply", "saxpy", "neg", "dot_mul", "reduce",
        "unpack_pressure", "radix_count", "radix_scan", "radix_scatter",
        "saxpy_tex", "pcg_beta", "pcg_alpha", "pcg_rel",
    )
    src = {}
    for i, name in enumerate(names):
        src[name] = bn.shader(f, name + "_glsl", f"shaders/liquid/{name}.glsl", -1, i)
    bn.shader(f, "bus3d_glsl", "shaders/common/bus3d.glsl", -1, len(names))
    zero_src = bn.shader(f, "zero_glsl", "shaders/common/zero3d.glsl", -1, len(names) + 1)
    empty_src = bn.shader(f, "empty_glsl", "shaders/common/empty3d.glsl", -1, len(names) + 2)

    scene = (P + "Mode.menuIndex", P + "Spacing", P + "Density", P + "Transferspeed")
    part = (P + "Resx", P + "Resy", P + "Resx * " + P + "Resy", 0)
    # 0.2 of one cell of fluid. h is the bus cell, not the particle spacing.
    cell = "((op.simbus3d.par.Boundsx1 - op.simbus3d.par.Boundsx0) / op.simbus3d.par.Resw)"
    thresh = (P + "Threshfrac * " + P + "Density * " + cell + " * " + cell + " * " + cell, 0, 0, 0)
    grav = (P + "Gravityx", P + "Gravityy", P + "Gravityz", 0)
    flip_u = (P + "Flip", P + "Maxspeed", 0, 0)

    dummy = bn.place(f.create(td.constantTOP, "dummy"), 0, 8)
    dummy.par.outputresolution = "custom"
    dummy.par.resolutionw.expr = P + "Resx"
    dummy.par.resolutionh.expr = P + "Resy"
    dummy.par.format = "rgba32float"

    for i, field in enumerate(("pos", "vel")):
        seed = glsl2d(f, "seed_" + field, src["seed"], [dummy], "rgba32float",
                      [("uBMin", BMIN), ("uBMax", BMAX), ("uScene", scene), ("uWrite", (0 if field == "pos" else 1, 0, 0, 0)),
                       ("uBox0", (P + "Boxx0", P + "Boxy0", P + "Boxz0", P + "Boxenable")),
                       ("uBox1", (P + "Boxx1", P + "Boxy1", P + "Boxz1", 0))],
                      0, i)
        load = bn.place(f.create(td.moviefileinTOP, "load_" + field), 1, i)
        load.par.playmode = "locked"
        load.par.index = 0
        choose = bn.switch(f, "init_" + field, [seed, load], 2, i)
        sel = bn.select(f, "src_" + field, choose.path, 3, i)
        sel.par.top.expr = ""
        sel.par.top = choose.path

    keys = glsl2d(f, "keys", src["keys"], [f.op("src_pos")], "rgba32float",
                  [("uBMin", BMIN), ("uBMax", BMAX), ("uGrid", GRID)], 4, 0)
    # Full bitonic sort, cooked only by SortAgrees(). The live step is the radix chain below.
    # Its bypass stays off so the validator's bypass list still sees a live node.
    glsl2d(f, "sort", src["bitonic"], [keys], "rgba32float", [("uUnused", (0, 0, 0, 0))], 5, 0,
           npasses=P + "Sortpasses")
    # Six static digits. The histogram is at most 4096 wide so a 2M-particle key (32768 tiles)
    # still fits; the scan adds each digit's global offset in that same texture.
    tiles = "max(1, int(parent().par.Resx * parent().par.Resy / 64))"
    stride = "min(4096, max(1, int(parent().par.Resx * parent().par.Resy / 64)))"
    hist_h = ("16 * max(1, int((max(1, int(parent().par.Resx * parent().par.Resy / 64))"
              " + min(4096, max(1, int(parent().par.Resx * parent().par.Resy / 64))) - 1)"
              " / min(4096, max(1, int(parent().par.Resx * parent().par.Resy / 64)))))")
    ordered = keys
    for digit, shift in enumerate((0, 4, 8, 12, 16, 20)):
        hist_u = ("uHist", (stride, tiles, 0, 0))
        count = glsl2d(f, f"radix_count_{digit}", src["radix_count"], [ordered], "mono32float",
                       [("uShift", (float(shift), 0, 0, 0)), hist_u], 6, digit)
        count.par.resolutionw.expr = stride
        count.par.resolutionh.expr = hist_h
        scan = glsl2d(f, f"radix_scan_{digit}", src["radix_scan"], [count], "mono32float",
                      [hist_u], 7, digit)
        scan.par.resolutionw.expr = stride
        scan.par.resolutionh.expr = hist_h
        ordered = glsl2d(f, f"radix_scatter_{digit}", src["radix_scatter"], [ordered, scan], "rgba32float",
                         [("uShift", (float(shift), 0, 0, 0)), hist_u], 8, digit)
    bins = b3.glsl3d(bn, f, "bins", src["bins"], [ordered], "rgba32float", [("uPart", part)], col=6, row=0, include_bus=False)
    p2g = b3.glsl3d(bn, f, "p2g", src["p2g"], [ordered, f.op("src_pos"), f.op("src_vel"), bins], "rgba32float",
                    [("uBMin", BMIN), ("uBMax", BMAX), ("uPart", part)], col=7, row=0, include_bus=False)
    vel_p2g = b3.glsl3d(bn, f, "vel_p2g", src["vel_from_mass"], [p2g], "rgba32float",
                        [("uUnused", (0, 0, 0, 0))], col=8, row=0, include_bus=False)

    sphere = b3.glsl3d(bn, f, "sphere", src["sphere"], [], "rgba32float",
                       [("uSphere", (P + "Spherx", P + "Sphery", P + "Spherez", P + "Spherer")),
                        ("uPath", (P + "Pathampx", P + "Pathampz", 0, 0))], col=6, row=2)
    empty = b3.glsl3d(bn, f, "empty_solids", empty_src, [], "rgba32float", col=6, row=3)
    # Retarget this select at a destruction obstacle TOP to make pieces solid without publishing phi back in.
    extra = bn.select(f, "extra_solids", "op('empty_solids')", 6, 4)
    solids = b3.glsl3d(bn, f, "solids", src["solids"], [sphere, extra], "rgba32float",
                       [("uUnused", (0, 0, 0, 0))], col=7, row=2, include_bus=False)
    forces_bus = bn.select(f, "bus_forces", "op.simbus3d.path + '/forces'", 7, 3)

    classify = b3.glsl3d(bn, f, "classify", src["classify"], [p2g, solids], "rgba32float",
                         [("uThresh", thresh)], col=9, row=0)
    forced = b3.glsl3d(bn, f, "forces", src["forces"], [vel_p2g, classify, solids, forces_bus], "rgba32float",
                       [("uGravity", grav)], col=10, row=0)
    extra = b3.glsl3d(bn, f, "extrapolate", src["extrapolate"], [forced, classify, solids], "rgba32float",
                      col=11, row=0, npasses=2)
    div = b3.glsl3d(bn, f, "divergence", src["divergence"], [extra, classify, solids], "mono32float", col=12, row=0)

    types = [classify]
    zeros = []
    for level in range(LEVELS):
        zeros.append(b3.glsl3d(bn, f, f"zero_l{level}", zero_src, [], "mono32float", col=13, row=level, level=level, include_bus=False))
        if level:
            types.append(b3.glsl3d(bn, f, f"type_l{level}", src["restrict_type"], [types[-1]], "rgba32float",
                                   col=14, row=level, level=level))

    def cycle(prefix, pin, col, rhs0, nlevels, smooth="Smooth", coarse="Coarse"):
        pre = []
        rhs = [rhs0]
        for level in range(nlevels):
            pressure_in = pin if level == 0 else zeros[level]
            last = level == nlevels - 1
            passes = P + coarse if last else P + smooth
            omega = 1.0 if last else 2.0 / 3.0
            sm = b3.glsl3d(bn, f, f"{prefix}_pre{level}", src["smooth"], [pressure_in, rhs[level], types[level]],
                           "mono32float", [("uOmega", (omega, 0, 0, 0))], col=col, row=level, npasses=passes, level=level)
            pre.append(sm)
            if not last:
                rhs.append(b3.glsl3d(bn, f, f"{prefix}_rhs{level + 1}", src["restrict"], [sm, rhs[level], types[level]],
                                     "mono32float", col=col + 1, row=level, level=level + 1))
        correction = pre[-1]
        for level in range(nlevels - 2, -1, -1):
            prolonged = b3.glsl3d(
                bn, f, f"{prefix}_pro{level}", src["prolong"], [pre[level], correction, types[level]], "mono32float",
                [("uScale", (1, 0, 0, 0))], col=col + 2, row=level, filtering="linear", level=level)
            correction = b3.glsl3d(
                bn, f, f"{prefix}_post{level}", src["smooth"], [prolonged, rhs[level], types[level]], "mono32float",
                [("uOmega", (2.0 / 3.0, 0, 0, 0))], col=col + 3, row=level, npasses=P + smooth, level=level)
        return correction

    pcg_rhs = bn.select(f, "pcg_rhs", "op('divergence')", 15, 2)
    first = cycle("a", zeros[0], 16, pcg_rhs, LEVELS)
    pcg_pre = bn.select(f, "pcg_pre", first.path, 15, 1)
    pcg_pre.par.top.expr = ""
    pcg_pre.par.top = first.path
    second = cycle("b", first, 21, div, LEVELS)
    # A few fine Jacobi sweeps after the V-cycle carry the floor's divergence up the column.
    # Without them the bulk keeps the gravity impulse and the tank drains onto the floor.
    polish = b3.glsl3d(bn, f, "polish", src["smooth"], [second, div, classify], "mono32float",
                       [("uOmega", (1.0, 0, 0, 0))], col=25, row=2, npasses=P + "Polish")
    # MGPCG working set. Selects point at the latest cache so a saxpy never reads the cache it writes.
    neg_div = b3.glsl3d(bn, f, "neg_div", src["neg"], [div], "mono32float",
                        [("uUnused", (0, 0, 0, 0))], col=15, row=3, include_bus=False)
    for name in ("x", "r", "d", "z", "ap"):
        sel = bn.select(f, "sel_" + name, "op('zero_l0')", 14, 4)
        sel.par.top.expr = ""
        sel.par.top = f.op("zero_l0").path
    apply = b3.glsl3d(bn, f, "apply", src["apply"], [f.op("sel_d"), classify], "mono32float", col=15, row=5)
    for axis in ("a", "b"):
        sel = bn.select(f, "dot_" + axis, "op('zero_l0')", 14, 6)
        sel.par.top.expr = ""
        sel.par.top = f.op("zero_l0").path
    dot_mul = b3.glsl3d(bn, f, "dot_mul", src["dot_mul"], [f.op("dot_a"), f.op("dot_b"), classify], "mono32float",
                        col=15, row=7)
    # Quarter the sum four times. 256³ lands on 1³; smaller grids clamp at 1.
    dot_src = dot_mul
    for i, div in enumerate((4, 16, 64, 256)):
        red = b3.glsl3d(bn, f, f"dot_r{i}", src["reduce"], [dot_src], "mono32float", col=16, row=7 + i, include_bus=True)
        red.par.resolutionw.expr = f"max(1, int(op.simbus3d.par.Resw / {div}))"
        red.par.resolutionh.expr = f"max(1, int(op.simbus3d.par.Resh / {div}))"
        red.par.customdepth.expr = f"max(1, int(op.simbus3d.par.Resd / {div}))"
        dot_src = red
    neg_r = b3.glsl3d(bn, f, "neg_r", src["neg"], [f.op("sel_r")], "mono32float",
                      [("uUnused", (0, 0, 0, 0))], col=15, row=6, include_bus=False)
    coef = ("uCoef", (1, 0, 0, 0))
    saxpy_x = b3.glsl3d(bn, f, "saxpy_x", src["saxpy"], [f.op("sel_x"), f.op("sel_d"), classify], "mono32float",
                        [coef], col=16, row=4)
    saxpy_r = b3.glsl3d(bn, f, "saxpy_r", src["saxpy"], [f.op("sel_r"), f.op("sel_ap"), classify], "mono32float",
                        [coef], col=16, row=5)
    saxpy_d = b3.glsl3d(bn, f, "saxpy_d", src["saxpy"], [f.op("sel_z"), f.op("sel_d"), classify], "mono32float",
                        [coef], col=16, row=6)
    for name, src_top in (("hold_x0", saxpy_x), ("hold_x1", saxpy_x), ("hold_r0", saxpy_r), ("hold_r1", saxpy_r),
                          ("hold_d0", saxpy_d), ("hold_d1", saxpy_d), ("hold_z", first), ("hold_ap", apply),
                          ("hold_neg", neg_div)):
        b3.hold(bn, f, name, src_top, "mono32float", 17, 0)
    # A Cache TOP keeps only the first image written in a frame, so each MGPCG iteration
    # gets its own bank. Twelve is the parameter maximum.
    for k in range(12):
        b3.hold(bn, f, f"z{k}", pcg_pre, "mono32float", 32, k)
        b3.hold(bn, f, f"ap{k}", apply, "mono32float", 33, k)
        b3.hold(bn, f, f"x{k}", saxpy_x, "mono32float", 34, k)
        b3.hold(bn, f, f"r{k}", saxpy_r, "mono32float", 35, k)
        b3.hold(bn, f, f"d{k}", saxpy_d, "mono32float", 36, k)
    # Previous pressure is an inactive cache. A null cannot hold a 3D image. The cache
    # written this step is not the one the chain just read, so there is no cycle.
    def pressure_cache(name, col, row):
        node = b3.hold(bn, f, name, f.op("zero_l0"), "mono32float", col, row)
        return node

    p_a = pressure_cache("p_a", 18, 3)
    p_b = pressure_cache("p_b", 18, 4)
    p_warm = bn.select(f, "p_warm", p_a.path, 19, 3)
    p_warm.par.top.expr = ""
    p_warm.par.top = p_a.path
    zero_dot = b3.glsl3d(bn, f, "zero_dot", zero_src, [], "mono32float", col=19, row=4, include_bus=False)
    for axis, expr in (("resolutionw", "1"), ("resolutionh", "1"), ("customdepth", "1")):
        getattr(zero_dot.par, axis).expr = expr

    def dot_sum(name, a, b, col, row):
        mul = b3.glsl3d(bn, f, name + "_m", src["dot_mul"], [a, b, classify], "mono32float",
                        col=col, row=row, include_bus=False)
        red = mul
        for i, divs in enumerate((4, 16, 64, 256)):
            red = b3.glsl3d(bn, f, f"{name}_r{i}", src["reduce"], [red], "mono32float",
                            col=col, row=row, include_bus=True)
            red.par.resolutionw.expr = f"max(1, int(op.simbus3d.par.Resw / {divs}))"
            red.par.resolutionh.expr = f"max(1, int(op.simbus3d.par.Resh / {divs}))"
            red.par.customdepth.expr = f"max(1, int(op.simbus3d.par.Resd / {divs}))"
        return red

    def unit(name, shader_name, inputs, uniforms, col, row):
        top = b3.glsl3d(bn, f, name, src[shader_name], inputs, "rgba32float", uniforms,
                        col=col, row=row, include_bus=False)
        for axis in ("resolutionw", "resolutionh", "customdepth"):
            getattr(top.par, axis).expr = "1"
        return top

    # r = -div - A p_prev. Zero pressure makes this -div.
    apply_prev = b3.glsl3d(bn, f, "apply_prev", src["apply"], [p_warm, classify], "mono32float",
                           col=38, row=0)
    r_cur = b3.glsl3d(bn, f, "r_init", src["saxpy"], [neg_div, apply_prev, classify], "mono32float",
                      [("uCoef", (1, -1, 0, 0))], col=39, row=0)
    x_cur = p_warm
    d_prev = f.op("zero_l0")
    rho_prev = zero_dot
    for k in range(PCG_STEPS):
        col = 40 + k * 4
        neg_k = b3.glsl3d(bn, f, f"neg_{k}", src["neg"], [r_cur], "mono32float",
                          [("uUnused", (0, 0, 0, 0))], col=col, row=1, include_bus=False)
        z_k = cycle(f"g{k}", zeros[0], col, neg_k, LEVELS, "Pcgsmooth", "Pcgcoarse")
        rho_k = dot_sum(f"rho{k}", r_cur, z_k, col, 6)
        beta_k = unit(f"beta{k}", "pcg_beta", [rho_k, rho_prev],
                      [("uFirst", (1 if k == 0 else 0, 0, 0, 0))], col, 7)
        d_k = b3.glsl3d(bn, f, f"cgd{k}", src["saxpy_tex"], [z_k, d_prev, classify, beta_k], "mono32float",
                        [("uSign", (1, 0, 0, 0))], col=col, row=8)
        ad_k = b3.glsl3d(bn, f, f"ad{k}", src["apply"], [d_k, classify], "mono32float",
                         col=col, row=9)
        den_k = dot_sum(f"den{k}", d_k, ad_k, col, 10)
        alpha_k = unit(f"alpha{k}", "pcg_alpha", [rho_k, den_k], [], col, 11)
        x_cur = b3.glsl3d(bn, f, f"cgx{k}", src["saxpy_tex"], [x_cur, d_k, classify, alpha_k], "mono32float",
                          [("uSign", (1, 0, 0, 0))], col=col, row=12)
        r_cur = b3.glsl3d(bn, f, f"cg_r{k}", src["saxpy_tex"], [r_cur, ad_k, classify, alpha_k], "mono32float",
                          [("uSign", (-1, 0, 0, 0))], col=col, row=13)
        d_prev = d_k
        rho_prev = rho_k
    rr = dot_sum("rr", r_cur, r_cur, 70, 0)
    bb = dot_sum("bb", neg_div, neg_div, 71, 0)
    pcg_rel = unit("pcg_rel", "pcg_rel", [rr, bb], [], 72, 0)
    x_cur.name = "pcg_x"
    pcg_x = x_cur
    # project reads this select. _pcg points it at the snapped pressure after the solve so a
    # later cook cannot run the chain a second time against the pressure it just wrote.
    pcg_p = bn.select(f, "pcg_p", pcg_x.path, 18, 0)
    pcg_p.par.top.expr = ""
    pcg_p.par.top = pcg_x.path
    p_use = bn.select(f, "p_use", "op('zero_l0')", 18, 5)
    p_use.par.top.expr = ""
    p_use.par.top = f.op("zero_l0").path
    pack_in = bn.select(f, "pressure_pack", "op('zero_l0')", 18, 1)
    pack_in.par.top.expr = ""
    pack_in.par.top = f.op("zero_l0").path
    b3.glsl3d(bn, f, "unpack_pressure", src["unpack_pressure"], [pack_in], "mono32float", col=18, row=2)
    # One input of the switch is the projection, the other is the pre-gravity velocity (transfer check).
    choose_vel = bn.place(f.create(td.switchTOP, "vel_for_g2p"), 26, 0)
    project = b3.glsl3d(bn, f, "project", src["project"], [extra, p_use, classify, solids], "rgba32float",
                        [("uLimit", (P + "Maxspeed", 0, 0, 0))], col=26, row=1)
    project.outputConnectors[0].connect(choose_vel.inputConnectors[0])
    vel_p2g.outputConnectors[0].connect(choose_vel.inputConnectors[1])
    choose_vel.par.index.expr = "0 if parent().par.Project else 1"

    g2p = glsl2d(f, "g2p", src["g2p"], [f.op("src_pos"), f.op("src_vel"), choose_vel, vel_p2g], "rgba32float",
                 [("uBMin", BMIN), ("uBMax", BMAX), ("uGrid", GRID), ("uFlip", flip_u)], 27, 0)
    advect = glsl2d(f, "advect", src["advect"], [f.op("src_pos"), g2p, solids], "rgba32float",
                    [("uBMin", BMIN), ("uBMax", BMAX), ("uClock", CLOCK), ("uCollide", (P + "Spacing * 0.5", 0, 0, 0))],
                    27, 1)
    bn.null(f, "vel_state", g2p, 28, 0)
    bn.null(f, "pos_state", advect, 28, 1)
    for i, field in enumerate(("pos", "vel")):
        # The state null is the solved picture. _out is a locked copy so a read cannot integrate again.
        mem = bn.null(f, field + "_out", f.op("vel_state" if field == "vel" else "pos_state"), 29, i)
        mem.lock = True

    phi = b3.glsl3d(bn, f, "phi", src["phi"], [classify], "rgba32float", col=30, row=0)
    b3.hold(bn, f, "phi_out", phi, "rgba32float", 31, 0)
    div_after = b3.glsl3d(bn, f, "div_after", src["divergence"], [project, classify, solids], "mono32float", col=30, row=1)
    b3.hold(bn, f, "div_out", div_after, "mono32float", 31, 1)
    b3.hold(bn, f, "type_out", classify, "rgba32float", 31, 2)

    plate = bn.place(f.create(td.constantTOP, "plate"), 30, 3)
    plate.par.outputresolution = "custom"
    plate.par.resolutionw.expr = P + "Renderw"
    plate.par.resolutionh.expr = P + "Renderh"
    plate.par.format = "rgba8fixed"
    plate.par.colorr, plate.par.colorg, plate.par.colorb, plate.par.alpha = 0.04, 0.05, 0.07, 1
    plate_switch = bn.switch(f, "plate_switch", [plate], 31, 3)
    render = b3.glsl_pixel(
        bn, f, "render", src["render"], [f.op("phi_out"), solids, plate_switch], "rgba8fixed",
        [("uBMin", BMIN), ("uBMax", BMAX),
         ("uCam", (P + "Camaz", P + "Camel", P + "Camdist", P + "Camfov")),
         ("uTarget", (P + "Targetx", P + "Targety", P + "Targetz", 0)),
         ("uLight", (0.4, 0.85, 0.3, 0)),
         ("uRay", (P + "Raysteps", 0, 0, 0)),
         ("uLook", (1.2, 0.55, 0.65, 0)),
         ("uSmooth", (1.0, 0, 0, 0))],
        P + "Renderw", P + "Renderh", 32, 0, filtering="linear")
    bn.null(f, "render_out", render, 33, 0)
    # Live phi, not the cache: the cache keeps only the first image of a TouchDesigner frame,
    # and a group bake ticks more than once per frame.
    preview = b3.glsl_pixel(bn, f, "preview", src["preview"], [phi], "rgba8fixed",
                            [("uUnused", (0, 0, 0, 0))], "256", "256", 32, 1)
    preview_out = bn.null(f, "preview_out", preview, 33, 1)
    preview_out.lock = True
    return f
