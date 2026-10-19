"""`/sims/scenes` preset switcher and the combined composite."""

from __future__ import annotations

import td


def build_scenes(sims, bn):
    comp = bn.place(sims.create(td.baseCOMP, "scenes"), 12, 0)
    bn.par(comp, "Scene", "menu", "Preset", "realtime", menu=["realtime", "showcase"])
    bn.par(comp, "Scene", "pulse", "Applypreset", label="Apply Preset")
    bn.extension(comp, "scenes_ext", "python/scenes.py", "Scenes")
    bn.pulse_callbacks(comp, {"Applypreset": "o.Apply(o.par.Preset.eval())"})
    src = bn.shader(comp, "composite_glsl", "shaders/scenes/composite.glsl", -1, 0)
    for i, (name, expr) in enumerate((
        ("in_volume", "op('/sims/fluid3d/render_out')"),
        ("in_pieces", "op('/sims/destruction/pieces_render')"),
        ("in_liquid", "op('/sims/liquid/render_out')"),
    )):
        bn.select(comp, name, expr, 0, i)
    top = bn.place(comp.create(td.glslTOP, "composite"), 2, 0)
    top.par.glslversion = "glsl460"
    top.par.pixeldat = src
    top.par.format = "rgba8fixed"
    top.par.outputresolution = "custom"
    top.par.resolutionw = 480
    top.par.resolutionh = 270
    top.par.inputfiltertype = "nearest"
    top.par.filtertype = "nearest"
    top.setInputs([comp.op("in_volume"), comp.op("in_pieces"), comp.op("in_liquid")])
    top.par.tops = "/sims/cloth/cloth_render"
    top.par.vec0name = "uGain"
    top.par.vec0valuex = 1
    top.par.vec0valuey = 1
    top.par.vec0valuez = 1
    top.par.vec0valuew = 0.85
    for dock in list(top.docked):
        dock.destroy()
    debris_src = bn.shader(comp, "debris_glsl", "shaders/scenes/debris.glsl", -1, 1)
    bn.select(comp, "in_debris", "op('/sims/particles/beauty')", 0, 3)
    over = bn.place(comp.create(td.glslTOP, "with_debris"), 3, 0)
    over.par.glslversion = "glsl460"
    over.par.pixeldat = debris_src
    over.par.format = "rgba8fixed"
    over.par.outputresolution = "custom"
    over.par.resolutionw = 480
    over.par.resolutionh = 270
    over.par.inputfiltertype = "nearest"
    over.par.filtertype = "nearest"
    over.setInputs([top, comp.op("in_debris")])
    for dock in list(over.docked):
        dock.destroy()
    bn.null(comp, "picture_out", over, 4, 0)
    return comp
