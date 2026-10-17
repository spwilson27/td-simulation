"""Field bus: the shared contract every simulation reads from and writes to.

The bus COMP (global shortcut `op.simbus`) owns the grid (resolution + world bounds) and these TOPs:

    vel, scalars, pressure          fields of the active fluid (live simulation or cache playback)
    sources, forces, obstacles      merged contributions of every registered publisher

Publishers are any TOPs at bus resolution following the encodings in shaders/common/bus.glsl. They are summed
(sources, forces) or unioned by nearest distance (obstacles) through the GLSL TOP `tops` parameter, so registering
one never rewires the network. The registry is the `publishers` table DAT (kind, path).

The pure functions below hold the logic and are tested outside TouchDesigner; FieldBus is the thin TD extension.
"""

from __future__ import annotations

FIELDS = {
    "vel": "rg32float",
    "scalars": "rgba16float",
    "pressure": "mono32float",
    "sources": "rgba32float",
    "forces": "rgba32float",
    "obstacles": "rgba32float",
}
STATE_FIELDS = ("vel", "scalars", "pressure")
PUBLISH_KINDS = ("sources", "forces", "obstacles")


def check_kind(kind: str) -> str:
    if kind not in PUBLISH_KINDS:
        raise ValueError(f"publisher kind must be one of {PUBLISH_KINDS}, not {kind!r}")
    return kind


def tops_param(rows, kind: str) -> str:
    """Space-separated publisher paths of one kind, in registration order, without duplicates."""
    check_kind(kind)
    seen, out = set(), []
    for k, path in rows:
        if k == kind and path and path not in seen:
            seen.add(path)
            out.append(path)
    return " ".join(out)


def cell_size(bounds, res) -> tuple[float, float]:
    x0, y0, x1, y1 = bounds
    return (x1 - x0) / res[0], (y1 - y0) / res[1]


def matched_height(bounds, width: int) -> int:
    """Grid height giving square cells for the bounds at this width."""
    x0, y0, x1, y1 = bounds
    return max(1, round(width * (y1 - y0) / (x1 - x0)))


def world_to_uv(p, bounds):
    x0, y0, x1, y1 = bounds
    return (p[0] - x0) / (x1 - x0), (p[1] - y0) / (y1 - y0)


def uv_to_world(uv, bounds):
    x0, y0, x1, y1 = bounds
    return x0 + uv[0] * (x1 - x0), y0 + uv[1] * (y1 - y0)


class FieldBus:
    """Extension for the bus COMP. Custom parameters: Resw, Resh, Boundsx0, Boundsy0, Boundsx1, Boundsy1,
    Fieldsource (COMP providing vel/scalars/pressure: a live sim or a CacheReader)."""

    MERGE_TOPS = {"sources": "sources_merge", "forces": "forces_merge", "obstacles": "obstacles_merge"}

    def __init__(self, ownerComp):
        self.ownerComp = ownerComp

    @property
    def Bounds(self):
        p = self.ownerComp.par
        return (p.Boundsx0.eval(), p.Boundsy0.eval(), p.Boundsx1.eval(), p.Boundsy1.eval())

    @property
    def Res(self):
        return (int(self.ownerComp.par.Resw.eval()), int(self.ownerComp.par.Resh.eval()))

    @property
    def Cell(self):
        return cell_size(self.Bounds, self.Res)

    def SetGrid(self, width: int, bounds=None) -> None:
        p = self.ownerComp.par
        if bounds is not None:
            p.Boundsx0, p.Boundsy0, p.Boundsx1, p.Boundsy1 = bounds
        p.Resw = int(width)
        p.Resh = matched_height(self.Bounds, int(width))

    def Field(self, name: str):
        if name not in FIELDS:
            raise KeyError(name)
        return self.ownerComp.op(name)

    def _table(self):
        return self.ownerComp.op("publishers")

    def _rows(self):
        t = self._table()
        return [(t[r, 0].val, t[r, 1].val) for r in range(1, t.numRows)]

    def Publishers(self, kind: str):
        return tops_param(self._rows(), kind).split()

    def Publish(self, kind: str, top) -> None:
        check_kind(kind)
        path = top if isinstance(top, str) else top.path
        if path not in self.Publishers(kind):
            self._table().appendRow([kind, path])
        self._refresh()

    def Unpublish(self, top, kind: str | None = None) -> None:
        path = top if isinstance(top, str) else top.path
        t = self._table()
        for r in reversed(range(1, t.numRows)):
            if t[r, 1].val == path and (kind is None or t[r, 0].val == kind):
                t.deleteRow(r)
        self._refresh()

    def _refresh(self) -> None:
        rows = self._rows()
        for kind, name in self.MERGE_TOPS.items():
            self.ownerComp.op(name).par.tops = tops_param(rows, kind)

    def BindFields(self, comp) -> None:
        """Route vel/scalars/pressure from `comp` (which must have TOPs vel_out, scalars_out, pressure_out)."""
        self.ownerComp.par.Fieldsource = comp if isinstance(comp, str) else comp.path

    def PublisherSettings(self) -> dict:
        """Per-kind cache identity of every registered publisher: path, GLSL vec uniforms, the owning COMP's
        custom parameters, and a digest of the pixel shader. Clock uniforms are omitted (see cache.py)."""
        import importlib  # noqa: PLC0415

        import cache  # noqa: PLC0415

        importlib.reload(cache)
        out = {}
        for kind in PUBLISH_KINDS:
            out[kind] = [self._publisher_settings(cache, path) for path in self.Publishers(kind)]
        return out

    def _publisher_settings(self, cache, path: str) -> dict:
        top = op(path)  # noqa: F821
        if top is None:
            return {"path": path, "missing": True}
        blocks = []
        try:
            count = int(top.seq.vec.numBlocks)
        except (AttributeError, TypeError, ValueError):
            count = 0
        for i in range(count):
            name_par = getattr(top.par, f"vec{i}name", None)
            if name_par is None:
                break
            components = []
            for axis in "xyzw":
                par = getattr(top.par, f"vec{i}value{axis}")
                components.append((par.expr or "", par.eval()))
            blocks.append((name_par.eval(), components))
        shader = ""
        if hasattr(top.par, "pixeldat"):
            dat = top.par.pixeldat.eval()
            dat_op = dat if getattr(dat, "text", None) is not None else (op(str(dat)) if dat else None)  # noqa: F821
            if dat_op is not None and hasattr(dat_op, "text"):
                shader = dat_op.text or ""
        pars = [(p.name, p.expr or "", p.eval(), str(p.style)) for p in top.parent().customPars]
        import hashlib  # noqa: PLC0415

        return {
            "path": path,
            "uniforms": cache.glsl_uniform_identity(blocks),
            "owner_pars": cache.custom_par_identity(pars),
            "shader": hashlib.sha256(shader.encode("utf-8")).hexdigest()[:16],
        }
