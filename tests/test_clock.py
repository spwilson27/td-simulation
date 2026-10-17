import pytest

import simclock
import simclock
from simclock import BAKE, SimClock, SimClockExt, SimClockExt


def test_realtime_defaults():
    c = SimClock()
    assert c.dt == pytest.approx(1 / 60)
    assert c.frame == 0 and c.frame_complete
    c.advance()
    assert c.frame == 1 and c.time == pytest.approx(1 / 60)


def test_substeps_map_steps_to_frames():
    c = SimClock(fps=30, substeps=4, start_frame=10, mode=BAKE)
    assert c.dt == pytest.approx(1 / 120)
    seen = []
    for _ in range(8):
        seen.append((c.frame, c.substep, c.frame_complete))
        c.advance()
    assert seen[:4] == [(10, 0, False), (10, 1, False), (10, 2, False), (10, 3, True)]
    assert seen[4] == (11, 0, False)
    assert c.time == pytest.approx(8 / 120)


def test_reset_to_resume_frame():
    c = SimClock(fps=30, substeps=2, start_frame=5)
    c.reset(8)
    assert c.step == 6 and c.frame == 8 and c.substep == 0
    assert c.time == pytest.approx(6 / 60)
    with pytest.raises(ValueError):
        c.reset(4)


def test_paused_clock_does_not_advance():
    c = SimClock()
    c.running = False
    c.advance()
    assert c.step == 0


@pytest.mark.parametrize("kw", [dict(fps=0), dict(substeps=0), dict(mode="offline")])
def test_rejects_bad_config(kw):
    with pytest.raises(ValueError):
        SimClock(**kw)


def test_uniforms_and_dict():
    c = SimClock(fps=24, substeps=2, seed=7)
    c.advance()
    u = c.uniforms()
    assert u["uDt"] == pytest.approx(1 / 48) and u["uStep"] == 1.0 and u["uSeed"] == 7.0
    assert c.to_dict() == {"fps": 24.0, "substeps": 2, "start_frame": 0, "mode": "realtime", "seed": 7}


class _Slot:
    def __init__(self, store, name):
        self._store = store
        self._name = name

    def eval(self):
        return self._store[self._name]


class _Pars:
    """Stand-in for a TD par collection: `.Name.eval()` reads, assignment writes."""

    def __init__(self, values):
        object.__setattr__(self, "_values", dict(values))

    def __getattr__(self, name):
        if name not in self._values:
            raise AttributeError(name)
        return _Slot(self._values, name)

    def __setattr__(self, name, value):
        if name not in self._values:
            raise AttributeError(name)
        self._values[name] = value


def _clock_owner(**overrides):
    values = dict(
        Fps=60.0, Substeps=1, Startframe=0, Mode="realtime", Seed=0, Running=True,
        Step=0, Sims="", Listeners="", Frame=0, Substep=0, Time=0.0, Dt=0.0,
    )
    values.update(overrides)
    owner = type("Owner", (), {})()
    owner.par = _Pars(values)
    return owner


def _patch_op(monkeypatch, mapping):
    monkeypatch.setattr(simclock, "op", lambda path: mapping.get(path), raising=False)


def test_reset_tick_runs_before_advance_and_notifies_listeners_after_sims(monkeypatch):
    events = []
    sim = type("Sim", (), {"path": "/sim"})()
    listener = type("Listener", (), {"path": "/listener"})()

    def begin(reset):
        events.append(("begin", reset, ext.Clock.step))

    def step():
        events.append("step")

    def on_step(clock_ext):
        events.append(("listen", clock_ext))

    sim.BeginStep = begin
    sim.Step = step
    listener.OnStep = on_step
    owner = _clock_owner(Substeps=2, Step=99, Sims="/sim", Listeners="/listener")
    _patch_op(monkeypatch, {"/sim": sim, "/listener": listener})
    ext = SimClockExt(owner)
    ext.RequestReset(4)
    ext.Tick()
    assert ext.Clock.step == 8
    assert events == [("begin", True, 8), "step", ("listen", ext)]
    assert owner.par.Step.eval() == 8
    assert owner.par.Frame.eval() == 4
    assert owner.par.Substep.eval() == 0
    assert owner.par.Dt.eval() == pytest.approx(1 / 120)
    assert owner.par.Time.eval() == pytest.approx(8 / 120)
    ext.Tick()
    assert ext.Clock.step == 9
    assert events[3:] == [("begin", False, 9), "step", ("listen", ext)]


def test_paused_clock_does_not_step_sims(monkeypatch):
    sim = type("Sim", (), {"path": "/sim", "calls": 0})()
    sim.BeginStep = lambda reset: sim.__setattr__("calls", sim.calls + 1)
    sim.Step = lambda: sim.__setattr__("calls", sim.calls + 1)
    owner = _clock_owner(Running=False, Step=3, Sims="/sim", Frame=-1)
    _patch_op(monkeypatch, {"/sim": sim})
    ext = SimClockExt(owner)
    ext.Tick()
    assert ext.Clock.step == 3
    assert sim.calls == 0
    assert owner.par.Frame.eval() == -1


def test_add_and_remove_listener(monkeypatch):
    heard = []
    first = type("Listener", (), {"path": "/a"})()
    second = type("Listener", (), {"path": "/b"})()
    first.OnStep = lambda clock_ext: heard.append("/a")
    second.OnStep = lambda clock_ext: heard.append("/b")
    owner = _clock_owner()
    _patch_op(monkeypatch, {"/a": first, "/b": second})
    ext = SimClockExt(owner)
    ext.AddListener(first)
    ext.AddListener(second)
    ext.AddListener(first)
    assert owner.par.Listeners.eval() == "/a /b"
    ext.Tick()
    assert heard == ["/a", "/b"]
    ext.RemoveListener(first)
    assert owner.par.Listeners.eval() == "/b"
    ext.RemoveListener(first)
    assert owner.par.Listeners.eval() == "/b"
    ext.Tick()
    assert heard == ["/a", "/b", "/b"]
    ext.RemoveListener(second)
    assert owner.par.Listeners.eval() == ""
