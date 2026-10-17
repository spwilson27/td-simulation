"""Fixed-step simulation clock. Pure Python: wrapped by the SimClock extension in TD, tested outside it.

One simulation step runs per TouchDesigner cook. Output frame f consists of `substeps` steps, so

    dt   = 1 / (fps * substeps)
    step = (frame - start_frame) * substeps + substep

In realtime mode substeps is normally 1 and TD's real-time flag stays on. In bake mode TD's real-time flag is
switched off so every step is computed whatever it costs, and the baker writes the state after the last substep
of each frame.
"""

from __future__ import annotations

from dataclasses import dataclass

REALTIME, BAKE = "realtime", "bake"


@dataclass
class SimClock:
    fps: float = 60.0
    substeps: int = 1
    start_frame: int = 0
    mode: str = REALTIME
    seed: int = 0
    step: int = 0
    running: bool = True

    def __post_init__(self):
        self.configure(self.fps, self.substeps, self.start_frame, self.mode)

    def configure(self, fps: float, substeps: int, start_frame: int = 0, mode: str = REALTIME) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if int(substeps) < 1:
            raise ValueError("substeps must be >= 1")
        if mode not in (REALTIME, BAKE):
            raise ValueError(f"mode must be {REALTIME!r} or {BAKE!r}")
        self.fps, self.substeps, self.start_frame, self.mode = float(fps), int(substeps), int(start_frame), mode

    @property
    def dt(self) -> float:
        return 1.0 / (self.fps * self.substeps)

    @property
    def time(self) -> float:
        """Simulation time in seconds at the start of the current step, measured from start_frame = 0 s."""
        return self.step * self.dt

    @property
    def frame(self) -> int:
        """Output frame the current step belongs to."""
        return self.start_frame + self.step // self.substeps

    @property
    def substep(self) -> int:
        return self.step % self.substeps

    @property
    def frame_complete(self) -> bool:
        """True when the current step is the last substep of its frame (the state to save after it cooks)."""
        return self.substep == self.substeps - 1

    def reset(self, frame: int | None = None) -> None:
        """Rewind to the first step of `frame` (default start_frame). Resuming a bake uses frame = resume point."""
        frame = self.start_frame if frame is None else int(frame)
        if frame < self.start_frame:
            raise ValueError("cannot reset before start_frame")
        self.step = (frame - self.start_frame) * self.substeps

    def advance(self) -> int:
        if self.running:
            self.step += 1
        return self.step

    def uniforms(self) -> dict:
        """Values pushed into every simulation shader each step."""
        return {"uDt": self.dt, "uTime": self.time, "uStep": float(self.step), "uFrame": float(self.frame),
                "uSeed": float(self.seed)}

    def to_dict(self) -> dict:
        return {"fps": self.fps, "substeps": self.substeps, "start_frame": self.start_frame, "mode": self.mode,
                "seed": self.seed}


class SimClockExt:
    """Extension for the clock COMP (global shortcut `op.simclock`).

    Custom parameters: Fps, Substeps, Startframe, Mode, Seed, Running, Sims (space-separated COMP paths stepped in
    order), Listeners (COMP paths whose extension has OnStep(clockExt), e.g. a baker), Resetpulse, and the read-only
    outputs Step, Frame, Substep, Time, Dt that shaders read through expressions.

    Tick() runs from an Execute DAT at frame start. It applies a pending reset or advances one step, publishes the
    clock values, then cooks each sim (BeginStep(reset) + Step()) and notifies listeners. Cooking explicitly at frame
    start, after the parameters change, gives exactly one simulation step per TouchDesigner frame.
    """

    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.clock = SimClock()
        self._pending_reset = None
        self._sync_config()
        self.clock.step = int(ownerComp.par.Step.eval())

    def _sync_config(self) -> None:
        p = self.ownerComp.par
        self.clock.configure(p.Fps.eval(), int(p.Substeps.eval()), int(p.Startframe.eval()), p.Mode.eval())
        self.clock.seed = int(p.Seed.eval())

    def _ops(self, par):
        return [o for o in (op(path) for path in par.eval().split()) if o is not None]  # noqa: F821 (td global)

    @property
    def Clock(self) -> SimClock:
        return self.clock

    def RequestReset(self, frame: int | None = None) -> None:
        """Rewind on the next Tick; sims get BeginStep(reset=True) and read their init state for that step."""
        self._pending_reset = frame if frame is not None else int(self.ownerComp.par.Startframe.eval())

    def AddListener(self, comp) -> None:
        paths = self.ownerComp.par.Listeners.eval().split()
        if comp.path not in paths:
            self.ownerComp.par.Listeners = " ".join(paths + [comp.path])

    def RemoveListener(self, comp) -> None:
        self.ownerComp.par.Listeners = " ".join(p for p in self.ownerComp.par.Listeners.eval().split() if p != comp.path)

    def Tick(self) -> None:
        reset = self._pending_reset is not None
        if not reset and not self.ownerComp.par.Running.eval():
            return
        self._sync_config()
        if reset:
            self.clock.reset(self._pending_reset)
            self._pending_reset = None
        else:
            self.clock.advance()
        self._publish()
        for sim in self._ops(self.ownerComp.par.Sims):
            sim.BeginStep(reset)
            sim.Step()
        for listener in self._ops(self.ownerComp.par.Listeners):
            listener.OnStep(self)

    def _publish(self) -> None:
        p, c = self.ownerComp.par, self.clock
        p.Step, p.Frame, p.Substep, p.Time, p.Dt = c.step, c.frame, c.substep, c.time, c.dt
