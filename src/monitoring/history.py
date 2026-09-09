"""Monitoring history: what the monitor is allowed to remember at window `t`.

The single constraint that makes the whole project valid is stated in methodology §2.2: `s_t` may
depend only on information available at the end of window `t`. Labels arriving with delay `delta`
enter the feedback block only from window `t + delta` onward.

`History` enforces that by construction. `matured_risks(t, delta)` is the *only* accessor for the
realised risk series, and it refuses to return anything a monitor at time `t` could not have known.
A block that wants the risk series must go through it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["History"]


@dataclass
class History:
    """Append-only record of everything observed so far."""

    #: realised window risks, indexed by window. NaN where labels have not arrived at all.
    risks: list[float] = field(default_factory=list)
    #: per-window state dicts already computed, for volatility/trend features.
    states: list[dict] = field(default_factory=list)
    #: window indices at which an adaptation action was applied (Phase VII).
    interventions: list[int] = field(default_factory=list)
    #: parameter-space magnitude of each applied action, aligned with `interventions`.
    intervention_magnitudes: list[float] = field(default_factory=list)
    #: realised risk immediately before and after each intervention.
    intervention_risk_delta: list[float] = field(default_factory=list)

    def matured_risks(self, t: int, delta: int) -> np.ndarray:
        """Risks a monitor at window `t` is permitted to see: windows 0 .. t-delta inclusive.

        With `delta = 1`, a monitor standing at window `t` knows the risk of windows up to `t-1`
        but not its own. Returning `risks[t]` here would hand the monitor the very quantity the
        threshold event is defined on, which is the most direct leakage this project can suffer.
        """
        if delta < 0:
            raise ValueError(f"label delay must be >= 0, got {delta}")
        stop = t - delta + 1
        if stop <= 0:
            return np.empty(0, dtype=float)
        return np.asarray(self.risks[:stop], dtype=float)

    def is_matured(self, t: int, delta: int) -> bool:
        """Whether *any* labelled feedback is available to a monitor at window `t`."""
        return self.matured_risks(t, delta).size > 0

    def recent_states(self, k: int) -> list[dict]:
        """The last `k` state dicts, oldest first. Used for volatility features."""
        if k < 1:
            raise ValueError("k must be >= 1")
        return self.states[-k:]

    def windows_since_intervention(self, t: int) -> float:
        """Windows elapsed since the last adaptation action.

        Returns the elapsed count since the start of monitoring when nothing has been applied yet,
        which is the honest answer: "nothing has happened for t windows".
        """
        past = [i for i in self.interventions if i <= t]
        return float(t - past[-1]) if past else float(t)

    def append(self, *, risk: float, state: dict | None = None) -> None:
        self.risks.append(float(risk))
        if state is not None:
            self.states.append(dict(state))

    def record_intervention(self, t: int, magnitude: float = 0.0, risk_delta: float = 0.0) -> None:
        self.interventions.append(int(t))
        self.intervention_magnitudes.append(float(magnitude))
        self.intervention_risk_delta.append(float(risk_delta))
