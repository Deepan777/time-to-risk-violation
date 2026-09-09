"""Time profiles for shift severity.

A shift has two independent axes: *what* changes (the mechanism) and *how it arrives over time*
(the schedule). Separating them means each of the seven mechanisms can be run under each of the
five schedules without writing thirty-five generators, and it makes the forecastability rule
expressible: whether a precursor exists depends on the schedule, whether it is observable at all
depends on the mechanism.

`severity_profile` returns one weight per row in [0, 1]. Mechanisms multiply their peak severity by
it. The profile is a pure function of the schedule parameters, so the same spec always produces the
same time course.
"""
from __future__ import annotations

import numpy as np

__all__ = ["severity_profile"]


def severity_profile(
    schedule: str,
    n_rows: int,
    start: int,
    end: int,
    *,
    period: int | None = None,
    jump_frac: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-row severity weight in [0, 1].

    * ``abrupt`` — 0 before `start`, 1 from `start` onward. No precursor.
    * ``incremental`` — linear ramp 0 -> 1 across [start, end), held at 1 afterwards. The magnitude
      of the change itself moves smoothly, so a monitor sees it coming.
    * ``gradual`` — same ramp, but interpreted by the mechanism as a *mixing probability* between
      the old and new concept. Externally identical in shape; the difference is in how the
      mechanism consumes it (see `generators.apply_mechanism`).
    * ``recurring`` — square wave of the given period across [start, end): the concept alternates.
    * ``seasonal`` — raised sinusoid of the given period, so severity oscillates smoothly.

    `jump_frac` moves part of the total change out of the smooth profile and into a single
    unpredictable step at a random time inside [start, end). This is what turns an F stream into a
    PF one: the schedule remains partly forecastable, but a component of it cannot be anticipated.
    """
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    if not 0 <= start < end <= n_rows:
        raise ValueError(f"require 0 <= start < end <= n_rows, got {start}, {end}, {n_rows}")
    if not 0.0 <= jump_frac <= 1.0:
        raise ValueError(f"jump_frac must lie in [0, 1], got {jump_frac}")

    t = np.arange(n_rows)
    w = np.zeros(n_rows, dtype=float)
    span = max(end - start, 1)

    if schedule == "abrupt":
        w[t >= start] = 1.0
    elif schedule in ("incremental", "gradual"):
        ramp = np.clip((t - start) / span, 0.0, 1.0)
        w = ramp
    elif schedule == "recurring":
        if not period or period < 2:
            raise ValueError("recurring schedule requires period >= 2")
        phase = ((t - start) // (period // 2)) % 2
        w = np.where((t >= start) & (t < end), phase.astype(float), 0.0)
        w[t >= end] = 0.0
    elif schedule == "seasonal":
        if not period or period < 2:
            raise ValueError("seasonal schedule requires period >= 2")
        raised = 0.5 * (1.0 - np.cos(2.0 * np.pi * (t - start) / period))
        w = np.where(t >= start, raised, 0.0)
        w[t >= end] = 0.0
    else:
        raise ValueError(f"unknown schedule {schedule!r}")

    if jump_frac > 0.0:
        if rng is None:
            raise ValueError("jump_frac > 0 requires an rng: the jump time must be drawn")
        # The jump time is drawn independently of everything observable, exactly as tau is in the
        # U construction. That independence is what makes this component unforecastable.
        tau = int(rng.integers(start, end))
        jump = np.zeros(n_rows, dtype=float)
        jump[t >= tau] = 1.0
        w = (1.0 - jump_frac) * w + jump_frac * jump

    return np.clip(w, 0.0, 1.0)
