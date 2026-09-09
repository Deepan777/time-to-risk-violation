"""Shift specification, manifest, and the forecastability taxonomy.

Two ideas live here and they must not be confused:

* a **ShiftSpec** is what we ask for — mechanism, schedule, severity, window, affected features;
* a **ShiftManifest** is what was actually produced, serialised alongside the stream.

The `forecastability_type` tag is *derived* from the spec rather than accepted on trust
(`infer_forecastability`), and `ShiftSpec.validate` refuses a spec whose declared tag disagrees.
The taxonomy is the theoretical core of the paper (proposed_methodology.md section 3.2); a stream
that is mislabelled F when it is really U would quietly destroy the negative control, which is the
one experiment the framework is required to fail.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

__all__ = [
    "Forecastability", "Mechanism", "Schedule", "MECHANISMS", "SCHEDULES",
    "OBSERVABLE_IN_PX", "ShiftSpec", "ShiftManifest", "infer_forecastability",
]

Forecastability = Literal["F", "PF", "U"]
Mechanism = Literal[
    "covariate", "prior", "conditional", "feature_dropout",
    "sensor_corruption", "noise_escalation", "regime",
]
Schedule = Literal["abrupt", "gradual", "incremental", "recurring", "seasonal"]

MECHANISMS: tuple[str, ...] = (
    "covariate", "prior", "conditional", "feature_dropout",
    "sensor_corruption", "noise_escalation", "regime",
)
SCHEDULES: tuple[str, ...] = ("abrupt", "gradual", "incremental", "recurring", "seasonal")

#: Mechanisms whose action is visible in the input marginal P(X) and therefore *can* leave a
#: precursor for a label-free monitor to see. `conditional` and `prior` are deliberately absent:
#: they move P(Y|X) or P(Y) while leaving the observable inputs alone.
OBSERVABLE_IN_PX: frozenset[str] = frozenset({
    "covariate", "feature_dropout", "sensor_corruption", "noise_escalation", "regime",
})

#: Schedules that spread the change over time, so that a monitor watching P(X) has something to
#: see *before* the risk threshold is crossed.
_PROGRESSIVE: frozenset[str] = frozenset({"gradual", "incremental", "seasonal", "recurring"})


def infer_forecastability(mechanism: str, schedule: str, jump_frac: float = 0.0) -> str:
    """Derive the forecastability class from the generative mechanism.

    The rule follows the table in proposed_methodology.md section 3.2:

    * **U** — the change is in P(Y|X) or P(Y) only, delivered abruptly at an exogenous time. There
      is nothing in the observable history to see, so no estimator can beat the marginal curve.
    * **F** — the change is visible in P(X) and arrives progressively, so a precursor exists.
    * **PF** — everything else: a visible-but-abrupt change (the precursor appears only as the
      change lands), or a progressive change carrying an unpredictable jump component.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}; expected one of {MECHANISMS}")
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
    if not 0.0 <= jump_frac <= 1.0:
        raise ValueError(f"jump_frac must lie in [0, 1], got {jump_frac}")

    observable = mechanism in OBSERVABLE_IN_PX
    progressive = schedule in _PROGRESSIVE

    if not observable:
        # P(X) is untouched. Abrupt -> the canonical unforecastable control.
        # Progressive -> the label rule erodes over time; a *label-fed* monitor can see it once
        # labels mature, but no label-free precursor exists, so at best partial.
        return "U" if schedule == "abrupt" else "PF"
    if progressive and jump_frac == 0.0:
        return "F"
    return "PF"


@dataclass(frozen=True)
class ShiftSpec:
    """What we ask the generator to do."""

    mechanism: Mechanism
    schedule: Schedule
    severity: float                       # peak severity, in [0, 1]
    start: int                            # first affected row
    end: int | None = None                # None -> end of stream
    features: tuple[int, ...] | None = None   # None -> chosen by the builder, recorded in manifest
    forecastability: Forecastability | None = None   # None -> derived; if given, must agree
    period: int | None = None             # required by recurring / seasonal
    jump_frac: float = 0.0                # share of the change delivered as an unpredictable jump
    label: str = ""                       # human-readable name for tables

    def validate(self, n_rows: int) -> None:
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"unknown mechanism {self.mechanism!r}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {self.schedule!r}")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must lie in [0, 1], got {self.severity}")
        if not 0 <= self.start < n_rows:
            raise ValueError(f"start {self.start} outside stream bounds [0, {n_rows})")
        end = n_rows if self.end is None else self.end
        if not self.start < end <= n_rows:
            raise ValueError(f"end {end} must satisfy start < end <= {n_rows}")
        if self.schedule in ("recurring", "seasonal"):
            if not self.period or self.period < 2:
                raise ValueError(f"schedule {self.schedule!r} requires period >= 2")
        derived = infer_forecastability(self.mechanism, self.schedule, self.jump_frac)
        if self.forecastability is not None and self.forecastability != derived:
            raise ValueError(
                f"declared forecastability {self.forecastability!r} disagrees with the class "
                f"implied by mechanism={self.mechanism!r}, schedule={self.schedule!r}, "
                f"jump_frac={self.jump_frac} (= {derived!r}). The tag is derived from the "
                "generative mechanism and cannot be asserted."
            )

    @property
    def derived_forecastability(self) -> str:
        return infer_forecastability(self.mechanism, self.schedule, self.jump_frac)


@dataclass(frozen=True)
class ShiftManifest:
    """What was actually produced. Serialised next to every generated stream.

    Field names match experimental_protocol.md section 2 exactly, and every one of them is
    listed in `src/utils/leakage.py` as forbidden in the feature tensor: this is metadata for
    analysis, never an input.
    """

    shift_type: str
    start_time: int
    end_time: int
    severity: float
    affected_features: tuple[int, ...]
    ground_truth_change_point: int
    forecastability_type: Forecastability
    schedule: str = "abrupt"
    mechanism: str = ""
    period: int | None = None
    jump_frac: float = 0.0
    label: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["affected_features"] = list(self.affected_features)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ShiftManifest":
        d = dict(d)
        d["affected_features"] = tuple(d.get("affected_features", ()))
        d.setdefault("notes", [])
        return cls(**d)
