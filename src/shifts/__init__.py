"""Controlled shift generation with ground-truth forecastability tagging."""
from __future__ import annotations

from .builder import build_shifted_stream, compound_forecastability, make_u_stream
from .generators import (apply_mechanism, apply_shift, conditional_shift, covariate_shift,
                         gradual_covariate_shift, make_unforecastable_stream)
from .manifest import (MECHANISMS, OBSERVABLE_IN_PX, SCHEDULES, Forecastability, Mechanism,
                       Schedule, ShiftManifest, ShiftSpec, infer_forecastability)
from .schedules import severity_profile

__all__ = [
    "ShiftSpec", "ShiftManifest", "Forecastability", "Mechanism", "Schedule",
    "MECHANISMS", "SCHEDULES", "OBSERVABLE_IN_PX", "infer_forecastability",
    "severity_profile",
    "covariate_shift", "conditional_shift", "gradual_covariate_shift",
    "make_unforecastable_stream", "apply_mechanism", "apply_shift",
    "build_shifted_stream", "compound_forecastability", "make_u_stream",
]
