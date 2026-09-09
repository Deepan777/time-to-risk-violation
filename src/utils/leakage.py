"""Schema-level leakage guards. These run before training, not after a suspicious result."""
from __future__ import annotations
from typing import Iterable

__all__ = ["FORBIDDEN_PREFIXES", "FORBIDDEN_EXACT", "assert_no_future_columns"]

FORBIDDEN_PREFIXES = ("risk_t_plus_", "target_", "future_")
FORBIDDEN_EXACT = frozenset({
    "true_validity_horizon", "censored", "event", "y_tilde",
    "shift_type", "shift_severity", "shift_start", "shift_end",
    "ground_truth_change_point", "forecastability_type", "risk_threshold",
})


def assert_no_future_columns(feature_columns: Iterable[str]) -> None:
    """Raise if any target or generator-metadata column has reached the feature tensor.

    This is the single most important test in the project: every plausible-but-wrong result in
    temporal ML begins with one of these columns leaking into the inputs.
    """
    cols = list(feature_columns)
    bad = sorted({c for c in cols
                  if c in FORBIDDEN_EXACT or c.startswith(FORBIDDEN_PREFIXES)})
    if bad:
        raise AssertionError(
            "temporal leakage: target/metadata columns present in the feature set: " + ", ".join(bad)
        )
