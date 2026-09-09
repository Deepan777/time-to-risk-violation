"""Discrete-time survival utilities (standard methodology, used here as a tool)."""
from __future__ import annotations
import numpy as np

__all__ = ["hazard_to_survival", "survival_to_expected_horizon", "median_horizon",
           "kaplan_meier"]


def hazard_to_survival(hazard: np.ndarray) -> np.ndarray:
    """S(h) = prod_{j<=h} (1 - lambda(j)). Monotone non-increasing by construction."""
    h = np.asarray(hazard, dtype=float)
    if np.any((h < 0) | (h > 1)):
        raise ValueError("hazards must lie in [0, 1]")
    return np.cumprod(1.0 - np.clip(h, 0.0, 1.0 - 1e-12), axis=-1)


def survival_to_expected_horizon(surv: np.ndarray) -> float:
    """E[T] over the evaluated horizon grid = sum_h S(h) (truncated at H)."""
    return float(np.sum(np.asarray(surv, dtype=float)))


def median_horizon(surv: np.ndarray) -> int:
    """First h with S(h) <= 0.5; H+1 if survival never drops that far (right-censored estimate)."""
    s = np.asarray(surv, dtype=float)
    idx = np.nonzero(s <= 0.5)[0]
    return int(idx[0] + 1) if idx.size else int(s.size + 1)


def kaplan_meier(y_tilde: np.ndarray, event: np.ndarray, H: int) -> np.ndarray:
    """Marginal survival curve on a discrete grid 1..H. This is the lower-reference baseline
    that PRISM-V must beat (Gate 2) and must NOT beat on unforecastable streams (Gate 0)."""
    y = np.asarray(y_tilde, dtype=int)
    d = np.asarray(event, dtype=int)
    surv, s = np.empty(H), 1.0
    for h in range(1, H + 1):
        at_risk = int(np.sum(y >= h))
        events = int(np.sum((y == h) & (d == 1)))
        if at_risk > 0:
            s *= (1.0 - events / at_risk)
        surv[h - 1] = s
    return surv
