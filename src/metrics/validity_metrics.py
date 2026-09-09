"""Validity-specific metrics. Guards are deliberate: several of these metrics are misleading
if reported alone, so the functions refuse to return without their companion quantity."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

__all__ = ["HorizonResult", "vhmae", "pvlt", "warning_metrics", "conditioned_mask"]


@dataclass(frozen=True)
class HorizonResult:
    vhmae: float
    n_uncensored: int
    censoring_rate: float          # never optional: VHMAE alone flatters the method


def vhmae(pred_h: np.ndarray, true_h: np.ndarray, event: np.ndarray) -> HorizonResult:
    """Validity-Horizon MAE over uncensored origins only, returned with the censoring rate."""
    p, t, e = (np.asarray(x) for x in (pred_h, true_h, event))
    if not (p.shape == t.shape == e.shape):
        raise ValueError("pred_h, true_h and event must have the same shape")
    if p.size == 0:
        raise ValueError("empty input")
    m = e == 1
    if not m.any():
        raise ValueError("no uncensored origins: VHMAE is undefined; report the censoring rate only")
    return HorizonResult(vhmae=float(np.mean(np.abs(p[m] - t[m]))),
                         n_uncensored=int(m.sum()),
                         censoring_rate=float(1.0 - m.mean()))


def pvlt(t_violation: np.ndarray, t_warning: np.ndarray) -> np.ndarray:
    """Predictive Validity Lead Time, over true positives only.

    Entries where no warning was issued (NaN) or no violation occurred (NaN) are dropped:
    averaging a lead time over cases with no warning is meaningless.
    """
    v, w = np.asarray(t_violation, dtype=float), np.asarray(t_warning, dtype=float)
    m = np.isfinite(v) & np.isfinite(w)
    return v[m] - w[m]


def warning_metrics(warned: np.ndarray, will_violate: np.ndarray) -> dict[str, float]:
    """Precision, recall and false alarms per 100 valid windows."""
    a, b = np.asarray(warned, dtype=bool), np.asarray(will_violate, dtype=bool)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("shape mismatch or empty input")
    tp = int(np.sum(a & b)); fp = int(np.sum(a & ~b)); fn = int(np.sum(~a & b))
    return {
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "false_alarms_per_100_valid_windows": 100.0 * fp / a.size,
        "n_origins": float(a.size),
    }


def conditioned_mask(risk_now: np.ndarray, risk_next: np.ndarray, eps: float) -> np.ndarray:
    """The Gate-3 protocol: keep only origins where the model is valid now AND next window.
    On this subset a pure detector cannot score, which is what isolates prognosis from detection."""
    a, b = np.asarray(risk_now, dtype=float), np.asarray(risk_next, dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & (a <= eps) & (b <= eps)
    if not m.any():
        raise ValueError("conditioning mask is empty: no origin is valid now and next window")
    return m
