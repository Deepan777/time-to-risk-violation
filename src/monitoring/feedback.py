"""`s^fb` — delayed feedback. The one label-dependent block, and the one most exposed to leakage.

Labels arrive with delay `delta`. A monitor standing at window `t` may use realised risk only up to
window `t - delta`. Handing it `risks[t]` would give it the very quantity the threshold event is
defined on — the most direct leakage this project can suffer — so the risk series is reachable only
through `History.matured_risks`, which enforces the cut.

When nothing has matured yet, every feature is zero and `fb_available` is 0. That is an explicit
availability mask, not silent imputation: the model can learn that the block is uninformative early
in a deployment, which is true, rather than being fed a fabricated value it cannot distinguish from
a real one.
"""
from __future__ import annotations

import numpy as np

from .history import History
from .reference import ReferenceStats

__all__ = ["feedback_stats", "FEEDBACK_KEYS"]

FEEDBACK_KEYS = (
    "fb_available", "fb_risk_last", "fb_risk_mean", "fb_risk_std", "fb_risk_min",
    "fb_risk_max", "fb_risk_slope", "fb_risk_delta", "fb_risk_vs_reference",
    "fb_sign_changes", "fb_increasing_run", "fb_ece", "fb_n_matured",
)


def _theil_sen_slope(y: np.ndarray) -> float:
    """Median of pairwise slopes: robust to the occasional wild window, which a least-squares fit
    would let dominate the trend."""
    n = y.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n)]
    return float(np.median(slopes)) if slopes else 0.0


def _ece(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of the confidence in the predicted class."""
    conf = np.max(proba, axis=1)
    correct = (np.argmax(proba, axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def feedback_stats(
    t: int,
    handle,
    ref: ReferenceStats,
    hist: History,
    *,
    delta: int,
    k: int = 5,
    matured_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """Feedback features for window `t`, using only risks matured by then.

    `matured_data` is the (X, y) of window `t - delta`, supplied by the assembler when it exists.
    It is used only for the calibration-error feature; every other feature comes from the matured
    risk series.
    """
    out = {key: 0.0 for key in FEEDBACK_KEYS}

    matured = hist.matured_risks(t, delta)
    matured = matured[np.isfinite(matured)]
    if matured.size == 0:
        return out

    recent = matured[-k:]
    out["fb_available"] = 1.0
    out["fb_n_matured"] = float(matured.size)
    out["fb_risk_last"] = float(matured[-1])
    out["fb_risk_mean"] = float(np.mean(recent))
    out["fb_risk_std"] = float(np.std(recent))
    out["fb_risk_min"] = float(np.min(recent))
    out["fb_risk_max"] = float(np.max(recent))
    out["fb_risk_slope"] = _theil_sen_slope(recent)
    out["fb_risk_delta"] = float(matured[-1] - matured[-2]) if matured.size >= 2 else 0.0
    out["fb_risk_vs_reference"] = float(matured[-1] - ref.ref_risk_mean)

    if recent.size >= 3:
        d = np.diff(recent)
        signs = np.sign(d)
        nz = signs[signs != 0]
        out["fb_sign_changes"] = float(np.sum(nz[1:] != nz[:-1])) if nz.size >= 2 else 0.0
        run = 0
        for v in d[::-1]:
            if v > 0:
                run += 1
            else:
                break
        out["fb_increasing_run"] = float(run)

    if matured_data is not None and handle.task != "regression":
        Xm, ym = matured_data
        try:
            out["fb_ece"] = _ece(handle.predict_proba(Xm), np.asarray(ym, dtype=int))
        except Exception:
            out["fb_ece"] = 0.0

    return out
