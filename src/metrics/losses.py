"""Bounded per-sample losses and the window risk.

The estimand rests on equation (1) of proposed_methodology.md:

    R_t = E_{(x,y) ~ P_t} [ l( f(x), y ) ]

`l` is required to be **bounded**. Boundedness is not a stylistic preference: an unbounded loss
lets a single outlier window dominate the risk series, and the threshold event `R_t > eps` then
measures the outlier rather than the deployment. For classification the natural bounded loss is
0-1, giving `R_t` = window error rate and `eps` = a maximum tolerable error rate. For regression
(Tier C) the loss is a normalised absolute error clipped to [0, 1], with the normalising scale
frozen on the pre-deployment window like every other reference statistic.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["zero_one_loss", "bounded_absolute_error", "brier_loss", "window_risk", "get_loss"]


def zero_one_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sample 0-1 loss. Bounded in {0, 1}; the window mean is the error rate."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    return (yt != yp).astype(float)


def brier_loss(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Per-sample Brier score for binary problems. Bounded in [0, 1].

    Offered as an alternative to 0-1 because it is sensitive to *calibration* drift, which 0-1
    cannot see: a model whose probabilities decay while its argmax holds is degrading in a way an
    error rate hides.
    """
    yt = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    if yt.shape != p.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {p.shape}")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    return (p - yt) ** 2


def bounded_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> np.ndarray:
    """Normalised absolute error clipped to [0, 1]: min(|y - yhat| / scale, 1).

    `scale` must be a **frozen** reference statistic (e.g. the pre-deployment interquartile range
    or standard deviation of the target). Recomputing it on deployment data would let the loss
    rescale itself as the stream drifts, which would hide exactly the degradation being measured.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be a positive finite number, got {scale!r}")
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    return np.clip(np.abs(yt - yp) / float(scale), 0.0, 1.0)


def window_risk(losses: np.ndarray) -> float:
    """R_t: the mean bounded loss over a window.

    NaN means "not yet matured" and propagates, rather than being dropped: a window whose labels
    have not arrived has an *unknown* risk, not a risk computed from the subset that happens to be
    available. `build_targets` treats NaN as an interrupted horizon for the same reason.
    """
    arr = np.asarray(losses, dtype=float)
    if arr.size == 0:
        raise ValueError("cannot compute a window risk over zero samples")
    if np.any(~np.isfinite(arr)):
        return float("nan")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("losses must be bounded in [0, 1]; check the loss definition")
    return float(np.mean(arr))


_REGISTRY: dict[str, Callable] = {
    "zero_one": zero_one_loss,
    "brier": brier_loss,
    "bounded_absolute_error": bounded_absolute_error,
}


def get_loss(name: str) -> Callable:
    if name not in _REGISTRY:
        raise ValueError(f"unknown loss {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
