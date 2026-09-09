"""Risk-path accuracy: per-horizon MAE, RMSE, and skill against persistence.

`proposed_methodology.md` section 8 specifies exactly these. Gate 1 is decided on horizon-wise MAE
at `h in {1, 5, 10}`, so the per-horizon breakdown is the primary output and the aggregate is
secondary — a method that wins on average while losing at `h = 10` has not passed.

Every function masks NaN targets. A path that runs past the end of the stream was never observed,
and scoring a forecast against it would reward or punish the model for nothing.
"""
from __future__ import annotations

import numpy as np

__all__ = ["horizon_mae", "horizon_rmse", "skill_score", "risk_path_metrics", "GATE1_HORIZONS"]

#: The horizons Gate 1 is decided on (go_no_go.md Gate 1).
GATE1_HORIZONS: tuple[int, ...] = (1, 5, 10)


def _masked(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    if pred.ndim != 2:
        raise ValueError("expected (N, H) arrays")
    return pred, target


def horizon_mae(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(H,) mean absolute error at each horizon, over origins with an observed value there."""
    pred, target = _masked(pred, target)
    out = np.full(pred.shape[1], np.nan)
    for h in range(pred.shape[1]):
        m = np.isfinite(target[:, h])
        if m.any():
            out[h] = float(np.mean(np.abs(pred[m, h] - target[m, h])))
    return out


def horizon_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred, target = _masked(pred, target)
    out = np.full(pred.shape[1], np.nan)
    for h in range(pred.shape[1]):
        m = np.isfinite(target[:, h])
        if m.any():
            out[h] = float(np.sqrt(np.mean((pred[m, h] - target[m, h]) ** 2)))
    return out


def skill_score(pred: np.ndarray, target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """(H,) skill = 1 - MAE(model) / MAE(reference). Positive means better than the reference.

    Reported against persistence, because "better than assuming tomorrow equals today" is the
    weakest claim worth making and the one a practitioner checks first.
    """
    m_model = horizon_mae(pred, target)
    m_ref = horizon_mae(reference, target)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m_ref > 0, 1.0 - m_model / m_ref, np.nan)


def per_origin_abs_error(pred: np.ndarray, target: np.ndarray, h: int) -> np.ndarray:
    """Absolute errors at horizon `h` (1-indexed), one per origin with an observed value.

    Paired statistical tests need per-origin errors, not the aggregate: the Wilcoxon signed-rank
    test in `evaluation/statistics.py` consumes this.
    """
    pred, target = _masked(pred, target)
    if not 1 <= h <= pred.shape[1]:
        raise ValueError(f"horizon {h} outside 1..{pred.shape[1]}")
    m = np.isfinite(target[:, h - 1])
    return np.abs(pred[m, h - 1] - target[m, h - 1])


def risk_path_metrics(pred: np.ndarray, target: np.ndarray,
                      reference: np.ndarray | None = None) -> dict:
    """Full risk-path summary, with the Gate-1 horizons broken out explicitly."""
    mae, rmse = horizon_mae(pred, target), horizon_rmse(pred, target)
    out: dict = {
        "mae_by_horizon": mae.tolist(),
        "rmse_by_horizon": rmse.tolist(),
        "mae_mean": float(np.nanmean(mae)),
        "rmse_mean": float(np.nanmean(rmse)),
        "n_origins": int(pred.shape[0]),
        "H": int(pred.shape[1]),
    }
    for h in GATE1_HORIZONS:
        if h <= pred.shape[1]:
            out[f"mae_h{h}"] = float(mae[h - 1])
            out[f"rmse_h{h}"] = float(rmse[h - 1])
    if reference is not None:
        skill = skill_score(pred, target, reference)
        out["skill_by_horizon"] = skill.tolist()
        out["skill_mean"] = float(np.nanmean(skill))
        for h in GATE1_HORIZONS:
            if h <= pred.shape[1]:
                out[f"skill_h{h}"] = float(skill[h - 1])
    return out
