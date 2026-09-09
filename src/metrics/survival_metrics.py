"""Survival metrics on the discrete horizon grid 1..H.

`proposed_methodology.md` section 8 names four: time-dependent concordance (Antolini), IPCW
integrated Brier score, D-calibration, and per-horizon calibration curves. All four treat censoring
explicitly — a censored origin is *not* a non-event, and scoring it as one would make every method
look better on exactly the streams where events are rare.

These are standard estimators, used here as tools. No novelty is claimed for any of them.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "censoring_km", "ipcw_integrated_brier", "brier_at_horizon",
    "antolini_concordance", "d_calibration", "horizon_calibration_curve",
]

_EPS = 1e-8


def censoring_km(y_tilde: np.ndarray, event: np.ndarray, H: int) -> np.ndarray:
    """Kaplan-Meier estimate of the *censoring* survival `G(h) = P(C > h)`, on 1..H.

    This is the reverse-KM: censorings are the events. It supplies the inverse-probability weights
    that keep the Brier score unbiased under right censoring.
    """
    y = np.asarray(y_tilde, dtype=int)
    d = np.asarray(event, dtype=int)
    G, g = np.empty(H, dtype=float), 1.0
    for h in range(1, H + 1):
        at_risk = int(np.sum(y >= h))
        censored = int(np.sum((y == h) & (d == 0)))
        if at_risk > 0:
            g *= (1.0 - censored / at_risk)
        G[h - 1] = max(g, _EPS)
    return G


def brier_at_horizon(surv: np.ndarray, y_tilde: np.ndarray, event: np.ndarray,
                     h: int, G: np.ndarray | None = None) -> float:
    """IPCW Brier score at a single horizon `h` (1-indexed).

    Three cases per origin: an observed violation at or before `h` (target 1), survival past `h`
    (target 0), and censoring before `h` — which contributes nothing, because its outcome at `h`
    was never observed.
    """
    y = np.asarray(y_tilde, dtype=int)
    d = np.asarray(event, dtype=int)
    H = surv.shape[1]
    if not 1 <= h <= H:
        raise ValueError(f"horizon {h} outside 1..{H}")
    G = censoring_km(y, d, H) if G is None else G

    s_h = np.clip(surv[:, h - 1], 0.0, 1.0)
    total, n = 0.0, 0
    for i in range(len(y)):
        if y[i] <= h and d[i] == 1:
            w = 1.0 / max(G[max(y[i] - 2, 0)], _EPS)
            total += w * (s_h[i] - 0.0) ** 2
            n += 1
        elif y[i] > h:
            w = 1.0 / max(G[h - 1], _EPS)
            total += w * (s_h[i] - 1.0) ** 2
            n += 1
        # censored at or before h: outcome unobserved, contributes nothing
    return float(total / n) if n else float("nan")


def ipcw_integrated_brier(surv: np.ndarray, y_tilde: np.ndarray, event: np.ndarray) -> float:
    """Mean IPCW Brier score over the horizon grid. Lower is better; used for early stopping."""
    surv = np.asarray(surv, dtype=float)
    if surv.ndim != 2:
        raise ValueError("surv must be (N, H)")
    H = surv.shape[1]
    G = censoring_km(y_tilde, event, H)
    vals = [brier_at_horizon(surv, y_tilde, event, h, G) for h in range(1, H + 1)]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        raise ValueError("no horizon yielded a defined Brier score; the sample is entirely censored")
    return float(np.mean(vals))


def antolini_concordance(surv: np.ndarray, y_tilde: np.ndarray, event: np.ndarray) -> float:
    """Time-dependent concordance (Antolini).

    A comparable pair is one where `i` has an observed event strictly earlier than `j`'s observed
    time. It is concordant when the model gives `i` the lower survival probability *at i's own event
    time* — which is what makes this time-dependent rather than a single risk score.
    """
    y = np.asarray(y_tilde, dtype=int)
    d = np.asarray(event, dtype=int)
    S = np.clip(np.asarray(surv, dtype=float), 0.0, 1.0)
    n = len(y)

    conc, perm = 0.0, 0
    for i in range(n):
        if d[i] != 1:
            continue
        h = y[i] - 1
        others = np.flatnonzero(y > y[i])
        if others.size == 0:
            continue
        si = S[i, h]
        sj = S[others, h]
        conc += float(np.sum(si < sj) + 0.5 * np.sum(si == sj))
        perm += int(others.size)

    if perm == 0:
        raise ValueError("no comparable pairs: concordance is undefined on this sample")
    return float(conc / perm)


def d_calibration(surv: np.ndarray, y_tilde: np.ndarray, event: np.ndarray,
                  n_bins: int = 10) -> dict[str, float]:
    """D-calibration: under a well-calibrated model, `S_i(T_i)` is Uniform(0, 1).

    Uncensored origins contribute their probability integral transform directly. A censored origin
    is known only to have survived past `S_i(C_i)`, so its mass is spread uniformly over the bins
    below that value rather than discarded — discarding it would bias the test toward whichever
    method predicts early failure.
    """
    y = np.asarray(y_tilde, dtype=int)
    d = np.asarray(event, dtype=int)
    S = np.clip(np.asarray(surv, dtype=float), _EPS, 1.0)
    counts = np.zeros(n_bins, dtype=float)

    for i in range(len(y)):
        s_at = S[i, y[i] - 1]
        if d[i] == 1:
            b = min(int(s_at * n_bins), n_bins - 1)
            counts[b] += 1.0
        else:
            # spread the censored origin's mass over [0, s_at]
            if s_at <= _EPS:
                counts[0] += 1.0
                continue
            edges = np.linspace(0.0, 1.0, n_bins + 1)
            overlap = np.clip(np.minimum(edges[1:], s_at) - edges[:-1], 0.0, None)
            counts += overlap / s_at

    total = counts.sum()
    if total <= 0:
        raise ValueError("no origins contributed to D-calibration")
    expected = total / n_bins
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    from scipy.stats import chi2 as chi2_dist
    p = float(chi2_dist.sf(chi2, df=n_bins - 1))
    return {"d_cal_chi2": chi2, "d_cal_p": p, "d_cal_n_bins": float(n_bins),
            "d_cal_max_dev": float(np.max(np.abs(counts - expected)) / expected)}


def horizon_calibration_curve(surv: np.ndarray, y_tilde: np.ndarray, event: np.ndarray,
                              h: int, n_bins: int = 10) -> dict[str, np.ndarray]:
    """Predicted vs observed violation-within-`h`, binned. Feeds the per-horizon calibration plot.

    Origins censored before `h` are excluded: their outcome at `h` is genuinely unknown.
    """
    y = np.asarray(y_tilde, dtype=int)
    d = np.asarray(event, dtype=int)
    p = 1.0 - np.clip(surv[:, h - 1], 0.0, 1.0)

    keep = ((y <= h) & (d == 1)) | (y > h)
    if not keep.any():
        raise ValueError(f"no origin has an observed outcome at horizon {h}")
    obs = ((y <= h) & (d == 1))[keep].astype(float)
    pk = p[keep]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    pred_m, obs_m, cnt = [], [], []
    for b in range(n_bins):
        m = (pk >= edges[b]) & (pk < edges[b + 1] if b < n_bins - 1 else pk <= 1.0)
        if m.any():
            pred_m.append(float(pk[m].mean()))
            obs_m.append(float(obs[m].mean()))
            cnt.append(int(m.sum()))
    pred_m, obs_m, cnt = np.array(pred_m), np.array(obs_m), np.array(cnt)
    ece = float(np.sum(cnt / cnt.sum() * np.abs(pred_m - obs_m))) if cnt.size else float("nan")
    return {"predicted": pred_m, "observed": obs_m, "count": cnt, "ece": np.array([ece])}
