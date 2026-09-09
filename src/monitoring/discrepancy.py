"""`s^dist` — distribution discrepancy against the frozen reference window. Label-free.

Five families of two-sample statistic, all measured against the *same* immutable reference: MMD
with an RBF kernel at a frozen bandwidth, per-feature 1-D Wasserstein, energy distance, PSI on
frozen bin edges, and per-feature KS.

This block is the closest thing in `s_t` to a classical drift detector, and including it is
deliberate: experiment E6 exists to show that drift magnitude and realised harm *decouple*, and
that argument is only credible if the drift signal was available to the model and still failed to
account for the outcome on its own.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .history import History
from .reference import ReferenceStats, monitor_view

__all__ = ["discrepancy_stats"]


def _mmd2_rbf(A: np.ndarray, B: np.ndarray, bandwidth: float) -> float:
    """Unbiased squared MMD with an RBF kernel at a frozen bandwidth.

    Both samples are capped before the pairwise computation so that per-window cost stays linear
    in the number of windows rather than quadratic in window size.
    """
    gamma = 1.0 / (2.0 * max(bandwidth, 1e-12) ** 2)

    def k(P, Q):
        d2 = np.sum(P ** 2, 1)[:, None] + np.sum(Q ** 2, 1)[None, :] - 2.0 * P @ Q.T
        return np.exp(-gamma * np.clip(d2, 0.0, None))

    n, m = A.shape[0], B.shape[0]
    if n < 2 or m < 2:
        return 0.0
    Kaa, Kbb, Kab = k(A, A), k(B, B), k(A, B)
    np.fill_diagonal(Kaa, 0.0)
    np.fill_diagonal(Kbb, 0.0)
    return float(Kaa.sum() / (n * (n - 1)) + Kbb.sum() / (m * (m - 1)) - 2.0 * Kab.mean())


def _energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1-D energy distance, averaged over features."""
    vals = []
    for j in range(a.shape[1]):
        try:
            vals.append(float(stats.energy_distance(a[:, j], b[:, j])))
        except Exception:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def _psi(x: np.ndarray, edges: np.ndarray) -> float:
    """Population Stability Index against frozen bin edges."""
    ref_frac = 1.0 / (len(edges) - 1)                 # quantile bins => uniform by construction
    cur, _ = np.histogram(x, bins=edges)
    cur_frac = cur / max(cur.sum(), 1)
    c = np.clip(cur_frac, 1e-6, None)
    r = np.clip(np.full_like(c, ref_frac), 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def discrepancy_stats(X: np.ndarray, handle, ref: ReferenceStats, hist: History,
                      trailing: int = 5, max_points: int = 400) -> dict[str, float]:
    out: dict[str, float] = {}
    Xw = monitor_view(X, handle)     # raw inputs, or the representation for image streams (D25)

    # standardise both sides with the FROZEN reference scale
    A = (Xw - ref.feature_mean) / ref.feature_std
    B = (ref.ref_X - ref.feature_mean) / ref.feature_std

    a = A[:max_points] if A.shape[0] > max_points else A
    b = B[:max_points] if B.shape[0] > max_points else B

    out["dist_mmd2"] = _mmd2_rbf(a, b, ref.mmd_bandwidth)
    out["dist_energy"] = _energy_distance(a, b)

    wass, ks, psi = [], [], []
    for j in range(Xw.shape[1]):
        wass.append(float(stats.wasserstein_distance(A[:, j], B[:, j])))
        ks.append(float(stats.ks_2samp(A[:, j], B[:, j]).statistic))
        psi.append(_psi(Xw[:, j], ref.feature_bin_edges[j]))

    for name, vals in (("wasserstein", wass), ("ks", ks), ("psi", psi)):
        arr = np.asarray(vals, dtype=float)
        out[f"dist_{name}_mean"] = float(np.mean(arr))
        out[f"dist_{name}_max"] = float(np.max(arr))
        out[f"dist_{name}_q90"] = float(np.quantile(arr, 0.90))

    # how far the window's own mean has moved, in frozen reference units
    out["dist_mean_shift"] = float(np.linalg.norm(A.mean(axis=0)) / np.sqrt(A.shape[1]))
    out["dist_scale_ratio"] = float(np.mean(A.std(axis=0)))

    recent = [s["dist_mmd2"] for s in hist.recent_states(trailing) if "dist_mmd2" in s]
    out["dist_volatility"] = float(np.std(recent)) if len(recent) >= 2 else 0.0
    out["dist_trend"] = float(recent[-1] - recent[0]) if len(recent) >= 2 else 0.0

    return out
