"""Frozen reference statistics.

Every quantity in here is fitted **once**, on the pre-deployment window, and is immutable
thereafter: normalisation parameters, the PCA basis, the MMD bandwidth, PSI bin edges, and the
k-NN index. This is integrity rule 3, and it is the difference between a monitor that measures
drift and one that silently follows it. A PCA basis refitted each window would rotate to track the
data and report that nothing had changed.

`ReferenceStats` is frozen at the dataclass level and holds only read-only artefacts, so a later
stage cannot casually update it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ReferenceStats", "fit_reference", "MAX_REFERENCE_POINTS",
           "MONITOR_DIM_CAP", "monitor_view"]

#: Reference subsample used for the quadratic-cost statistics (MMD, k-NN). Capping this keeps
#: per-window monitoring cheap; the cap is recorded in the manifest so it is never invisible.
MAX_REFERENCE_POINTS = 2000

#: Above this raw input dimension the distribution-discrepancy block switches to the predictor's
#: representation instead of raw inputs (decision D25). Two reasons, both decisive:
#: per-feature KS / PSI / Wasserstein over 3072 pixel dimensions is ~1.1M two-sample tests per run
#: and computationally hopeless; and nobody monitors image drift pixel-by-pixel - it is universally
#: done in embedding space. The switch is recorded in the manifest and stated in the paper.
MONITOR_DIM_CAP = 256


def monitor_view(X, handle):
    """The matrix the discrepancy block should compare against the reference.

    Raw inputs for tabular streams; the predictor's representation for high-dimensional ones.
    """
    import numpy as _np
    X = _np.asarray(X)
    flat = X.reshape(X.shape[0], -1)
    if flat.shape[1] <= MONITOR_DIM_CAP:
        return flat.astype(float)
    return _np.asarray(handle.embed(flat), dtype=float)


@dataclass(frozen=True)
class ReferenceStats:
    """Immutable snapshot of the pre-deployment window."""

    # -- raw feature space
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_bin_edges: list[np.ndarray]      # PSI bins, per feature
    ref_X: np.ndarray                        # subsampled reference inputs
    mmd_bandwidth: float                     # median heuristic, reference only

    # -- representation space
    ref_embed: np.ndarray                    # subsampled reference embeddings
    embed_mean: np.ndarray
    embed_std: np.ndarray
    pca_components: np.ndarray               # (k, d) basis fitted on reference embeddings
    pca_mean: np.ndarray
    knn_index: Any                           # fitted sklearn NearestNeighbors on ref_embed

    # -- predictive behaviour on the reference window
    pred_class_prior: np.ndarray             # predicted-class histogram
    ref_confidence_mean: float
    ref_entropy_mean: float

    # -- risk calibration of the threshold
    ref_risk_mean: float
    ref_risk_std: float
    ref_risk_quantiles: dict[str, float]     # for the D18 quantile-defined eps
    target_scale: float | None = None

    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ derived
    def eps_from_quantile(self, q: float, margin: float = 0.05) -> float:
        """`eps` as a frozen pre-deployment risk quantile **plus an operational tolerance margin**
        (decision D18, corrected by D21).

        A pure quantile is not usable. A predictor that is near-perfect on its pre-deployment
        window gives `quantile ~ 0`, so `eps ~ 0`, and the deployment is "invalid" from its first
        window: there are then no origins with `R_t <= eps`, and the estimand is empty. That is not
        a hypothetical - it is what the first end-to-end run produced on noiseless SEA.

        The margin also matches how service levels are actually written. An SLA is not "never be
        worse than your best commissioning day"; it is "stay within `margin` of the performance you
        were commissioned at". So:

            eps = quantile_q( pre-deployment window risks ) + margin

        Both terms are computed on pre-deployment data only. `margin` is an operational constant
        declared in the config, never tuned on test data, and it is reported in the dataset table.
        """
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile must lie in (0, 1), got {q}")
        if not 0.0 <= margin < 1.0:
            raise ValueError(f"margin must lie in [0, 1), got {margin}")
        key = f"q{q:.4f}"
        if key not in self.ref_risk_quantiles:
            raise KeyError(
                f"quantile {q} was not precomputed on the reference window "
                f"(available: {sorted(self.ref_risk_quantiles)}). Recompute the reference rather "
                "than estimating it from deployment data."
            )
        return float(min(self.ref_risk_quantiles[key] + margin, 1.0))


def _bin_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges with degenerate columns handled explicitly rather than by accident."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if edges.size < 2:                        # constant feature on the reference window
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _median_bandwidth(X: np.ndarray, rng: np.random.Generator, n: int = 512) -> float:
    """Median pairwise-distance heuristic for the RBF kernel, computed on the reference only."""
    m = min(n, X.shape[0])
    idx = rng.choice(X.shape[0], size=m, replace=False)
    S = X[idx]
    d2 = np.sum((S[:, None, :] - S[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices_from(d2, k=1)
    med = float(np.median(d2[iu])) if iu[0].size else 1.0
    return float(np.sqrt(med / 2.0)) if med > 0 else 1.0


def fit_reference(
    pre_X: np.ndarray,
    pre_y: np.ndarray,
    handle,
    *,
    window_size: int,
    n_pca: int = 8,
    knn_k: int = 10,
    psi_bins: int = 10,
    seed: int = 0,
    risk_quantiles: tuple[float, ...] = (0.5, 0.6, 0.75, 0.8, 0.9, 0.95),
) -> ReferenceStats:
    """Fit every reference statistic on the pre-deployment segment. Called exactly once."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    pre_X = np.asarray(pre_X, dtype=float)
    if pre_X.shape[0] < window_size:
        raise ValueError(
            f"pre-deployment segment has {pre_X.shape[0]} rows, fewer than one window "
            f"({window_size}); reference statistics would be fitted on a partial window."
        )

    # -- subsample for the quadratic-cost statistics
    n_ref = min(MAX_REFERENCE_POINTS, pre_X.shape[0])
    sub = rng.choice(pre_X.shape[0], size=n_ref, replace=False)

    # the discrepancy block works on the monitoring view, which for image streams is the
    # predictor's representation rather than raw pixels (D25)
    mon = monitor_view(pre_X, handle)
    ref_X = mon[sub]

    feature_mean = mon.mean(axis=0)
    feature_std = mon.std(axis=0)
    feature_std = np.where(feature_std > 0, feature_std, 1.0)
    bin_edges = [_bin_edges(mon[:, j], psi_bins) for j in range(mon.shape[1])]
    bandwidth = _median_bandwidth((ref_X - feature_mean) / feature_std, rng)

    # -- representation space
    embed = np.asarray(handle.embed(pre_X), dtype=float)
    ref_embed = embed[sub]
    embed_mean = embed.mean(axis=0)
    embed_std = np.where(embed.std(axis=0) > 0, embed.std(axis=0), 1.0)

    k_pca = int(min(n_pca, ref_embed.shape[1], max(1, ref_embed.shape[0] - 1)))
    pca = PCA(n_components=k_pca, random_state=seed).fit(ref_embed)

    knn = NearestNeighbors(n_neighbors=int(min(knn_k, max(1, n_ref - 1)))).fit(ref_embed)

    # -- predictive behaviour on the reference
    if handle.task == "regression":
        prior = np.array([1.0])
        conf_mean, ent_mean = 0.0, 0.0
    else:
        proba = handle.predict_proba(pre_X)
        n_classes = proba.shape[1]
        preds = np.argmax(proba, axis=1)
        prior = np.bincount(preds, minlength=n_classes).astype(float)
        prior = prior / max(prior.sum(), 1.0)
        conf_mean = float(np.mean(np.max(proba, axis=1)))
        ent_mean = float(np.mean(-np.sum(proba * np.log(np.clip(proba, 1e-12, 1.0)), axis=1)))

    # -- reference risk distribution, over whole pre-deployment windows
    n_win = pre_X.shape[0] // window_size
    risks = []
    for i in range(n_win):
        s, e = i * window_size, (i + 1) * window_size
        risks.append(float(np.mean(handle.per_sample_loss(pre_X[s:e], pre_y[s:e]))))
    risks_arr = np.asarray(risks, dtype=float)
    if risks_arr.size == 0:
        raise ValueError("no complete pre-deployment window: cannot characterise reference risk")

    quantiles = {f"q{q:.4f}": float(np.quantile(risks_arr, q)) for q in risk_quantiles}

    return ReferenceStats(
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_bin_edges=bin_edges,
        ref_X=ref_X,
        mmd_bandwidth=bandwidth,
        ref_embed=ref_embed,
        embed_mean=embed_mean,
        embed_std=embed_std,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        knn_index=knn,
        pred_class_prior=prior,
        ref_confidence_mean=conf_mean,
        ref_entropy_mean=ent_mean,
        ref_risk_mean=float(np.mean(risks_arr)),
        ref_risk_std=float(np.std(risks_arr)),
        ref_risk_quantiles=quantiles,
        target_scale=handle.target_scale,
        meta={
            "n_pre_deployment_rows": int(pre_X.shape[0]),
            "raw_input_dim": int(pre_X.reshape(pre_X.shape[0], -1).shape[1]),
            "monitor_view_dim": int(mon.shape[1]),
            "monitor_view_is_representation": bool(mon.shape[1] != pre_X.reshape(
                pre_X.shape[0], -1).shape[1]),
            "n_reference_points": int(n_ref),
            "n_pre_deployment_windows": int(n_win),
            "window_size": int(window_size),
            "n_pca_components": int(k_pca),
            "knn_k": int(min(knn_k, max(1, n_ref - 1))),
            "psi_bins": int(psi_bins),
            "seed": int(seed),
            "predictor_id": handle.predictor_id,
        },
    )
