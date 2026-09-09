"""`s^rep` — representation statistics. Label-free.

Where this window's inputs land inside the model's own learned geometry. The PCA basis and the
k-NN index are the frozen ones from `ReferenceStats`: projecting onto a basis refitted each window
would rotate the coordinate system to follow the data and report that nothing moved.

For tree ensembles the "representation" is the projected leaf-index vector described in
`predictors/base.PredictorHandle.embed` — a legitimate description of where the ensemble places a
point in its own partition, but not the same object as a neural penultimate layer. The manuscript
says so rather than blurring the two.
"""
from __future__ import annotations

import numpy as np

from .history import History
from .reference import ReferenceStats

__all__ = ["representation_stats"]


def representation_stats(X: np.ndarray, handle, ref: ReferenceStats, hist: History,
                         trailing: int = 5) -> dict[str, float]:
    Z = np.asarray(handle.embed(X), dtype=float)
    out: dict[str, float] = {}

    # -- centroid movement, in units of the frozen reference scale
    centroid = Z.mean(axis=0)
    out["rep_centroid_shift"] = float(
        np.linalg.norm((centroid - ref.embed_mean) / ref.embed_std) / np.sqrt(Z.shape[1])
    )
    out["rep_norm_mean"] = float(np.mean(np.linalg.norm(Z, axis=1)))
    out["rep_norm_std"] = float(np.std(np.linalg.norm(Z, axis=1)))

    # -- second moments: spread and degeneracy of the window's representation cloud
    if Z.shape[0] > 1:
        cov = np.cov(Z, rowvar=False)
        cov = np.atleast_2d(cov)
        out["rep_cov_trace"] = float(np.trace(cov))
        # log-det via eigenvalues, floored: a degenerate window is informative, not an error
        eig = np.linalg.eigvalsh(cov + 1e-9 * np.eye(cov.shape[0]))
        out["rep_cov_logdet"] = float(np.sum(np.log(np.clip(eig, 1e-12, None))))
        out["rep_cov_condition"] = float(
            np.clip(eig.max(), 1e-12, None) / np.clip(eig.min(), 1e-12, None)
        )
    else:
        out["rep_cov_trace"] = 0.0
        out["rep_cov_logdet"] = 0.0
        out["rep_cov_condition"] = 1.0

    # -- coordinates in the FROZEN reference PCA basis
    proj = (Z - ref.pca_mean) @ ref.pca_components.T
    for i in range(proj.shape[1]):
        out[f"rep_pca{i + 1}_mean"] = float(np.mean(proj[:, i]))
    out["rep_pca_energy"] = float(
        np.sum(proj ** 2) / max(float(np.sum((Z - ref.pca_mean) ** 2)), 1e-12)
    )

    # -- local density: how far this window sits from the reference manifold
    dist, _ = ref.knn_index.kneighbors(Z)
    mean_knn = dist.mean(axis=1)
    out["rep_knn_dist_mean"] = float(np.mean(mean_knn))
    out["rep_knn_dist_q90"] = float(np.quantile(mean_knn, 0.90))
    out["rep_knn_dist_max"] = float(np.max(mean_knn))

    recent = [s["rep_centroid_shift"] for s in hist.recent_states(trailing)
              if "rep_centroid_shift" in s]
    out["rep_volatility"] = float(np.std(recent)) if len(recent) >= 2 else 0.0

    return out
