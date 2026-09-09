"""`s^unc` — uncertainty. Label-free.

Uncertainty is monitored separately from the raw prediction statistics because the two can move in
opposite directions, and the dissociation is informative: a model that becomes *more* confident
while its epistemic uncertainty rises is extrapolating, which is a characteristic precursor of
degradation under covariate shift.

Ensemble disagreement is obtained from truncated sub-ensembles rather than from a separately
trained deep ensemble. Training five independent predictors per stream per seed would multiply the
base-predictor cost by five for a single feature block, and the truncation gives a genuine measure
of how much the ensemble's opinion still moves as members are added. It is *not* the same object as
a deep ensemble's disagreement, and it is labelled as a truncation proxy wherever it is reported.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .history import History
from .reference import ReferenceStats

__all__ = ["uncertainty_stats"]

_N_MEMBERS = 5


def _sub_ensemble_probas(handle, X: np.ndarray, n_members: int = _N_MEMBERS
                         ) -> np.ndarray | None:
    """(n_members, n, n_classes) probabilities from nested truncations of the ensemble."""
    m = handle.model
    mod = type(m).__module__
    total = int(handle.hyperparameters.get("n_estimators", 0))
    if total < n_members * 2:
        return None

    fracs = np.linspace(1.0 / n_members, 1.0, n_members)
    cuts = [max(1, int(round(f * total))) for f in fracs]
    out = []
    try:
        if mod.startswith("xgboost"):
            for c in cuts:
                out.append(m.predict_proba(X, iteration_range=(0, c)))
        elif mod.startswith("lightgbm"):
            for c in cuts:
                out.append(m.predict_proba(X, num_iteration=c))
        else:
            return None
    except Exception:
        return None
    return np.stack(out, axis=0)


def uncertainty_stats(X: np.ndarray, handle, ref: ReferenceStats, hist: History,
                      trailing: int = 5) -> dict[str, float]:
    out: dict[str, float] = {}

    if handle.task == "regression":
        preds = np.asarray(handle.predict(X), dtype=float)
        out["unc_pred_var"] = float(np.var(preds))
        out["unc_pred_iqr"] = float(np.subtract(*np.percentile(preds, [75, 25])))
        out["unc_energy_mean"] = 0.0
        out["unc_epistemic"] = 0.0
        out["unc_aleatoric"] = 0.0
        out["unc_ensemble_var"] = 0.0
    else:
        proba = np.clip(handle.predict_proba(X), 1e-12, 1.0)
        entropy = -np.sum(proba * np.log(proba), axis=1)
        out["unc_entropy_mean"] = float(np.mean(entropy))
        out["unc_entropy_q90"] = float(np.quantile(entropy, 0.90))
        out["unc_entropy_max"] = float(np.max(entropy))

        # -- energy score, from genuine logits (see PredictorHandle.raw_scores)
        try:
            raw = handle.raw_scores(X)
            raw = raw.reshape(raw.shape[0], -1)
            if raw.shape[1] == 1:                       # binary margin -> two-column logit form
                raw = np.hstack([np.zeros_like(raw), raw])
            energy = -logsumexp(raw, axis=1)
            out["unc_energy_mean"] = float(np.mean(energy))
            out["unc_energy_q10"] = float(np.quantile(energy, 0.10))
        except NotImplementedError:
            # recorded as absent rather than imputed; the assembler turns this into an explicit
            # availability mask instead of a silently-zero feature
            out["unc_energy_mean"] = float("nan")
            out["unc_energy_q10"] = float("nan")

        # -- epistemic / aleatoric decomposition from sub-ensemble disagreement
        members = _sub_ensemble_probas(handle, X)
        if members is not None:
            members = np.clip(members, 1e-12, 1.0)
            mean_p = members.mean(axis=0)
            total_h = -np.sum(mean_p * np.log(mean_p), axis=1)          # H[E p]
            aleatoric = np.mean(
                -np.sum(members * np.log(members), axis=2), axis=0)     # E H[p]
            out["unc_aleatoric"] = float(np.mean(aleatoric))
            out["unc_epistemic"] = float(np.mean(total_h - aleatoric))  # mutual information
            out["unc_ensemble_var"] = float(np.mean(np.var(members[:, :, -1], axis=0)))
        else:
            out["unc_aleatoric"] = float(np.mean(entropy))
            out["unc_epistemic"] = 0.0
            out["unc_ensemble_var"] = 0.0

    recent_key = "unc_entropy_mean" if handle.task != "regression" else "unc_pred_var"
    recent = [s[recent_key] for s in hist.recent_states(trailing) if recent_key in s]
    out["unc_volatility"] = float(np.std(recent)) if len(recent) >= 2 else 0.0

    return out
