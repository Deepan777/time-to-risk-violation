"""`s^pred` — prediction statistics. Label-free.

What the model's own output distribution looks like this window, and how far that has moved from
the pre-deployment reference. This is the cheapest block to compute and the one most likely to
carry signal early: a model whose confidence profile is sliding is often a model whose inputs have
started to move, and it costs nothing to observe.
"""
from __future__ import annotations

import numpy as np

from .history import History
from .reference import ReferenceStats

__all__ = ["prediction_stats"]

_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def prediction_stats(X: np.ndarray, handle, ref: ReferenceStats, hist: History,
                     trailing: int = 5) -> dict[str, float]:
    out: dict[str, float] = {}

    if handle.task == "regression":
        preds = np.asarray(handle.predict(X), dtype=float)
        out["pred_mean"] = float(np.mean(preds))
        out["pred_std"] = float(np.std(preds))
        for q in _QUANTILES:
            out[f"pred_q{int(q * 100):02d}"] = float(np.quantile(preds, q))
        out["pred_range"] = float(np.ptp(preds))
    else:
        proba = handle.predict_proba(X)
        proba = np.clip(proba, 1e-12, 1.0)
        msp = np.max(proba, axis=1)
        entropy = -np.sum(proba * np.log(proba), axis=1)

        out["pred_msp_mean"] = float(np.mean(msp))
        out["pred_msp_std"] = float(np.std(msp))
        for q in _QUANTILES:
            out[f"pred_msp_q{int(q * 100):02d}"] = float(np.quantile(msp, q))

        out["pred_entropy_mean"] = float(np.mean(entropy))
        out["pred_entropy_std"] = float(np.std(entropy))

        # top-1 minus top-2 probability: how decisively the model separates its best two options
        if proba.shape[1] >= 2:
            part = np.partition(proba, -2, axis=1)
            out["pred_margin_mean"] = float(np.mean(part[:, -1] - part[:, -2]))
        else:
            out["pred_margin_mean"] = 0.0

        # divergence of the predicted-class histogram from the frozen reference prior
        preds = np.argmax(proba, axis=1)
        n_classes = proba.shape[1]
        hist_now = np.bincount(preds, minlength=n_classes).astype(float)
        hist_now = hist_now / max(hist_now.sum(), 1.0)
        prior = ref.pred_class_prior
        if prior.size != hist_now.size:                      # class never predicted on reference
            prior = np.resize(prior, hist_now.size)
            prior = prior / max(prior.sum(), 1e-12)
        p, q = np.clip(hist_now, 1e-12, 1.0), np.clip(prior, 1e-12, 1.0)
        out["pred_class_kl"] = float(np.sum(p * np.log(p / q)))
        out["pred_class_tv"] = float(0.5 * np.sum(np.abs(p - q)))

        # movement relative to the frozen reference behaviour
        out["pred_conf_shift"] = float(np.mean(msp) - ref.ref_confidence_mean)
        out["pred_entropy_shift"] = float(np.mean(entropy) - ref.ref_entropy_mean)

    # volatility of this block's headline statistic over a short trailing window
    key = "pred_mean" if handle.task == "regression" else "pred_msp_mean"
    recent = [s[key] for s in hist.recent_states(trailing) if key in s]
    out["pred_volatility"] = float(np.std(recent)) if len(recent) >= 2 else 0.0

    return out
