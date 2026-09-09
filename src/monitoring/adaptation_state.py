"""`s^adapt` — adaptation state.

Through Phases I-VI the predictor is frozen, so every feature here is constant and the block
carries no signal. That is expected and is left in rather than special-cased: the ablation in E14
should show this block contributing nothing while the model is frozen, which is a useful check that
the ablation machinery reports what it ought to.

The block earns its place in Phase VII, where interventions actually occur and the history of past
updates genuinely predicts future validity — a model retrained three windows ago behaves
differently from one that has run untouched for eighty.
"""
from __future__ import annotations

import numpy as np

from .history import History

__all__ = ["adaptation_stats"]


def adaptation_stats(t: int, hist: History) -> dict[str, float]:
    out: dict[str, float] = {
        "adapt_windows_since_update": hist.windows_since_intervention(t),
        "adapt_n_updates": float(len([i for i in hist.interventions if i <= t])),
        "adapt_last_magnitude": 0.0,
        "adapt_last_risk_delta": 0.0,
        "adapt_ever_updated": 0.0,
    }
    past = [j for j, i in enumerate(hist.interventions) if i <= t]
    if past:
        last = past[-1]
        out["adapt_ever_updated"] = 1.0
        if last < len(hist.intervention_magnitudes):
            out["adapt_last_magnitude"] = float(hist.intervention_magnitudes[last])
        if last < len(hist.intervention_risk_delta):
            out["adapt_last_risk_delta"] = float(hist.intervention_risk_delta[last])
    out["adapt_update_rate"] = float(out["adapt_n_updates"] / max(t, 1))
    return out
