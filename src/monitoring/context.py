"""`s^ctx` — context. Calendar structure and deployment age.

One omission here is deliberate and worth stating, because it is a leak that looks like a feature.

A normalised stream position `t / T` requires `T`, the total stream length — which is **not known
at window `t`**. Worse, on generated streams whose change point is drawn from a bounded range, `t/T`
hands the model the prior over that change point directly, and it would appear to "forecast" shifts
by learning where in the stream they usually happen. That would corrupt the negative control while
looking entirely innocuous.

So position enters only as **absolute deployment age**, which a real monitor genuinely knows, and
which carries no information about the remaining length of the stream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["context_stats"]


def _cyclical(value: float, period: float, name: str) -> dict[str, float]:
    ang = 2.0 * np.pi * (float(value) % period) / period
    return {f"ctx_{name}_sin": float(np.sin(ang)), f"ctx_{name}_cos": float(np.cos(ang))}


def context_stats(t: int, t_start, regime: float | None = None) -> dict[str, float]:
    """Context features for window `t`, whose timestamp starts at `t_start`."""
    out: dict[str, float] = {}

    # deployment age: known at time t, says nothing about how much stream remains
    out["ctx_deployment_age"] = float(t)
    out["ctx_deployment_age_log"] = float(np.log1p(t))

    ts = pd.Timestamp(t_start) if not isinstance(t_start, (int, np.integer)) else None
    if ts is not None and not pd.isna(ts):
        out.update(_cyclical(ts.hour, 24.0, "hour"))
        out.update(_cyclical(ts.dayofweek, 7.0, "dow"))
        out.update(_cyclical(ts.month, 12.0, "month"))
        out["ctx_is_weekend"] = float(ts.dayofweek >= 5)
    else:
        # integer-clocked stream: no calendar exists. Zeros here are honest (the information is
        # genuinely absent) and the columns are kept so the schema is stable across datasets.
        for name in ("hour", "dow", "month"):
            out[f"ctx_{name}_sin"] = 0.0
            out[f"ctx_{name}_cos"] = 0.0
        out["ctx_is_weekend"] = 0.0

    out["ctx_regime"] = float(regime) if regime is not None else 0.0
    return out
