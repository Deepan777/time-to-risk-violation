"""Assembling the deployment state `s_t`, and walking a stream to produce the state table.

`build_state` is a pure function of `(window, handle, ref, hist)`. Purity is the property the whole
leakage argument rests on: if the state at window `t` depended on anything outside those four
arguments, no schema assertion downstream could rescue it.

The ordering in `build_state_table` is the other half of that argument, and it is deliberately
strict:

    1. compute s_t from the history as it stands  (risks 0 .. t-1 are present)
    2. compute the realised risk r_t             (this is a TARGET, never an input)
    3. append r_t to the history

Step 2 can never influence step 1 for the same `t`, and `History.matured_risks` additionally holds
back the last `delta` windows. Reversing steps 1 and 2 would leak the outcome into its own
predictor, and would look like an excellent result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..datasets.stream import Stream
from ..metrics.losses import window_risk
from ..utils.leakage import assert_no_future_columns
from .adaptation_state import adaptation_stats
from .context import context_stats
from .discrepancy import discrepancy_stats
from .feedback import feedback_stats
from .history import History
from .prediction_stats import prediction_stats
from .representation_stats import representation_stats
from .reference import ReferenceStats
from .uncertainty import uncertainty_stats

__all__ = ["ALL_BLOCKS", "BLOCK_PREFIX", "build_state", "build_state_table"]

ALL_BLOCKS: tuple[str, ...] = ("pred", "rep", "unc", "dist", "fb", "ctx", "adapt")

BLOCK_PREFIX: dict[str, str] = {
    "pred": "pred_", "rep": "rep_", "unc": "unc_", "dist": "dist_",
    "fb": "fb_", "ctx": "ctx_", "adapt": "adapt_",
}


def build_state(
    window,
    handle,
    ref: ReferenceStats,
    hist: History,
    *,
    blocks: tuple[str, ...] = ALL_BLOCKS,
    delta: int = 1,
    trailing: int = 5,
    feedback_window: int = 5,
    matured_data: tuple[np.ndarray, np.ndarray] | None = None,
    regime: float | None = None,
) -> dict[str, float]:
    """Compute `s_t` for one window. Pure in `(window, handle, ref, hist)`."""
    unknown = set(blocks) - set(ALL_BLOCKS)
    if unknown:
        raise ValueError(f"unknown state blocks {sorted(unknown)}; expected subset of {ALL_BLOCKS}")

    state: dict[str, float] = {}
    X = window.X

    if "pred" in blocks:
        state.update(prediction_stats(X, handle, ref, hist, trailing))
    if "rep" in blocks:
        state.update(representation_stats(X, handle, ref, hist, trailing))
    if "unc" in blocks:
        state.update(uncertainty_stats(X, handle, ref, hist, trailing))
    if "dist" in blocks:
        state.update(discrepancy_stats(X, handle, ref, hist, trailing))
    if "fb" in blocks:
        state.update(feedback_stats(window.index, handle, ref, hist, delta=delta,
                                    k=feedback_window, matured_data=matured_data))
    if "ctx" in blocks:
        state.update(context_stats(window.index, window.t_start, regime))
    if "adapt" in blocks:
        state.update(adaptation_stats(window.index, hist))

    # stable, sorted feature names: the encoder consumes a matrix, and a column order that varied
    # with dict insertion order would silently permute features between runs.
    return {k: state[k] for k in sorted(state)}


def build_state_table(
    deployment: Stream,
    handle,
    ref: ReferenceStats,
    *,
    window_size: int,
    stride: int,
    blocks: tuple[str, ...] = ALL_BLOCKS,
    delta: int = 1,
    trailing: int = 5,
    feedback_window: int = 5,
    hist: History | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Walk the deployment stream, returning (state table, realised risk series).

    The returned frame contains **only** state features. The risk series is returned separately and
    is never merged into it: `risk_t_plus_*` and friends are targets, and keeping them in a
    different object is a stronger guarantee than keeping them in a differently-named column.
    """
    hist = hist if hist is not None else History()
    rows: list[dict[str, float]] = []
    risks: list[float] = []

    windows = list(deployment.windows(window_size, stride))
    if not windows:
        raise ValueError("deployment segment yields no complete windows")

    for w in windows:
        matured_idx = w.index - delta
        matured_data = None
        if 0 <= matured_idx < len(windows):
            mw = windows[matured_idx]
            matured_data = (mw.X, mw.y)

        # 1. state first, from history as it stands
        state = build_state(w, handle, ref, hist, blocks=blocks, delta=delta,
                            trailing=trailing, feedback_window=feedback_window,
                            matured_data=matured_data)
        rows.append(state)

        # 2. realised risk is a TARGET, computed after the state
        r = window_risk(handle.per_sample_loss(w.X, w.y))
        risks.append(r)

        # 3. only now does it enter the history
        hist.append(risk=r, state=state)

    df = pd.DataFrame(rows)
    assert_no_future_columns(df.columns)
    return df, np.asarray(risks, dtype=float)
