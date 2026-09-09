"""Deployment-state correctness: purity, delay masking, schema stability, reference immutability.

These are the tests that make the leakage argument checkable rather than rhetorical.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.monitoring import (
    ALL_BLOCKS, BLOCK_PREFIX, History, build_state, build_state_table, fit_reference,
)
from src.utils.leakage import assert_no_future_columns

WINDOW = 200


# ----------------------------------------------------------------- purity and determinism

def test_build_state_is_pure_in_its_arguments(forecastable_pipeline):
    """`build_state` must be a pure function of (window, handle, ref, hist).

    Called twice on identical inputs it must return identical values. Anything reading global RNG
    state, wall-clock time, or a mutable cache would break here.
    """
    p = forecastable_pipeline
    windows = list(p["dep"].windows(WINDOW, WINDOW))
    hist = History()
    a = build_state(windows[3], p["handle"], p["ref"], hist)
    b = build_state(windows[3], p["handle"], p["ref"], hist)
    assert a == b


def test_state_does_not_depend_on_later_windows(forecastable_pipeline):
    """Truncating the stream after window `t` must not change `s_t`.

    If a state feature could see beyond its own window, this is where it shows up: the same window
    computed inside a longer stream would differ from the same window computed at the end of a
    truncated one.
    """
    p = forecastable_pipeline
    full, _ = build_state_table(p["dep"], p["handle"], p["ref"],
                                window_size=WINDOW, stride=WINDOW, delta=1)
    truncated_stream = p["dep"].head_windows(10, WINDOW)
    trunc, _ = build_state_table(truncated_stream, p["handle"], p["ref"],
                                 window_size=WINDOW, stride=WINDOW, delta=1)
    common = [c for c in trunc.columns if c in full.columns]
    np.testing.assert_allclose(
        trunc[common].to_numpy(dtype=float),
        full[common].head(len(trunc)).to_numpy(dtype=float),
        rtol=1e-9, atol=1e-9,
    )


# ----------------------------------------------------------------- the label delay

@pytest.mark.parametrize("delta", [1, 2, 5])
def test_feedback_block_respects_label_delay(forecastable_pipeline, delta):
    """`s^fb` must be unavailable for the first `delta` windows and available afterwards."""
    p = forecastable_pipeline
    states, _ = build_state_table(p["dep"], p["handle"], p["ref"],
                                  window_size=WINDOW, stride=WINDOW, delta=delta)
    avail = states["fb_available"].to_numpy()
    assert avail[:delta].sum() == 0, "feedback available before any label could have arrived"
    assert avail[delta:].all(), "feedback missing after labels should have matured"

    fb_cols = [c for c in states.columns if c.startswith("fb_")]
    early = states.loc[: delta - 1, fb_cols].to_numpy(dtype=float)
    assert np.all(early == 0.0), "unmatured feedback features must be zero, not imputed"


def test_matured_risks_never_returns_the_current_window():
    """`History.matured_risks(t, delta)` is the only route to the risk series, and with delta=1 it
    must stop at t-1. Returning risks[t] would hand the monitor its own target."""
    hist = History()
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
        hist.append(risk=r)
    np.testing.assert_allclose(hist.matured_risks(3, 1), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(hist.matured_risks(3, 2), [0.1, 0.2])
    assert hist.matured_risks(0, 1).size == 0


# ----------------------------------------------------------------- schema

def test_feature_names_are_sorted_and_stable(forecastable_pipeline):
    p = forecastable_pipeline
    cols = list(p["states"].columns)
    assert cols == sorted(cols), "state columns must be sorted; encoder input order must be stable"
    assert len(cols) == len(set(cols)), "duplicate state column"


def test_every_block_contributes_features(forecastable_pipeline):
    states = forecastable_pipeline["states"]
    for block in ALL_BLOCKS:
        n = sum(1 for c in states.columns if c.startswith(BLOCK_PREFIX[block]))
        assert n > 0, f"state block s^{block} contributed no features"


def test_state_table_contains_no_target_or_metadata_columns(forecastable_pipeline):
    """The schema assertion, run against the real table rather than a synthetic column list."""
    assert_no_future_columns(forecastable_pipeline["states"].columns)


def test_state_table_has_no_nans(forecastable_pipeline):
    states = forecastable_pipeline["states"]
    n_nan = int(states.isna().sum().sum())
    assert n_nan == 0, f"{n_nan} NaN entries; imputation must be explicit and logged, not silent"


def test_block_subset_is_a_strict_subset_of_full_state(forecastable_pipeline):
    """Ablations remove blocks; the surviving columns must be identical to the full run's."""
    p = forecastable_pipeline
    windows = list(p["dep"].windows(WINDOW, WINDOW))
    hist = History()
    full = build_state(windows[2], p["handle"], p["ref"], hist)
    subset = build_state(windows[2], p["handle"], p["ref"], hist, blocks=("pred", "dist"))
    assert set(subset) < set(full)
    for k, v in subset.items():
        assert full[k] == v, f"ablating other blocks changed {k}"


def test_unknown_block_is_rejected(forecastable_pipeline):
    p = forecastable_pipeline
    windows = list(p["dep"].windows(WINDOW, WINDOW))
    with pytest.raises(ValueError, match="unknown state blocks"):
        build_state(windows[0], p["handle"], p["ref"], History(), blocks=("pred", "nonsense"))


# ----------------------------------------------------------------- frozen reference

def test_reference_stats_are_frozen(forecastable_pipeline):
    """Integrity rule 3: reference statistics are fitted once and never refitted."""
    ref = forecastable_pipeline["ref"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.ref_risk_mean = 0.5  # type: ignore[misc]


def test_reference_refuses_a_partial_window(forecastable_pipeline):
    p = forecastable_pipeline
    tiny = p["pre"].X()[:50]
    with pytest.raises(ValueError, match="fewer than one window"):
        fit_reference(tiny, p["pre"].y()[:50], p["handle"], window_size=WINDOW, seed=0)


def test_eps_margin_prevents_a_degenerate_threshold(forecastable_pipeline):
    """D21: a quantile-only eps collapses to zero on a strong predictor. The margin is what keeps
    the estimand non-empty, so it must actually be added."""
    ref = forecastable_pipeline["ref"]
    bare = ref.ref_risk_quantiles["q0.7500"]
    assert ref.eps_from_quantile(0.75, margin=0.05) == pytest.approx(bare + 0.05)
    with pytest.raises(ValueError):
        ref.eps_from_quantile(1.5)
    with pytest.raises(ValueError):
        ref.eps_from_quantile(0.75, margin=-0.1)


def test_estimand_is_non_empty_and_has_events(forecastable_pipeline):
    """The whole project rests on there being origins at which the model is still valid."""
    from src.validity.targets import build_targets, censoring_rate

    p = forecastable_pipeline
    recs = build_targets(p["risks"], p["eps"], H=20)
    assert len(recs) > 0, "no valid origins: the estimand is empty (see D21)"
    assert sum(r.event for r in recs) > 0, "no uncensored events: nothing to learn"
    assert 0.0 <= censoring_rate(recs) < 1.0
