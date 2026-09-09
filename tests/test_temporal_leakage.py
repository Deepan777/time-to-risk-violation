"""Blocking leakage tests. If any of these fail, no result from this repository may be reported."""
import numpy as np
import pytest
from src.utils.leakage import assert_no_future_columns
from src.validity.targets import build_targets


SAFE = ["pred_entropy_mean", "rep_centroid_shift", "unc_ensemble_var", "dist_mmd",
        "fb_rolling_risk", "ctx_month_sin", "adapt_windows_since_update"]


def test_clean_feature_set_passes():
    assert_no_future_columns(SAFE)


@pytest.mark.parametrize("bad", [
    "risk_t_plus_5", "true_validity_horizon", "censored", "event", "y_tilde",
    "shift_type", "shift_severity", "ground_truth_change_point", "forecastability_type",
    "risk_threshold", "future_risk_mean", "target_violation",
])
def test_each_forbidden_column_is_caught(bad):
    with pytest.raises(AssertionError):
        assert_no_future_columns(SAFE + [bad])


def test_targets_never_read_the_origin_window_itself_as_an_outcome():
    """R_t defines eligibility, never the outcome: the scan starts at t+1."""
    risk = np.array([0.10, 0.10, 0.99])
    r0 = build_targets(risk, 0.3, H=2)[0]
    assert r0.y_tilde == 2 and r0.event == 1


def test_targets_depend_only_on_the_future_of_the_origin():
    """Changing the past of an origin must not change its target."""
    a = np.array([0.05, 0.05, 0.10, 0.40])
    b = np.array([0.29, 0.28, 0.10, 0.40])       # different past, same future from index 2
    ta = [r for r in build_targets(a, 0.3, H=3) if r.origin == 2][0]
    tb = [r for r in build_targets(b, 0.3, H=3) if r.origin == 2][0]
    assert (ta.y_tilde, ta.event) == (tb.y_tilde, tb.event)


def test_no_random_split_helper_exists():
    """A random split of a temporal stream is never scientifically justified here; make sure no
    convenience helper sneaks one in."""
    import src.validity.targets as m
    assert not any("random_split" in n for n in dir(m))
