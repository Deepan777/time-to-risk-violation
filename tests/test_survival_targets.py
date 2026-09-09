"""Correctness of the censored target construction — the estimand the whole project rests on."""
import numpy as np
import pytest
from src.validity.targets import build_targets, censoring_rate
from src.survival.utils import hazard_to_survival, median_horizon, kaplan_meier


EPS = 0.30


def test_violation_at_known_horizon():
    # valid at t=0..2, first breach at index 3
    risk = np.array([0.10, 0.15, 0.20, 0.40, 0.45])
    recs = {r.origin: r for r in build_targets(risk, EPS, H=5)}
    assert recs[0].y_tilde == 3 and recs[0].event == 1
    assert recs[1].y_tilde == 2 and recs[1].event == 1
    assert recs[2].y_tilde == 1 and recs[2].event == 1
    assert 3 not in recs, "origins already in violation must be excluded"


def test_right_censoring_when_no_violation_in_window():
    risk = np.array([0.10] * 10)
    recs = build_targets(risk, EPS, H=3)
    assert all(r.event == 0 for r in recs)
    assert recs[0].y_tilde == 3                 # censored at the horizon budget
    assert recs[-1].y_tilde == 1                # censored by the end of the stream
    assert censoring_rate(recs) == 1.0


def test_horizon_budget_truncates_before_a_late_violation():
    risk = np.array([0.10, 0.10, 0.10, 0.10, 0.99])
    r0 = build_targets(risk, EPS, H=2)[0]
    assert (r0.y_tilde, r0.event) == (2, 0), "a violation beyond H must be censored, not observed"


def test_unmatured_window_terminates_the_scan_without_claiming_an_event():
    risk = np.array([0.10, 0.12, np.nan, 0.99])
    r0 = build_targets(risk, EPS, H=3)[0]
    assert (r0.y_tilde, r0.event) == (1, 0)


def test_strict_threshold():
    risk = np.array([0.10, EPS, 0.31])
    r0 = build_targets(risk, EPS, H=2)[0]
    assert (r0.y_tilde, r0.event) == (2, 1), "R == eps is still valid; violation is strict"


def test_risk_path_is_nan_padded_and_never_shorter_than_H():
    risk = np.array([0.1, 0.1, 0.1])
    r = build_targets(risk, EPS, H=5)[0]
    assert r.risk_path.shape == (5,) and np.isnan(r.risk_path[2:]).all()


def test_survival_is_monotone_and_median_is_consistent():
    s = hazard_to_survival(np.array([0.1, 0.2, 0.5, 0.9]))
    assert np.all(np.diff(s) <= 0)
    assert median_horizon(s) == 3


def test_kaplan_meier_recovers_a_known_curve():
    # 4 origins, events at h=1 and h=2, two censored at h=3
    y = np.array([1, 2, 3, 3]); e = np.array([1, 1, 0, 0])
    km = kaplan_meier(y, e, H=3)
    assert km[0] == pytest.approx(0.75)
    assert km[1] == pytest.approx(0.75 * (1 - 1 / 3))
    assert km[2] == pytest.approx(km[1])


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        build_targets(np.zeros(5), EPS, H=0)
    with pytest.raises(ValueError):
        censoring_rate([])
