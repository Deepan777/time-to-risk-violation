"""Metric correctness, and the guards that stop misleading metrics being reported alone."""
import numpy as np
import pytest
from src.metrics.validity_metrics import vhmae, pvlt, warning_metrics, conditioned_mask


def test_vhmae_uses_uncensored_only_and_returns_the_censoring_rate():
    res = vhmae(pred_h=np.array([3.0, 9.0, 5.0, 7.0]),
                true_h=np.array([4.0, 1.0, 6.0, 2.0]),
                event=np.array([1, 0, 1, 0]))
    assert res.vhmae == pytest.approx(1.0)          # |3-4| and |5-6| only
    assert res.n_uncensored == 2
    assert res.censoring_rate == pytest.approx(0.5)


def test_vhmae_refuses_when_everything_is_censored():
    with pytest.raises(ValueError):
        vhmae(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0, 0]))


def test_pvlt_drops_cases_without_a_warning_or_without_a_violation():
    lead = pvlt(np.array([10.0, 20.0, np.nan]), np.array([6.0, np.nan, 3.0]))
    assert lead.tolist() == [4.0]


def test_warning_metrics_are_arithmetically_correct():
    m = warning_metrics(warned=np.array([1, 1, 0, 0, 1], bool),
                        will_violate=np.array([1, 0, 1, 0, 1], bool))
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["false_alarms_per_100_valid_windows"] == pytest.approx(20.0)


def test_conditioned_mask_is_the_gate3_protocol():
    m = conditioned_mask(np.array([0.1, 0.4, 0.2]), np.array([0.2, 0.1, 0.5]), eps=0.3)
    assert m.tolist() == [True, False, False]


def test_conditioned_mask_refuses_to_return_an_empty_subset():
    with pytest.raises(ValueError):
        conditioned_mask(np.array([0.9, 0.9]), np.array([0.9, 0.9]), eps=0.3)
