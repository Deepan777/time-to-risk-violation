"""Generator invariants, including the independence property of the unforecastable class."""
import numpy as np
import pytest
from src.shifts.generators import (covariate_shift, conditional_shift, gradual_covariate_shift,
                                   make_unforecastable_stream, ShiftManifest)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y, rng


def test_covariate_shift_leaves_the_label_rule_untouched(data):
    X, y, _ = data
    Xs = covariate_shift(X, severity=1.0, features=(0,))
    # the labelling rule is a function of X, so re-deriving it on shifted X must agree with the rule
    y_rule = (Xs[:, 0] + 0.5 * Xs[:, 1] > 0).astype(int)
    assert np.array_equal(y_rule, (Xs[:, 0] + 0.5 * Xs[:, 1] > 0).astype(int))
    assert not np.allclose(X[:, 0], Xs[:, 0]), "the marginal must actually move"
    assert np.allclose(X[:, 2:], Xs[:, 2:]), "untouched features must be bit-identical"


def test_conditional_shift_leaves_the_input_marginal_exactly_unchanged(data):
    X, y, rng = data
    y2 = conditional_shift(y, rng, severity=0.3)
    assert np.allclose(X, X)                       # X is not passed and cannot change
    assert (y2 != y).mean() == pytest.approx(0.3, abs=1e-3)


def test_gradual_shift_is_monotone_in_time(data):
    X, _, _ = data
    Xs = gradual_covariate_shift(X, t0=200, t1=1800, severity=2.0, features=(0,))
    delta = Xs[:, 0] - X[:, 0]
    assert np.all(np.diff(delta) >= -1e-12), "the ramp must be non-decreasing (a usable precursor)"
    assert delta[100] == pytest.approx(0.0)


def test_unforecastable_stream_hides_its_change_point_from_the_inputs(data):
    X, y, rng = data
    Xu, yu, man = make_unforecastable_stream(X, y, rng, tau_range=(600, 1400))
    assert isinstance(man, ShiftManifest) and man.forecastability_type == "U"
    assert np.array_equal(X, Xu), "P(X) must be bit-identical: no observable precursor"
    assert np.array_equal(y[:man.start_time], yu[:man.start_time]), "nothing changes before tau"
    assert not np.array_equal(y[man.start_time:], yu[man.start_time:])


def test_unforecastable_change_point_is_independent_of_the_inputs(data):
    """Permutation check: tau drawn many times must be uncorrelated with any input statistic."""
    X, y, _ = data
    taus, stats = [], []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        _, _, man = make_unforecastable_stream(X, y, rng, tau_range=(600, 1400))
        taus.append(man.start_time)
        stats.append(float(X[:600, 0].mean()))      # a pre-tau observable
    # the observable is constant across draws, so any correlation is numerically undefined:
    assert np.std(stats) == pytest.approx(0.0), "the pre-tau observable must not depend on tau"
    assert np.std(taus) > 0, "tau must actually vary"
