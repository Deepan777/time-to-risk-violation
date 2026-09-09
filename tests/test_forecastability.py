"""The negative control, tested at the generator level.

`experimental_protocol.md` section 2 requires a generator-side test asserting
`I(S_{1:t}; 1[T_t <= h]) ~ 0` on class-U streams, via a permutation test. This file is that test,
and it is the precondition for Gate 0: if the *generator* leaks, every downstream equivalence
result is meaningless, because the method would be learning something genuinely present.

Two design points that the test enforces rather than assumes:

* `tau` is drawn independently per stream (`make_u_stream`). With `tau` pinned, time-to-violation
  is exactly `tau - t` and the state's own `ctx_deployment_age` predicts it perfectly.
* Only origins **strictly before** `tau` are used. After `tau` the conditional has already changed
  and the delayed feedback block legitimately registers it — that is detection, which the method is
  allowed to do. The claim under test is about *prognosis* before the event.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.datasets import make_synthetic_stream
from src.monitoring import build_state_table, fit_reference
from src.predictors.base import fit_on_pre_deployment
from src.shifts import make_u_stream
from src.validity.targets import build_targets

WINDOW = 200
DEPLOY_START = 7200
H_TEST = 5
N_STREAMS = 6
N_PERMUTATIONS = 200


def _pooled_u_origins(n_streams: int = N_STREAMS) -> tuple[np.ndarray, np.ndarray]:
    """Pool pre-tau origins across several U streams, each with its own tau.

    Returns (states, labels) where label = 1[violation within H_TEST windows].
    """
    base = make_synthetic_stream("sea", n_rows=24000, seed=0)
    feats: list[np.ndarray] = []
    labels: list[int] = []

    for i in range(n_streams):
        rng = np.random.default_rng(1000 + i)
        stream, _, tau_row = make_u_stream(base, rng, deployment_start=DEPLOY_START)
        handle, pre, dep = fit_on_pre_deployment(stream, family="xgboost", seed=0,
                                                 pre_deployment_frac=0.30, window_size=WINDOW)
        ref = fit_reference(pre.X(), pre.y(), handle, window_size=WINDOW, seed=0)
        eps = ref.eps_from_quantile(0.75, margin=0.05)
        states, risks = build_state_table(dep, handle, ref, window_size=WINDOW, stride=WINDOW,
                                          delta=1)
        tau_window = (tau_row - DEPLOY_START) // WINDOW

        for rec in build_targets(risks, eps, H=20):
            if rec.origin >= tau_window:      # post-tau origins are detection, not prognosis
                continue
            feats.append(states.iloc[rec.origin].to_numpy(dtype=float))
            labels.append(int(rec.event == 1 and rec.y_tilde <= H_TEST))

    return np.asarray(feats, dtype=float), np.asarray(labels, dtype=int)


def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC, ties handled by average rank."""
    from scipy.stats import rankdata
    pos, neg = int(y_true.sum()), int((1 - y_true).sum())
    if pos == 0 or neg == 0:
        return 0.5
    r = rankdata(score)
    return float((r[y_true == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _cross_fitted_auc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Out-of-fold AUC from a linear probe. A probe that cannot separate is evidence the
    information is absent; a probe that can is evidence it is present."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, 4)
    scores = np.zeros(len(y), dtype=float)
    for f in folds:
        train = np.setdiff1d(idx, f)
        if len(np.unique(y[train])) < 2:
            scores[f] = 0.0
            continue
        sc = StandardScaler().fit(X[train])
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X[train]), y[train])
        scores[f] = clf.decision_function(sc.transform(X[f]))
    return _auc(y, scores)


@pytest.fixture(scope="module")
def u_origins():
    return _pooled_u_origins()


def test_u_streams_yield_a_usable_test_set(u_origins):
    X, y = u_origins
    assert len(y) >= 40, f"only {len(y)} pre-tau origins pooled; test would be underpowered"
    assert 0 < y.sum() < len(y), "pre-tau origins are single-class; nothing to test"


def test_no_state_feature_is_individually_informative(u_origins):
    """Cheap screen before the expensive test: no single feature should separate the label
    on a class-U stream. A feature with near-perfect AUC is a precursor that should not exist."""
    X, y = u_origins
    aucs = np.array([abs(_auc(y, X[:, j]) - 0.5) for j in range(X.shape[1])])
    worst = float(aucs.max())
    assert worst < 0.35, (
        f"a single state feature reaches AUC {0.5 + worst:.3f} on a class-U stream; "
        "the generator has leaked an observable precursor"
    )


def test_mutual_information_with_future_violation_is_indistinguishable_from_zero(u_origins):
    """The permutation test required by experimental_protocol.md section 2.

    A linear probe is fitted on the pooled pre-tau states to predict `1[T_t <= h]`. Its out-of-fold
    AUC is compared against the null distribution obtained by permuting the labels. Under the U
    construction the observed AUC must sit inside that null.
    """
    X, y = u_origins
    observed = _cross_fitted_auc(X, y, seed=0)

    rng = np.random.default_rng(0)
    null = np.array([_cross_fitted_auc(X, rng.permutation(y), seed=0)
                     for _ in range(N_PERMUTATIONS)])

    # two-sided p-value, +1 smoothing so it can never be exactly zero
    p = (1.0 + np.sum(np.abs(null - 0.5) >= abs(observed - 0.5))) / (N_PERMUTATIONS + 1.0)
    assert p > 0.05, (
        f"class-U stream carries predictive information: observed AUC={observed:.3f}, "
        f"permutation p={p:.4f} (null mean {null.mean():.3f}). "
        "I(S; 1[T<=h]) is not ~0 - HALT and audit the generator before any other result."
    )


def test_tau_is_actually_random_across_streams():
    """If `tau` were pinned, deployment age alone would predict time-to-violation and the whole
    control would be vacuous. This asserts the draw really varies."""
    base = make_synthetic_stream("sea", n_rows=24000, seed=0)
    taus = [make_u_stream(base, np.random.default_rng(2000 + i),
                          deployment_start=DEPLOY_START)[2] for i in range(8)]
    assert len(set(taus)) >= 6, f"tau is insufficiently variable across streams: {sorted(set(taus))}"


def test_u_construction_leaves_the_input_distribution_untouched():
    """P(X) must be bit-identical: a class-U shift replaces P(Y|X) only."""
    base = make_synthetic_stream("sea", n_rows=24000, seed=0)
    stream, _, _ = make_u_stream(base, np.random.default_rng(7), deployment_start=DEPLOY_START)
    assert np.array_equal(base.X(), stream.X()), "class-U shift altered P(X)"
    assert not np.array_equal(base.y(), stream.y()), "class-U shift did not alter P(Y|X)"
