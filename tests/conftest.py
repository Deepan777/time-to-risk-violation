"""Shared fixtures.

The pipeline fixtures are session-scoped because building a stream, fitting a predictor and
walking the deployment segment costs a few seconds; repeating that per test would make the suite
slow enough that people stop running it, and a leakage suite nobody runs is worse than none.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.datasets import make_synthetic_stream
from src.monitoring import build_state_table, fit_reference
from src.predictors.base import fit_on_pre_deployment
from src.shifts import ShiftSpec, build_shifted_stream

WINDOW = 200
DELTA = 1
H = 20


@pytest.fixture(scope="session")
def forecastable_pipeline():
    """A class-F stream: gradual covariate drift with an observable precursor."""
    base = make_synthetic_stream("sea", n_rows=24000, seed=0)
    spec = ShiftSpec(mechanism="covariate", schedule="incremental", severity=0.9,
                     start=4000, end=14000)
    stream, mans = build_shifted_stream(base, [spec], np.random.default_rng(0),
                                        deployment_start=7200)
    handle, pre, dep = fit_on_pre_deployment(stream, family="xgboost", seed=0,
                                             pre_deployment_frac=0.30, window_size=WINDOW)
    ref = fit_reference(pre.X(), pre.y(), handle, window_size=WINDOW, seed=0)
    eps = ref.eps_from_quantile(0.75, margin=0.05)
    states, risks = build_state_table(dep, handle, ref, window_size=WINDOW, stride=WINDOW,
                                      delta=DELTA)
    return {"stream": stream, "manifests": mans, "handle": handle, "pre": pre, "dep": dep,
            "ref": ref, "eps": eps, "states": states, "risks": risks}


@pytest.fixture(scope="session")
def unforecastable_pipeline():
    """A class-U stream: P(Y|X) is replaced at an exogenous tau, P(X) is untouched.

    This is the negative control. Nothing observable before `tau` may carry information about it.
    """
    base = make_synthetic_stream("sea", n_rows=24000, seed=0)
    spec = ShiftSpec(mechanism="conditional", schedule="abrupt", severity=0.45, start=4000)
    stream, mans = build_shifted_stream(base, [spec], np.random.default_rng(1),
                                        deployment_start=7200)
    handle, pre, dep = fit_on_pre_deployment(stream, family="xgboost", seed=0,
                                             pre_deployment_frac=0.30, window_size=WINDOW)
    ref = fit_reference(pre.X(), pre.y(), handle, window_size=WINDOW, seed=0)
    eps = ref.eps_from_quantile(0.75, margin=0.05)
    states, risks = build_state_table(dep, handle, ref, window_size=WINDOW, stride=WINDOW,
                                      delta=DELTA)
    # change point expressed in DEPLOYMENT windows
    tau_row = mans[0].ground_truth_change_point
    tau_window = (tau_row - 7200) // WINDOW
    return {"base": base, "stream": stream, "manifests": mans, "handle": handle,
            "ref": ref, "eps": eps, "states": states, "risks": risks,
            "tau_window": tau_window}
