"""Controlled-shift generators, including the mandatory unforecastable class.

Mechanism (what changes) and schedule (how it arrives) are orthogonal: `schedules.severity_profile`
supplies a per-row weight, and `apply_mechanism` below consumes it. Seven mechanisms times five
schedules covers the list in experimental_protocol.md section 2 without thirty-five special cases.

One semantic distinction is worth stating because it is easy to get wrong. Under an
``incremental`` schedule the *magnitude* of the change ramps up, so every row is slightly shifted.
Under a ``gradual`` schedule the *probability* of drawing from the new concept ramps up, so each
row is either fully shifted or not at all. Both are called "gradual drift" in the literature; they
are different generative processes and both are needed.

The first four functions are the Phase-II proof-of-concept primitives. They are kept with their
original signatures because `tests/test_shift_generation.py` pins their behaviour.
"""
from __future__ import annotations

import numpy as np

from .manifest import (MECHANISMS, OBSERVABLE_IN_PX, SCHEDULES, Forecastability, ShiftManifest,
                       ShiftSpec, infer_forecastability)
from .schedules import severity_profile

__all__ = [
    "ShiftManifest", "ShiftSpec", "Forecastability", "MECHANISMS", "SCHEDULES",
    "infer_forecastability",
    "covariate_shift", "conditional_shift", "gradual_covariate_shift",
    "make_unforecastable_stream", "apply_mechanism", "apply_shift",
]


# --------------------------------------------------------------------------- primitives
# (original signatures preserved; pinned by tests/test_shift_generation.py)

def covariate_shift(X: np.ndarray, severity: float, features: tuple[int, ...]) -> np.ndarray:
    """Translate the marginal of the named features. P(Y|X) is untouched by construction."""
    Xo = X.copy()
    Xo[:, list(features)] += severity * np.std(X[:, list(features)], axis=0, keepdims=True)
    return Xo


def conditional_shift(y: np.ndarray, rng: np.random.Generator, severity: float) -> np.ndarray:
    """Flip a fraction of labels: changes P(Y|X) while leaving P(X) exactly unchanged."""
    yo = y.copy()
    idx = rng.choice(y.size, size=int(round(severity * y.size)), replace=False)
    yo[idx] = 1 - yo[idx]
    return yo


def gradual_covariate_shift(X: np.ndarray, t0: int, t1: int, severity: float,
                            features: tuple[int, ...]) -> np.ndarray:
    """A precursor-bearing (class F) shift: the marginal drifts smoothly between t0 and t1."""
    Xo = X.copy()
    n = X.shape[0]
    ramp = np.clip((np.arange(n) - t0) / max(t1 - t0, 1), 0.0, 1.0)[:, None]
    sd = np.std(X[:, list(features)], axis=0, keepdims=True)
    Xo[:, list(features)] += severity * ramp * sd
    return Xo


def make_unforecastable_stream(X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                               tau_range: tuple[int, int], severity: float = 0.5
                               ) -> tuple[np.ndarray, np.ndarray, ShiftManifest]:
    """Class-U stream: at a time tau drawn independently of every observable, P(Y|X) is replaced.

    P(X) is left bit-identical, no context feature encodes tau, and nothing before tau differs in
    distribution from a no-shift stream. Any method that appears to forecast this is leaking.
    """
    tau = int(rng.integers(*tau_range))
    y_out = y.copy()
    y_out[tau:] = conditional_shift(y[tau:], rng, severity)
    man = ShiftManifest(shift_type="conditional_abrupt_exogenous", start_time=tau,
                        end_time=int(y.size), severity=severity, affected_features=(),
                        ground_truth_change_point=tau, forecastability_type="U",
                        schedule="abrupt", mechanism="conditional")
    return X.copy(), y_out, man


# --------------------------------------------------------------------------- mechanisms

def _row_weights(w: np.ndarray, schedule: str, rng: np.random.Generator) -> np.ndarray:
    """Turn the severity profile into the per-row multiplier the mechanism applies.

    Under ``gradual`` the profile is a mixing probability, so we draw a Bernoulli mask: a row is
    either fully in the new concept or fully in the old one. Under every other schedule the profile
    scales the magnitude directly.
    """
    if schedule == "gradual":
        return (rng.random(w.shape[0]) < w).astype(float)
    return w


def apply_mechanism(
    X: np.ndarray,
    y: np.ndarray,
    spec: ShiftSpec,
    w: np.ndarray,
    features: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one mechanism under a precomputed severity profile. Returns (X', y').

    Reference scale (`sd`) is taken from the *unshifted* input, so severity means the same thing
    regardless of what a previous compound component already did.
    """
    Xo, yo = X.copy(), y.copy()
    cols = list(features)
    m = _row_weights(w, spec.schedule, rng) * spec.severity   # per-row effective severity
    active = m > 0.0
    if not active.any():
        return Xo, yo

    sd = np.std(X, axis=0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)

    if spec.mechanism == "covariate":
        Xo[:, cols] += m[:, None] * sd[:, cols]

    elif spec.mechanism == "regime":
        # move into a region the predictor never saw during pre-deployment: a large translation
        # plus a sign flip on the affected coordinates, so it is not merely "more of the same".
        Xo[:, cols] = Xo[:, cols] * (1.0 - 2.0 * m[:, None]) + 3.0 * m[:, None] * sd[:, cols]

    elif spec.mechanism == "sensor_corruption":
        gain = 1.0 + 0.5 * m[:, None]
        bias = 0.75 * m[:, None] * sd[:, cols]
        Xo[:, cols] = Xo[:, cols] * gain + bias

    elif spec.mechanism == "noise_escalation":
        Xo[:, cols] += rng.normal(size=(X.shape[0], len(cols))) * m[:, None] * sd[:, cols]

    elif spec.mechanism == "feature_dropout":
        # a sensor going offline reads a constant: blend toward the reference mean
        ref_mean = np.mean(X[:, cols], axis=0, keepdims=True)
        Xo[:, cols] = (1.0 - m[:, None]) * Xo[:, cols] + m[:, None] * ref_mean

    elif spec.mechanism == "conditional":
        # P(Y|X) changes; P(X) is untouched. Flip each active row's label with probability m.
        flip = rng.random(y.shape[0]) < m
        yo[flip] = 1 - yo[flip]

    elif spec.mechanism == "prior":
        # P(Y) changes while P(X|Y) is preserved: replace active rows with rows resampled from a
        # single class pool. Resampling (rather than relabelling) is what keeps P(X|Y) intact.
        target = int(np.argmin(np.bincount(y.astype(int), minlength=2)))  # the rarer class
        pool = np.flatnonzero(y == target)
        if pool.size:
            take = rng.random(y.shape[0]) < m
            src = rng.choice(pool, size=int(take.sum()), replace=True)
            Xo[take] = X[src]
            yo[take] = y[src]
    else:
        raise ValueError(f"unknown mechanism {spec.mechanism!r}; expected one of {MECHANISMS}")

    return Xo, yo


def _default_features(spec: ShiftSpec, n_features: int, rng: np.random.Generator
                      ) -> tuple[int, ...]:
    """Choose affected features when the spec does not name them. Recorded in the manifest, so the
    choice is reproducible and auditable rather than implicit."""
    if spec.mechanism in ("conditional", "prior"):
        return ()   # these act on labels; naming input features would be misleading
    k = max(1, int(round(spec.severity * n_features)))
    k = min(k, n_features)
    return tuple(sorted(rng.choice(n_features, size=k, replace=False).tolist()))


def apply_shift(
    X: np.ndarray,
    y: np.ndarray,
    spec: ShiftSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, ShiftManifest]:
    """Apply one shift spec to arrays, returning the shifted arrays and the manifest.

    Refuses to emit a stream tagged ``U`` whose mechanism touches P(X): the negative control is
    only a control if the inputs really carry no precursor (implementation_plan.md section 3,
    failure cases).
    """
    n_rows, n_features = X.shape
    spec.validate(n_rows)
    end = n_rows if spec.end is None else spec.end

    forecastability = spec.derived_forecastability
    if forecastability == "U" and spec.mechanism in OBSERVABLE_IN_PX:
        raise ValueError(
            f"refusing to emit a U-class stream with mechanism {spec.mechanism!r}: it changes "
            "P(X) and therefore leaves an observable precursor. A U stream must be unobservable "
            "before the event."
        )

    features = spec.features if spec.features is not None else _default_features(spec, n_features, rng)
    if features and max(features) >= n_features:
        raise ValueError(f"affected feature index {max(features)} exceeds n_features={n_features}")

    w = severity_profile(spec.schedule, n_rows, spec.start, end,
                         period=spec.period, jump_frac=spec.jump_frac, rng=rng)
    Xo, yo = apply_mechanism(X, y, spec, w, features, rng)

    # the change point is the first row where the profile actually bites
    nz = np.flatnonzero(w > 0)
    change_point = int(nz[0]) if nz.size else int(spec.start)

    man = ShiftManifest(
        shift_type=f"{spec.mechanism}_{spec.schedule}",
        start_time=int(spec.start),
        end_time=int(end),
        severity=float(spec.severity),
        affected_features=tuple(int(f) for f in features),
        ground_truth_change_point=change_point,
        forecastability_type=forecastability,  # type: ignore[arg-type]
        schedule=spec.schedule,
        mechanism=spec.mechanism,
        period=spec.period,
        jump_frac=float(spec.jump_frac),
        label=spec.label or f"{spec.mechanism}/{spec.schedule}/sev{spec.severity:g}",
    )
    return Xo, yo, man
