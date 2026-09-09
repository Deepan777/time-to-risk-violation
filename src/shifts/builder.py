"""Stream-level shift application, including compound shifts.

`apply_shift` in `generators.py` works on arrays and handles one mechanism. This module lifts that
to the canonical `Stream` type and composes several specs into a compound stream, which is the
last mechanism named in experimental_protocol.md section 2.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..datasets.stream import Stream
from .generators import apply_shift
from .manifest import ShiftManifest, ShiftSpec

__all__ = ["build_shifted_stream", "compound_forecastability", "make_u_stream"]


def compound_forecastability(manifests: Sequence[ShiftManifest]) -> str:
    """Forecastability of a stream carrying several shifts.

    A stream is only F if every component is forecastable; it is only U if every component is
    unforecastable. Anything mixed is partially forecastable, because part of the risk path has a
    precursor and part does not — which is exactly the PF definition in
    proposed_methodology.md section 3.2.
    """
    if not manifests:
        raise ValueError("no shift manifests: an unshifted stream has no forecastability class")
    classes = {m.forecastability_type for m in manifests}
    if classes == {"F"}:
        return "F"
    if classes == {"U"}:
        return "U"
    return "PF"


def build_shifted_stream(
    base: Stream,
    specs: Sequence[ShiftSpec],
    rng: np.random.Generator,
    *,
    stream_id: str | None = None,
    deployment_start: int = 0,
) -> tuple[Stream, list[ShiftManifest]]:
    """Apply `specs` in order to `base`, returning the shifted stream and one manifest per spec.

    `deployment_start` offsets every spec's window, so a spec can be written in deployment-relative
    coordinates ("severity ramps from deployment window 10") while the manifest records absolute
    row indices. Shifts are never applied to the pre-deployment segment: the predictor is fitted
    there, and a shift inside it would be learned rather than suffered.
    """
    if not specs:
        raise ValueError("build_shifted_stream requires at least one ShiftSpec")
    if not 0 <= deployment_start < base.n_rows:
        raise ValueError(f"deployment_start {deployment_start} outside [0, {base.n_rows})")

    X = base.X()
    y = base.y()
    manifests: list[ShiftManifest] = []

    for spec in specs:
        shifted = ShiftSpec(
            mechanism=spec.mechanism,
            schedule=spec.schedule,
            severity=spec.severity,
            start=spec.start + deployment_start,
            end=(spec.end + deployment_start) if spec.end is not None else None,
            features=spec.features,
            forecastability=spec.forecastability,
            period=spec.period,
            jump_frac=spec.jump_frac,
            label=spec.label,
        )
        X, y, man = apply_shift(X, y, shifted, rng)
        manifests.append(man)

    df = base.df.copy()
    df[base.feature_cols] = X
    df[base.target_col] = y

    overall = compound_forecastability(manifests)
    meta = {
        **base.meta,
        "dataset_id": stream_id or f"{base.dataset_id}__shifted",
        "base_dataset_id": base.dataset_id,
        "deployment_start": int(deployment_start),
        "forecastability_type": overall,
        "shift_manifests": [m.to_dict() for m in manifests],
        "n_shift_components": len(manifests),
    }
    out = Stream(df=df, time_col=base.time_col, feature_cols=list(base.feature_cols),
                 target_col=base.target_col, task=base.task, meta=meta)
    return out, manifests


def make_u_stream(
    base: Stream,
    rng: np.random.Generator,
    *,
    deployment_start: int,
    tau_range: tuple[float, float] = (0.15, 0.75),
    severity: float = 0.45,
    stream_id: str | None = None,
) -> tuple[Stream, list[ShiftManifest], int]:
    """Build one class-U stream with `tau` drawn uniformly, and return `tau` in absolute rows.

    **Why tau must be random, and why this helper exists.**

    `proposed_methodology.md` section 3.3 requires `tau ~ Uniform`, drawn independently of every
    observable. It is tempting to treat that as a detail and pin `tau` to a fixed offset for
    reproducibility. That silently destroys the negative control.

    With `tau` fixed across streams, time-to-violation from origin `t` is exactly `tau - t`, and
    the state already contains `ctx_deployment_age = t`. A model would then "forecast" the
    unforecastable perfectly by reading its own clock — and the pipeline would look leaky when the
    real fault was in the stream design. Drawing `tau` per stream makes deployment age uninformative
    about the *remaining* time, which is precisely the property the control is asserting.

    `tau_range` is expressed as a fraction of the deployment segment, so the same range transfers
    across streams of different lengths.
    """
    lo, hi = tau_range
    if not 0.0 < lo < hi < 1.0:
        raise ValueError(f"tau_range must satisfy 0 < lo < hi < 1, got {tau_range}")
    dep_len = base.n_rows - deployment_start
    if dep_len <= 0:
        raise ValueError("deployment segment is empty")

    tau_offset = int(rng.uniform(lo, hi) * dep_len)
    spec = ShiftSpec(mechanism="conditional", schedule="abrupt", severity=severity,
                     start=tau_offset, label="negative control (class U)")
    stream, manifests = build_shifted_stream(base, [spec], rng, stream_id=stream_id,
                                             deployment_start=deployment_start)
    assert manifests[0].forecastability_type == "U", "class-U construction produced a non-U tag"
    return stream, manifests, deployment_start + tau_offset
