"""Tier-D synthetic base streams.

Tier D is for controlled ablation, forecastability-class construction, and the negative control —
never for headline evidence (experimental_protocol.md section 1.4).

The four classical generators come from `river` rather than being re-implemented here, for the same
reason the drift detectors do: a mis-implemented benchmark generator is a silent correctness bug
that would invalidate every Tier-D result. Their originating papers:

* SEA        — Street & Kim, "A streaming ensemble algorithm (SEA) for large-scale
               classification", KDD 2001.
* AGRAWAL    — Agrawal, Imielinski & Swami, "Database mining: a performance perspective",
               IEEE TKDE 5(6), 1993.
* Hyperplane — rotating-hyperplane drift stream, as used by Hulten, Spencer & Domingos,
               "Mining time-changing data streams", KDD 2001.
* RandomRBF  — radial-basis-function generator from the MOA benchmark suite.

`concept_schedule` switches the generating concept at given row indices. For SEA and AGRAWAL a
concept switch changes P(Y|X) while leaving P(X) untouched, which is exactly the mechanism the
class-U negative control needs; `src/shifts/` supplies the remaining mechanisms.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .stream import Stream

__all__ = ["GENERATORS", "make_synthetic_stream", "RIVER_LICENCE"]

#: river is BSD-3-Clause licensed. Recorded in every synthetic stream's provenance so that the
#: dataset table in the manuscript can state a licence for Tier D as it does for every other tier.
RIVER_LICENCE = "BSD-3-Clause (river); generators are published algorithms, see module docstring"

GENERATORS: tuple[str, ...] = ("sea", "agrawal", "hyperplane", "rbf")

_SOURCE_URL = "https://riverml.xyz/latest/api/datasets/synth/"


def _build_generator(name: str, concept: int, seed: int, n_features: int,
                     noise: float = 0.0):
    from river.datasets import synth

    if name == "sea":
        # four variants (thresholds 8, 9, 7, 9.5) on three U(0,10) features; only the first two
        # are informative, which is what makes SEA a useful controlled case.
        return synth.SEA(variant=int(concept) % 4, noise=noise, seed=seed)
    if name == "agrawal":
        return synth.Agrawal(classification_function=int(concept) % 10, seed=seed,
                             balance_classes=False, perturbation=noise)
    if name == "hyperplane":
        # mag_change=0 keeps the base stream stationary; drift is applied by src/shifts/ or by a
        # concept schedule, so that the shift mechanism is always ours and always recorded.
        return synth.Hyperplane(seed=seed + int(concept), n_features=n_features,
                                n_drift_features=2, mag_change=0.0,
                                noise_percentage=max(noise, 0.05), sigma=0.1)
    if name == "rbf":
        return synth.RandomRBF(seed_model=seed + int(concept), seed_sample=seed,
                               n_classes=2, n_features=n_features, n_centroids=20)
    raise ValueError(f"unknown generator {name!r}; expected one of {GENERATORS}")


def make_synthetic_stream(
    generator: str,
    n_rows: int,
    seed: int,
    *,
    concept_schedule: Sequence[tuple[int, int]] | None = None,
    n_features: int = 10,
    noise: float = 0.10,
    dataset_id: str | None = None,
) -> Stream:
    """Build a Tier-D `Stream`.

    Parameters
    ----------
    generator : one of GENERATORS.
    n_rows : stream length in samples.
    seed : sampling seed. The same seed reproduces the stream exactly.
    concept_schedule : ``[(start_row, concept_id), ...]``, sorted, first entry starting at 0.
        Between switch points the generating concept is held fixed. ``None`` means a single
        stationary concept — the default, because shift is normally applied by `src/shifts/`
        so that it carries a manifest.
    n_features : feature count for the generators that accept one (hyperplane, rbf).
    noise : label noise / perturbation. Defaults to 0.10 rather than 0. A noiseless generator is
        perfectly separable, a gradient-boosted predictor drives its pre-deployment risk to ~0, and
        the validity threshold then collapses to zero - leaving no origin at which the model is
        valid and no estimand to learn (decision D21). Real benchmarks always carry irreducible
        error; the synthetic ones must too.
    """
    generator = generator.lower()
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}; expected one of {GENERATORS}")
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")

    schedule = list(concept_schedule) if concept_schedule else [(0, 0)]
    schedule.sort(key=lambda p: p[0])
    if schedule[0][0] != 0:
        raise ValueError("concept_schedule must start at row 0")
    if len(schedule) > 1 and len({s for s, _ in schedule}) != len(schedule):
        raise ValueError("concept_schedule has duplicate switch points")

    # segment boundaries: [(start, stop, concept), ...]
    bounds = [(s, (schedule[i + 1][0] if i + 1 < len(schedule) else n_rows), c)
              for i, (s, c) in enumerate(schedule)]

    rows: list[dict] = []
    labels: list[int] = []
    for start, stop, concept in bounds:
        take = stop - start
        if take <= 0:
            continue
        gen = _build_generator(generator, concept, seed, n_features, noise)
        for x, y in gen.take(take):
            rows.append(x)
            labels.append(int(y))

    if len(rows) != n_rows:  # pragma: no cover - generator exhaustion should not happen
        raise RuntimeError(f"generator produced {len(rows)} rows, expected {n_rows}")

    df = pd.DataFrame(rows)
    df.columns = [f"f{i}" for i in range(df.shape[1])]
    feature_cols = list(df.columns)
    df["y"] = labels
    # strictly increasing integer time index: synthetic streams have no natural clock, and the
    # canonical Stream requires strict monotonicity.
    df.insert(0, "t", np.arange(n_rows, dtype=np.int64))

    concepts = sorted({c for _, c in schedule})
    return Stream(
        df=df,
        time_col="t",
        feature_cols=feature_cols,
        target_col="y",
        task="binary",
        meta={
            "dataset_id": dataset_id or f"synth_{generator}",
            "tier": "D",
            "family": "synthetic",
            "generator": generator,
            "licence": RIVER_LICENCE,
            "source_url": _SOURCE_URL,
            "seed": int(seed),
            "noise": float(noise),
            "n_rows": int(n_rows),
            "concept_schedule": [list(p) for p in schedule],
            "concepts_used": concepts,
            "domain": "synthetic",
            "task": "binary classification",
            "reason_for_inclusion":
                "Tier D: controlled ablation, forecastability-class construction, negative "
                "control. Never used as headline evidence.",
        },
    )
