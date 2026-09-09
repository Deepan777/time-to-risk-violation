"""Tier-B: long tabular streams with real temporal dependence.

`experimental_protocol.md` section 1.2 admits these **for comparability with prior work only**,
never as primary evidence, and attaches a caveat that must travel into the paper: ELEC2 and
Airlines have strong temporal autocorrelation, which makes naive persistence a deceptively strong
baseline. That caveat is recorded in each stream's metadata under `caveat` so the table generator
can print it rather than relying on anyone remembering.

Sources are `river`'s dataset module, so nothing is re-implemented and the download is the one the
streaming-benchmark community uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stream import Stream

__all__ = ["TIER_B_DATASETS", "load_elec2", "load_insects", "load_tier_b"]

TIER_B_DATASETS: tuple[str, ...] = ("elec2", "insects")

_RIVER_LICENCE = "BSD-3-Clause (river distribution)"

_AUTOCORR_CAVEAT = (
    "Strong temporal autocorrelation makes naive persistence a deceptively strong baseline on this "
    "stream (experimental_protocol.md section 1.2). Used for comparability with prior work, never "
    "as primary evidence."
)


def _from_river(iterable, *, target_is_bool: bool) -> tuple[pd.DataFrame, list[str]]:
    rows, ys = [], []
    for x, y in iterable:
        rows.append(x)
        ys.append(y)
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns]
    # river yields heterogeneous dicts; anything non-numeric is encoded here, in the loader,
    # rather than left for a downstream block to trip over
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.Categorical(df[c]).codes
    y_arr = np.asarray(ys)
    if target_is_bool or y_arr.dtype == bool:
        y_arr = y_arr.astype(int)
    else:
        y_arr = pd.Categorical(y_arr).codes
    df["y"] = y_arr
    return df, feature_cols


def load_elec2() -> Stream:
    """ELEC2: Australian (NSW) electricity market, 45,312 half-hourly records.

    The task is whether the spot price rises or falls relative to a moving average. The stream is a
    genuine, non-stationary economic process, which is why it has been a streaming benchmark for two
    decades - and why persistence is hard to beat on it.
    """
    from river.datasets import Elec2

    df, feature_cols = _from_river(Elec2(), target_is_bool=True)
    df.insert(0, "t", np.arange(len(df), dtype=np.int64))
    return Stream(
        df=df, time_col="t", feature_cols=feature_cols, target_col="y", task="binary",
        meta={
            "dataset_id": "elec2", "tier": "B", "family": "tier_b_tabular",
            "licence": _RIVER_LICENCE,
            "source_url": "https://riverml.xyz/latest/api/datasets/Elec2/",
            "citation": "Harries, SPLICE-2 comparative evaluation: electricity pricing, 1999.",
            "domain": "electricity market (economics)",
            "task": "binary classification (price up/down vs moving average)",
            "n_rows": int(len(df)),
            "temporal_structure": "half-hourly records, 1996-1998, strictly ordered",
            "shift_type": "natural concept drift (market regime changes)",
            "reason_for_inclusion":
                "Tier B: comparability with the streaming-learning literature; a second dataset "
                "family for the Gate-1 requirement.",
            "caveat": _AUTOCORR_CAVEAT,
        },
    )


def load_insects(variant: str = "abrupt_balanced") -> Stream:
    """Insects: optical sensor readings of flying insects under controlled temperature change.

    Included because the drift here is *physically* generated rather than annotated after the fact:
    the experimenters varied temperature, which changes wing-beat frequency, which changes the
    feature distribution. That gives a real stream whose shift mechanism is actually known.
    """
    from river.datasets import Insects

    df, feature_cols = _from_river(Insects(variant=variant), target_is_bool=False)
    df.insert(0, "t", np.arange(len(df), dtype=np.int64))
    return Stream(
        df=df, time_col="t", feature_cols=feature_cols, target_col="y", task="multiclass",
        meta={
            "dataset_id": f"insects_{variant}", "tier": "B", "family": "tier_b_tabular",
            "licence": _RIVER_LICENCE,
            "source_url": "https://riverml.xyz/latest/api/datasets/Insects/",
            "citation": "Souza et al., Challenges in benchmarking stream learning algorithms "
                        "with real-world data, Data Mining and Knowledge Discovery, 2020.",
            "domain": "entomology / optical sensing",
            "task": "multiclass classification (insect species)",
            "n_rows": int(len(df)),
            "temporal_structure": f"sequential sensor records, variant={variant}",
            "shift_type": "physically induced covariate drift (controlled temperature change)",
            "reason_for_inclusion":
                "Tier B: a real stream whose drift mechanism is known by construction rather than "
                "inferred, which makes it a useful check on the forecastability taxonomy.",
            "caveat": _AUTOCORR_CAVEAT,
        },
    )


def load_tier_b(name: str) -> Stream:
    if name == "elec2":
        return load_elec2()
    if name.startswith("insects"):
        return load_insects(name.split("insects_")[-1] if "_" in name else "abrupt_balanced")
    raise ValueError(f"unknown Tier-B dataset {name!r}; expected one of {TIER_B_DATASETS}")
