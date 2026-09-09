"""Building monitored streams from a dataset, and caching the expensive part.

One "stream" here is one *deployment*: a dataset paired with a particular frozen base predictor.
Running several seeds over the same dataset therefore produces several genuinely different risk
trajectories - a different frozen model degrades differently on the same world - which is what
gives the estimand enough origins on a real benchmark to fit anything.

**Why the split can stay chronological across pooled seeds.** Every seed shares one time axis,
because they monitor the same underlying stream. Splitting pooled origins by origin index with a
gap of `H` therefore still guarantees that every training origin's horizon closes before the first
calibration origin opens, whichever seed it came from. Pooling adds origins without weakening the
temporal guarantee.

State construction is the dominant cost (~2 minutes per Yearbook stream), so results are cached on
disk keyed by every input that could change them.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..datasets.stream import Stream
from ..monitoring import build_state_table, fit_reference
from ..predictors.base import fit_on_pre_deployment
from ..utils.io import ensure_dir

__all__ = ["StreamSpec", "build_entries", "DATASET_REGISTRY", "DATASET_FAMILY", "load_dataset"]

CACHE_DIR = Path("data/cache/monitored")


def load_dataset(dataset_id: str) -> Stream:
    """Load a registered dataset by id."""
    if dataset_id == "yearbook":
        from ..datasets.wildtime import load_yearbook
        return load_yearbook()
    if dataset_id == "huffpost":
        from ..datasets.wildtime import load_huffpost
        return load_huffpost()
    if dataset_id == "elec2":
        from ..datasets.tier_b import load_elec2
        return load_elec2()
    if dataset_id.startswith("insects"):
        from ..datasets.tier_b import load_insects
        return load_insects()
    from ..datasets.mechanism import MECHANISM_DATASETS, load_mechanism
    if dataset_id in MECHANISM_DATASETS:
        return load_mechanism(dataset_id)
    raise ValueError(f"unknown dataset {dataset_id!r}")


#: dataset id -> (predictor family, pre-deployment fraction, window size)
DATASET_REGISTRY: dict[str, tuple[str, float, int]] = {
    "yearbook": ("cnn", 0.20, 200),
    "huffpost": ("xgboost", 0.20, 300),
    "elec2": ("xgboost", 0.20, 200),
    "insects": ("xgboost", 0.20, 200),
    # PR-H1 drift-mechanism datasets. The window size is not a free choice here: it is whatever
    # `mechanism.choose_window` returns under the rule fixed in that module's docstring, and
    # `build_entries` refuses to proceed if these values ever drift apart from it.
    "gas_drift": ("xgboost", 0.20, 100),
    "air_quality": ("xgboost", 0.20, 50),
    "gas_temp_mod": ("xgboost", 0.20, 400),
    "news_popularity": ("xgboost", 0.20, 200),
    "metropt3": ("xgboost", 0.20, 400),
    "cmapss": ("xgboost", 0.20, 400),
}

#: Gate 1 requires >= 2 dataset *families*; this is the mapping it is judged on.
DATASET_FAMILY: dict[str, str] = {
    "yearbook": "tier_a_wildtime",
    "huffpost": "tier_a_wildtime_text",
    "elec2": "tier_b_tabular",
    "insects": "tier_b_tabular",
    "gas_drift": "physical_sensor",
    "air_quality": "physical_sensor",
    "gas_temp_mod": "physical_sensor",
    "news_popularity": "social_editorial",
    "metropt3": "physical_pneumatic",
    "cmapss": "physical_turbomachinery",
}


class StreamSpec:
    """Everything that determines a monitored stream, and therefore its cache key."""

    def __init__(self, dataset_id: str, seed: int, *, window_size: int, delta: int,
                 eps_quantile: float, eps_margin: float, pre_deployment_frac: float,
                 family: str):
        self.dataset_id, self.seed = dataset_id, seed
        self.window_size, self.delta = window_size, delta
        self.eps_quantile, self.eps_margin = eps_quantile, eps_margin
        self.pre_deployment_frac, self.family = pre_deployment_frac, family

    def key(self) -> str:
        raw = "|".join(str(v) for v in (
            self.dataset_id, self.seed, self.window_size, self.delta, self.eps_quantile,
            self.eps_margin, self.pre_deployment_frac, self.family))
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def stream_id(self) -> str:
        return f"{self.dataset_id}_seed{self.seed}"


def build_entries(
    dataset_ids: Sequence[str],
    seeds: Sequence[int],
    *,
    eps_quantile: float = 0.75,
    eps_margin: float = 0.05,
    delta: int = 1,
    use_cache: bool = True,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Monitor every (dataset, seed) pair and return the per-stream entries.

    The returned dicts are exactly what `build_validity_dataset` consumes.
    """
    ensure_dir(CACHE_DIR)
    entries: list[dict] = []

    for dataset_id in dataset_ids:
        family, pre_frac, window = DATASET_REGISTRY[dataset_id]
        base: Stream | None = None

        for seed in seeds:
            spec = StreamSpec(dataset_id, seed, window_size=window, delta=delta,
                              eps_quantile=eps_quantile, eps_margin=eps_margin,
                              pre_deployment_frac=pre_frac, family=family)
            cache = CACHE_DIR / f"{spec.stream_id()}_{spec.key()}.pkl"

            if use_cache and cache.is_file():
                entries.append(pickle.loads(cache.read_bytes()))
                log(f"  cached  {spec.stream_id()}")
                continue

            if base is None:
                base = load_dataset(dataset_id)
                declared = base.meta.get("window_size")
                if declared is not None and int(declared) != window:
                    raise ValueError(
                        f"{dataset_id}: DATASET_REGISTRY says window={window} but the fixed "
                        f"windowing rule in src/datasets/mechanism.py returns {declared}. The "
                        "rule decides; edit the registry, not the rule."
                    )
            handle, pre, dep = fit_on_pre_deployment(
                base, family=family, seed=seed,
                pre_deployment_frac=pre_frac, window_size=window)
            ref = fit_reference(pre.X(), pre.y(), handle, window_size=window, seed=seed)
            eps = ref.eps_from_quantile(eps_quantile, margin=eps_margin)
            states, risks = build_state_table(dep, handle, ref, window_size=window,
                                              stride=window, delta=delta)

            entry = {
                "stream_id": spec.stream_id(),
                "dataset_id": dataset_id,
                "family": DATASET_FAMILY[dataset_id],
                "seed": int(seed),
                "states": states,
                "risks": risks,
                "eps": float(eps),
                # Natural-shift benchmarks carry no generator tag. Calling them "F" would be an
                # assertion the data cannot support, so the class is recorded as unknown and these
                # streams are simply excluded from the taxonomy analysis (PR1).
                "forecastability": "NATURAL",
                "provenance": {
                    "dataset_meta": {k: v for k, v in base.meta.items()
                                     if k not in ("input_shape",)},
                    "predictor": handle.params(),
                    "reference": dict(ref.meta),
                    "eps": float(eps),
                    "eps_rule": f"quantile({eps_quantile}) + margin({eps_margin})",
                    "window_size": window,
                    "label_delay": delta,
                },
            }
            if use_cache:
                cache.write_bytes(pickle.dumps(entry))
            entries.append(entry)
            log(f"  built   {spec.stream_id()}: {len(risks)} windows, "
                f"eps={eps:.4f}, risk {risks.min():.3f}-{risks.max():.3f}")

        base = None      # release the dataset between datasets; Yearbook alone is ~0.9 GB

    return entries
