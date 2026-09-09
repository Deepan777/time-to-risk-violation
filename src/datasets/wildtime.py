"""Tier-A natural temporal distribution shift: the Wild-Time suite.

Wild-Time (Yao et al., NeurIPS 2022 Datasets & Benchmarks, arXiv:2211.14238) is the primary Tier-A
source because it is purpose-built for *temporal* natural shift and carries timestamp metadata.
This module converts its per-year dictionaries into the canonical `Stream`.

**Provenance is mandatory.** Every stream records name, domain, task, size, temporal structure,
shift type, reason for inclusion, source URL, licence, and a checksum of the downloaded artefact.
A dataset whose licence cannot be confirmed is dropped, never assumed permissive
(see proposal/implementation_plan.md).

Licences, as stated in the Wild-Time repository:

    Yearbook  MIT
    Huffpost  CC0 (public domain)
    arXiv     CC0 (public domain)
    Drug-BA   MIT
    FMoW      FMoW Challenge Public License
    MIMIC-IV  PhysioNet Credentialed Health Data License 1.5.0 -- REQUIRES CREDENTIALING

MIMIC-IV is deliberately absent from `WILDTIME_DATASETS`: the protocol forbids scheduling it
without confirmed credentialed access, and hosting it on a cloud platform does not grant that.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import sha256_file
from .stream import Stream

__all__ = ["WILDTIME_DATASETS", "WILDTIME_LICENCES", "load_yearbook", "load_huffpost",
           "wildtime_metadata", "HUFFPOST_HASH_FEATURES"]

#: Hashing-vectoriser width for Huffpost headlines. A *hashing* vectoriser is used
#: deliberately: it is stateless, so unlike TF-IDF there is no vocabulary or document
#: frequency fitted on any data at all, and the "reference statistics are frozen" rule
#: cannot be violated even in principle. Fitting a TF-IDF on the whole stream would
#: leak deployment-era vocabulary into the pre-deployment representation.
HUFFPOST_HASH_FEATURES = 512

WILDTIME_LICENCES: dict[str, str] = {
    "yearbook": "MIT",
    "huffpost": "CC0 1.0 Universal (Public Domain Dedication)",
    "arxiv": "CC0 1.0 Universal (Public Domain Dedication)",
    "drug": "MIT",
}

#: Datasets this project will load. MIMIC-IV and FMoW are excluded - see the module docstring and
#: decisions D13 / D14 in PROGRESS.md.
WILDTIME_DATASETS: tuple[str, ...] = ("yearbook", "huffpost", "arxiv", "drug")

_SOURCE_URL = "https://github.com/huaxiuyao/Wild-Time"
_CITATION = ("Yao et al., Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time, "
             "NeurIPS 2022 Datasets and Benchmarks, arXiv:2211.14238")


def wildtime_metadata(name: str, path: Path, **extra) -> dict:
    """The provenance record required for every dataset."""
    if name not in WILDTIME_LICENCES:
        raise ValueError(
            f"no confirmed licence for Wild-Time dataset {name!r}; the dataset is dropped rather "
            "than used under an assumed licence"
        )
    meta = {
        "dataset_id": f"wildtime_{name}",
        "tier": "A",
        "family": "wildtime",
        "licence": WILDTIME_LICENCES[name],
        "source_url": _SOURCE_URL,
        "citation": _CITATION,
        "artefact_path": str(path),
        "artefact_sha256": sha256_file(path),
        "artefact_bytes": int(path.stat().st_size),
        "shift_type": "natural temporal distribution shift (no synthetic shift applied)",
        "reason_for_inclusion":
            "Tier A: genuine natural temporal shift with timestamp metadata; the protocol requires "
            "at least one natural-shift benchmark in every headline experiment.",
    }
    meta.update(extra)
    return meta


def load_yearbook(
    path: str | Path = "data/raw/wildtime/yearbook.pkl",
    *,
    modes: tuple[int, ...] = (0, 1, 2),
    max_per_year: int | None = None,
) -> Stream:
    """US high-school yearbook portraits, 1930-2013; binary task; 32x32x3.

    The shift is genuine and well documented: photographic technology, hairstyles and framing
    conventions change continuously across eight decades while the labelling rule does not. That is
    covariate shift arriving on its own schedule, which is exactly what the estimand is for.

    Wild-Time splits each year into modes (train / val / test) for its own protocol. Here all
    requested modes are concatenated **within** each year and the years are laid end to end, because
    this project needs the full deployment stream rather than Wild-Time's held-out evaluation. Order
    within a year is the order Wild-Time stores; the tie-breaking rule for the strictly increasing
    time index is "year, then stored order", recorded here and in the manifest.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download it first (Google Drive id "
            "1mPpxoX2y2oijOvW1ymiHEYd7oMu2vVRb, ~1.83 GB)."
        )

    with path.open("rb") as fh:
        raw = pickle.load(fh)

    years = sorted(raw.keys())
    imgs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    year_col: list[np.ndarray] = []

    for yr in years:
        per_year_img, per_year_lab = [], []
        for m in modes:
            if m not in raw[yr]:
                continue
            # cast to float32 immediately: Wild-Time stores float64, and concatenating 74k
            # 32x32x3 images at that width needs 1.7 GiB before any cast could help
            per_year_img.append(np.asarray(raw[yr][m]["images"], dtype=np.float32))
            per_year_lab.append(np.asarray(raw[yr][m]["labels"]))
        if not per_year_img:
            continue
        xi = np.concatenate(per_year_img, axis=0)
        yi = np.concatenate(per_year_lab, axis=0)
        raw[yr] = None                       # release the pickle's copy as we go
        if max_per_year is not None and len(yi) > max_per_year:
            xi, yi = xi[:max_per_year], yi[:max_per_year]
        imgs.append(xi)
        labels.append(yi)
        year_col.append(np.full(len(yi), yr, dtype=np.int64))

    del raw
    X = np.concatenate(imgs, axis=0)
    imgs.clear()
    y = np.concatenate(labels, axis=0).astype(np.int64)
    yrs = np.concatenate(year_col, axis=0)

    # Wild-Time stores images already scaled to [0, 1]; they are kept at those exact values,
    # only narrowed to float32, so the pixels are what the benchmark distributes.
    X = np.ascontiguousarray(X)

    df = pd.DataFrame({
        "t": np.arange(len(y), dtype=np.int64),   # strictly increasing, as the Stream requires
        "year": yrs,
        "y": y,
    })

    meta = wildtime_metadata(
        "yearbook", path,
        domain="computer vision (historical portrait photography)",
        task="binary classification",
        n_rows=int(len(y)),
        temporal_structure=f"{years[0]}-{years[-1]}, {len(years)} yearly domains",
        input_shape=[int(X.shape[1]), int(X.shape[2]), int(X.shape[3])],
        modes_concatenated=list(modes),
        tie_breaking_rule="year ascending, then Wild-Time stored order within the year",
        class_balance=float(y.mean()),
    )

    return Stream(df=df, time_col="t", feature_cols=[], target_col="y",
                  task="binary", meta=meta, array=X)


def load_huffpost(
    path: str | Path = "data/raw/wildtime/huffpost.pkl",
    *,
    modes: tuple[int, ...] = (0, 1, 2),
    n_features: int = HUFFPOST_HASH_FEATURES,
) -> Stream:
    """HuffPost news headlines by publication year, 2012-2018; 11-category classification.

    The shift is editorial and linguistic rather than visual: the vocabulary of news changes year on
    year as topics, names and framing conventions turn over, while the category taxonomy does not.
    That gives a natural temporal covariate shift in a completely different modality from Yearbook,
    which is what makes it worth adding - the Gate 1 finding is much harder to dismiss as an artefact
    of one dataset if it reproduces across image and text streams.

    Headlines are vectorised with a **stateless** hashing vectoriser (see `HUFFPOST_HASH_FEATURES`),
    so no vocabulary statistic is fitted on any portion of the stream and the frozen-reference rule
    holds by construction rather than by care.
    """
    from sklearn.feature_extraction.text import HashingVectorizer

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download it first (Google Drive id "
            "1jKqbfPx69EPK_fjgU9RLuExToUg7rwIY, ~60 MB).")

    with path.open("rb") as fh:
        raw = pickle.load(fh)

    years = sorted(raw.keys())
    texts: list[str] = []
    labels: list[int] = []
    year_col: list[int] = []
    for yr in years:
        for m in modes:
            if m not in raw[yr]:
                continue
            head = list(raw[yr][m]["headline"])
            cat = list(raw[yr][m]["category"])
            texts.extend(str(t) for t in head)
            labels.extend(int(c) for c in cat)
            year_col.extend([yr] * len(head))

    vec = HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2",
                            lowercase=True, ngram_range=(1, 2))
    X = np.asarray(vec.transform(texts).todense(), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    yrs = np.asarray(year_col, dtype=np.int64)

    df = pd.DataFrame({"t": np.arange(len(y), dtype=np.int64), "year": yrs, "y": y})
    meta = wildtime_metadata(
        "huffpost", path,
        domain="natural language processing (news headlines)",
        task="multiclass classification (11 editorial categories)",
        n_rows=int(len(y)),
        temporal_structure=f"{years[0]}-{years[-1]}, {len(years)} yearly domains",
        input_shape=[int(n_features)],
        vectoriser=f"stateless HashingVectorizer(n_features={n_features}, ngram_range=(1,2), "
                   "norm='l2', alternate_sign=False)",
        modes_concatenated=list(modes),
        tie_breaking_rule="year ascending, then Wild-Time stored order within the year",
        n_classes=int(len(np.unique(y))),
    )
    return Stream(df=df, time_col="t", feature_cols=[], target_col="y",
                  task="multiclass", meta=meta, array=X)
