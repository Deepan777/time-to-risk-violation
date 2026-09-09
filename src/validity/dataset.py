"""The supervised, censored dataset, and the chronological splits.

One record per *valid* origin: a length-`L` window of deployment state, and the censored
time-to-risk-violation it leads to. Origins with `R_t > eps` are excluded by definition of the
estimand (methodology eq. 3).

Two properties are enforced here rather than trusted:

* **Splits are chronological with a gap of `H` windows** between segments, so no training origin's
  prediction horizon can overlap the calibration or test period. Without the gap, a training origin
  near the boundary predicts windows that a test origin is scored on, and the split leaks.
* **Normalisation is fitted on the training split only** and then frozen, like every other
  reference statistic in the project.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd

from ..utils.leakage import assert_no_future_columns
from .targets import build_targets

__all__ = ["ValidityDataset", "build_validity_dataset", "chronological_split",
           "stream_level_split", "Normaliser", "MIN_SPLIT_ORIGINS"]

#: A split smaller than this cannot support early stopping or a meaningful test.
MIN_SPLIT_ORIGINS = 30


@dataclass(frozen=True)
class Normaliser:
    """Frozen feature standardisation, fitted on the training split only."""

    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...]

    @classmethod
    def fit(cls, S: np.ndarray, feature_names: Sequence[str]) -> "Normaliser":
        flat = S.reshape(-1, S.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        return cls(mean=mean, std=np.where(std > 1e-8, std, 1.0),
                   feature_names=tuple(feature_names))

    def transform(self, S: np.ndarray) -> np.ndarray:
        return (S - self.mean) / self.std


@dataclass
class ValidityDataset:
    """Model inputs and targets. Inputs and targets live in separate arrays by construction."""

    S: np.ndarray               # (N, L, d)  deployment-state context windows  -- INPUT
    y_tilde: np.ndarray         # (N,)       observed time, 1..H               -- TARGET
    event: np.ndarray           # (N,)       1 = violation seen, 0 = censored  -- TARGET
    risk_path: np.ndarray       # (N, H)     realised R_{t+1..t+H}, NaN-padded -- TARGET
    risk_hist: np.ndarray       # (N, L)     MATURED risk history              -- BASELINE INPUT
    anchor: np.ndarray          # (N, H)     online level from risk_hist       -- ANCHOR (E16)
    risk_now: np.ndarray        # (N,)       R_t at the origin                 -- PROTOCOL
    risk_next: np.ndarray       # (N,)       R_{t+1}                           -- PROTOCOL
    origin: np.ndarray          # (N,)       window index within its stream    -- METADATA
    stream_id: np.ndarray       # (N,)       object array                      -- METADATA
    eps: np.ndarray             # (N,)       threshold in force at the origin  -- METADATA
    forecastability: np.ndarray # (N,)       'F' / 'PF' / 'U'                  -- METADATA ONLY
    feature_names: tuple[str, ...]
    H: int
    L: int
    normaliser: Normaliser | None = None

    def __len__(self) -> int:
        return int(self.S.shape[0])

    @property
    def n_events(self) -> int:
        return int(self.event.sum())

    @property
    def censoring_rate(self) -> float:
        return float(1.0 - self.event.mean()) if len(self) else float("nan")

    def subset(self, idx: np.ndarray) -> "ValidityDataset":
        idx = np.asarray(idx)
        return replace(
            self,
            S=self.S[idx], y_tilde=self.y_tilde[idx], event=self.event[idx],
            risk_path=self.risk_path[idx], risk_hist=self.risk_hist[idx],
            anchor=self.anchor[idx], risk_now=self.risk_now[idx],
            risk_next=self.risk_next[idx], origin=self.origin[idx],
            stream_id=self.stream_id[idx], eps=self.eps[idx],
            forecastability=self.forecastability[idx],
        )

    def with_normaliser(self, norm: Normaliser) -> "ValidityDataset":
        return replace(self, S=norm.transform(self.S), normaliser=norm)

    def conditioned_mask(self) -> np.ndarray:
        """The Gate-3 protocol subset: valid now **and** still valid next window.

        On this subset a pure detector cannot score, because nothing has yet gone wrong. It is what
        separates prognosis from detection, and it is the most important subset in the project.
        """
        m = (self.risk_now <= self.eps) & (self.risk_next <= self.eps) & np.isfinite(self.risk_next)
        if not m.any():
            raise ValueError(
                "conditioning mask is empty: no origin is valid now and still valid next window. "
                "Reporting a conditioned metric over nothing is prohibited."
            )
        return m


def _context_window(states: np.ndarray, t: int, L: int) -> np.ndarray:
    """States `s_{t-L+1..t}`, left-padded by repeating the earliest available row.

    Zero-padding would inject an artificial regime that the encoder could learn to recognise as
    "early in the stream"; edge replication keeps the padded region distributionally plausible.
    """
    lo = t - L + 1
    if lo >= 0:
        return states[lo:t + 1]
    pad = np.repeat(states[0:1], -lo, axis=0)
    return np.concatenate([pad, states[0:t + 1]], axis=0)


def build_validity_dataset(
    per_stream: Sequence[dict],
    *,
    H: int,
    L: int,
    delta: int = 1,
    anchor_alpha: float = 0.3,
) -> ValidityDataset:
    """Assemble the dataset from per-stream monitoring output.

    Each entry of `per_stream` is
    ``{"stream_id": str, "states": DataFrame, "risks": ndarray, "eps": float,
       "forecastability": "F"|"PF"|"U"}``.

    `anchor_alpha` sets the exponential-smoothing level computed from each origin's *matured*
    history. That level is the `anchor` field: the same online quantity the naive baselines
    extrapolate. Experiment E16 uses it to ask whether the deployment state adds anything **on top
    of** the best simple extrapolator, rather than in competition with it - the mechanism Gate 1
    identified as the reason PRISM-V lost. It is computed from `risk_hist` alone, so it inherits
    that field's delay guarantee and introduces no new information.

    `delta` is the label delay. It governs `risk_hist`, the matured error series that the naive
    temporal baselines of Gate 1 consume. Those baselines must see exactly what a practitioner
    would see at the origin - no more - or Gate 1 is not a fair contest.
    """
    if not per_stream:
        raise ValueError("no streams supplied")

    feature_names: tuple[str, ...] | None = None
    S_list, yt, ev, paths, r_now, r_next, orig, sid, eps_l, fc = [], [], [], [], [], [], [], [], [], []
    r_hist: list[np.ndarray] = []
    anchors: list[float] = []

    for entry in per_stream:
        states_df: pd.DataFrame = entry["states"]
        assert_no_future_columns(states_df.columns)
        names = tuple(states_df.columns)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(
                f"stream {entry['stream_id']!r} has a different state schema; every stream must "
                "produce identical, identically ordered features"
            )

        states = states_df.to_numpy(dtype=float)
        risks = np.asarray(entry["risks"], dtype=float)
        eps = float(entry["eps"])

        for rec in build_targets(risks, eps, H=H):
            t = rec.origin
            S_list.append(_context_window(states, t, L))
            # matured risk series visible at the origin: windows 0 .. t-delta
            stop = max(t - delta + 1, 0)
            hist_slice = risks[max(stop - L, 0):stop]
            if hist_slice.size < L:
                pad_val = hist_slice[0] if hist_slice.size else np.nan
                hist_slice = np.concatenate(
                    [np.full(L - hist_slice.size, pad_val), hist_slice])
            r_hist.append(hist_slice)
            fin = hist_slice[np.isfinite(hist_slice)]
            if fin.size:
                lvl = float(fin[0])
                for v in fin[1:]:
                    lvl = anchor_alpha * v + (1.0 - anchor_alpha) * lvl
            else:
                lvl = float(risks[t]) if np.isfinite(risks[t]) else 0.5
            anchors.append(lvl)
            yt.append(rec.y_tilde)
            ev.append(rec.event)
            paths.append(rec.risk_path)
            r_now.append(risks[t])
            r_next.append(risks[t + 1] if t + 1 < risks.size else np.nan)
            orig.append(t)
            sid.append(entry["stream_id"])
            eps_l.append(eps)
            fc.append(entry.get("forecastability", "UNKNOWN"))

    if not S_list:
        raise ValueError(
            "no valid origins across any stream: the estimand is empty. Check eps (see D21) "
            "and the stream length relative to H."
        )

    return ValidityDataset(
        S=np.stack(S_list).astype(np.float32),
        y_tilde=np.asarray(yt, dtype=np.int64),
        event=np.asarray(ev, dtype=np.int64),
        risk_path=np.stack(paths).astype(np.float32),
        risk_hist=np.stack(r_hist).astype(np.float32),
        anchor=np.repeat(np.asarray(anchors, dtype=np.float32)[:, None], H, axis=1),
        risk_now=np.asarray(r_now, dtype=np.float32),
        risk_next=np.asarray(r_next, dtype=np.float32),
        origin=np.asarray(orig, dtype=np.int64),
        stream_id=np.asarray(sid, dtype=object),
        eps=np.asarray(eps_l, dtype=np.float32),
        forecastability=np.asarray(fc, dtype=object),
        feature_names=feature_names or (),
        H=H, L=L,
    )


def chronological_split(
    ds: ValidityDataset,
    *,
    train_frac: float,
    cal_frac: float,
    gap: int | None = None,
) -> tuple[ValidityDataset, ValidityDataset, ValidityDataset]:
    """Split by origin time **within each stream**, leaving a gap of `gap` windows (default `H`).

    The gap is the whole point. A training origin at window `t` is supervised by windows up to
    `t + H`; without a gap of at least `H`, those windows are also scored as calibration or test
    origins, and the two segments share outcomes. Never shuffle: this is a temporal stream.
    """
    if not 0 < train_frac < 1 or not 0 < cal_frac < 1 or train_frac + cal_frac >= 1:
        raise ValueError(f"invalid split fractions: train={train_frac}, cal={cal_frac}")
    gap = ds.H if gap is None else int(gap)

    tr_idx, ca_idx, te_idx = [], [], []
    for s in np.unique(ds.stream_id):
        where = np.flatnonzero(ds.stream_id == s)
        order = where[np.argsort(ds.origin[where])]
        origins = ds.origin[order]
        # Boundaries are placed at QUANTILES OF THE ORIGIN COUNT, not fractions of the time span.
        # Origins exist only where R_t <= eps, so on a rising-risk stream they bunch up early;
        # splitting the time span then puts almost every origin in train and leaves calibration
        # and test degenerate (measured: 373 / 2 / 14). Quantiles keep the split chronological -
        # every train origin still precedes every test origin - while making the parts usable.
        tau1 = float(np.quantile(origins, train_frac))
        tau2 = float(np.quantile(origins, train_frac + cal_frac))

        for i, t in zip(order, origins):
            if t < tau1:
                tr_idx.append(i)
            elif tau1 + gap <= t < tau2:
                ca_idx.append(i)
            elif t >= tau2 + gap:
                te_idx.append(i)
            # origins inside a gap belong to neither split and are dropped, by design

    for name, idx in (("train", tr_idx), ("calibration", ca_idx), ("test", te_idx)):
        if not idx:
            raise ValueError(
                f"{name} split is empty after applying a gap of {gap} windows. The streams are too "
                f"short for H={ds.H}: each needs comfortably more than {2 * gap} origins."
            )

    # A split with no uncensored events is not merely small - it is degenerate. Early stopping is
    # on calibration IPCW-IBS, and with zero events that score cannot discriminate between models,
    # so the "best" epoch would be arbitrary and the fitted model effectively chosen at random.
    # Fail loudly here rather than let a meaningless selection propagate into a gate.
    for name, idx in (("train", tr_idx), ("calibration", ca_idx), ("test", te_idx)):
        if len(idx) < MIN_SPLIT_ORIGINS:
            raise ValueError(
                f"{name} split has only {len(idx)} origins (minimum {MIN_SPLIT_ORIGINS}). "
                f"Early stopping and the statistical tests are not meaningful at this size. "
                f"Use more streams or longer streams."
            )
        n_ev = int(ds.event[np.asarray(idx)].sum())
        if n_ev == 0:
            raise ValueError(
                f"{name} split contains {len(idx)} origins but ZERO uncensored events. "
                f"Early stopping and every survival metric are undefined on it. Use more streams, "
                f"longer streams, or a larger {name} fraction - do not proceed with this split."
            )

    train = ds.subset(np.asarray(tr_idx))
    norm = Normaliser.fit(train.S, ds.feature_names)      # fitted on TRAIN only, then frozen
    return (train.with_normaliser(norm),
            ds.subset(np.asarray(ca_idx)).with_normaliser(norm),
            ds.subset(np.asarray(te_idx)).with_normaliser(norm))


def stream_level_split(
    ds: ValidityDataset,
    *,
    train_frac: float = 0.5,
    cal_frac: float = 0.2,
    seed: int = 0,
) -> tuple[ValidityDataset, ValidityDataset, ValidityDataset]:
    """Split by **stream**: train on some deployments, evaluate on entirely unseen ones.

    This is the setting the headline experiments use, for two reasons.

    *It is the realistic one.* A model owner fits validity dynamics on deployments already
    observed and applies the result to a new one. Nobody has the future of the stream they are
    currently trying to forecast.

    *A within-stream chronological split cannot answer the question on a monotone shift.* Train
    would cover the low-risk early regime and test the high-risk late regime, so the comparison
    measures extrapolation to an unseen risk level rather than prognosis, and every method that
    reads current risk online - persistence above all - wins for the wrong reason.

    Streams are assigned by a seeded permutation, and the assignment is recorded in the manifest.
    No temporal gap is needed here: distinct streams share no windows, so no training origin's
    horizon can overlap a test origin.
    """
    if not 0 < train_frac < 1 or not 0 < cal_frac < 1 or train_frac + cal_frac >= 1:
        raise ValueError(f"invalid split fractions: train={train_frac}, cal={cal_frac}")

    streams = np.unique(ds.stream_id)
    if streams.size < 3:
        raise ValueError(
            f"stream-level split needs at least 3 streams, got {streams.size}"
        )
    order = np.random.default_rng(seed).permutation(streams.size)
    shuffled = streams[order]
    n_tr = max(1, int(round(train_frac * streams.size)))
    n_ca = max(1, int(round(cal_frac * streams.size)))
    if n_tr + n_ca >= streams.size:
        n_ca = max(1, streams.size - n_tr - 1)

    groups = {"train": shuffled[:n_tr], "calibration": shuffled[n_tr:n_tr + n_ca],
              "test": shuffled[n_tr + n_ca:]}
    idx = {k: np.flatnonzero(np.isin(ds.stream_id, v)) for k, v in groups.items()}

    for name, ii in idx.items():
        if ii.size < MIN_SPLIT_ORIGINS:
            raise ValueError(
                f"{name} split has only {ii.size} origins from streams {list(groups[name])} "
                f"(minimum {MIN_SPLIT_ORIGINS}); use more or longer streams"
            )
        if int(ds.event[ii].sum()) == 0:
            raise ValueError(f"{name} split contains zero uncensored events")

    train = ds.subset(idx["train"])
    norm = Normaliser.fit(train.S, ds.feature_names)
    return (train.with_normaliser(norm),
            ds.subset(idx["calibration"]).with_normaliser(norm),
            ds.subset(idx["test"]).with_normaliser(norm))
