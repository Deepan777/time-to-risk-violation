"""Construction of the PRISM-V learning target.

The estimand (see proposal/proposed_methodology.md, eq. 3-4):

    T_t = min{ h >= 1 : R_{t+h} > eps }     defined only for origins with R_t <= eps

right-censored at C_t = min(H, len(R) - 1 - t). Each valid origin yields
(Y_tilde, event) with Y_tilde = min(T_t, C_t) and event = 1[T_t <= C_t].

This module is deliberately dependency-light and pure: it is the one piece of the pipeline whose
correctness the whole project rests on, so it is unit-tested against hand-built fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = ["TargetRecord", "build_targets", "censoring_rate"]


@dataclass(frozen=True)
class TargetRecord:
    origin: int          # index t of the origin window
    y_tilde: int         # observed time, in windows, 1..H
    event: int           # 1 = violation observed, 0 = right-censored
    risk_path: np.ndarray  # realised R_{t+1..t+H}, NaN-padded past the end of the stream


def build_targets(risk: np.ndarray, eps: float, H: int) -> list[TargetRecord]:
    """Build one censored record per *valid* origin.

    Parameters
    ----------
    risk : 1-D array of realised window risks R_0..R_{T-1}. NaN means "not yet matured";
           an origin whose own risk is NaN is skipped, and a NaN inside the horizon terminates
           the scan (we cannot claim a violation we did not observe).
    eps  : the validity threshold. Violation is strict: R > eps.
    H    : horizon budget in windows, H >= 1.

    Returns
    -------
    A list of TargetRecord, ordered by origin. Origins with R_t > eps are excluded by
    definition of the estimand, as are origins with no remaining horizon.
    """
    if H < 1:
        raise ValueError("H must be >= 1")
    risk = np.asarray(risk, dtype=float)
    if risk.ndim != 1:
        raise ValueError("risk must be 1-D")

    n = risk.size
    out: list[TargetRecord] = []
    for t in range(n - 1):
        r_t = risk[t]
        if not np.isfinite(r_t) or r_t > eps:      # origin not observed, or already invalid
            continue
        c_t = min(H, n - 1 - t)                    # windows actually observable ahead
        if c_t < 1:
            continue
        path = np.full(H, np.nan)
        path[:c_t] = risk[t + 1: t + 1 + c_t]

        y_tilde, event = c_t, 0
        for h in range(1, c_t + 1):
            r = risk[t + h]
            if not np.isfinite(r):                 # horizon interrupted by an unmatured window
                y_tilde, event = h - 1, 0
                break
            if r > eps:
                y_tilde, event = h, 1
                break
        if y_tilde < 1:                            # nothing observable after the origin
            continue
        out.append(TargetRecord(origin=t, y_tilde=int(y_tilde), event=int(event), risk_path=path))
    return out


def censoring_rate(records: list[TargetRecord]) -> float:
    """Fraction of records that are right-censored. Reported with every horizon metric."""
    if not records:
        raise ValueError("no records: cannot report a censoring rate")
    return float(sum(1 for r in records if r.event == 0) / len(records))
