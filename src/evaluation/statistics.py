"""The statistical protocol, fixed in advance.

`experimental_protocol.md` section 7 and `go_no_go.md` fix every choice here before any experiment
runs: paired Wilcoxon signed-rank across matched streams, Holm-Bonferroni correction within an
experiment family, effect sizes reported alongside every p-value, and bootstrap confidence
intervals blocked by stream so temporal dependence is respected.

"Beats" has one definition in this project, encoded once in `beats`:

    the Holm-corrected paired test rejects at alpha = 0.05  AND  |Cliff's delta| >= 0.147

Both conditions. A significant result with a negligible effect does not count, and neither does a
large effect that fails its test.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "ALPHA", "MIN_EFFECT", "cliffs_delta", "paired_cohens_d", "wilcoxon_paired",
    "holm_bonferroni", "blocked_bootstrap_ci", "ComparisonResult", "compare", "beats",
]

#: Pre-set significance level and the small-effect threshold, both from go_no_go.md.
ALPHA = 0.05
MIN_EFFECT = 0.147


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a > b) - P(a < b), in [-1, 1]. Non-parametric, no distributional assumption.

    Computed by rank arithmetic rather than the O(nm) pairwise loop, which matters because this runs
    across thousands of origins in every ablation cell.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return float("nan")
    combined = np.concatenate([a, b])
    ranks = stats.rankdata(combined)
    r_a = ranks[:n].sum()
    # Mann-Whitney U for `a`, converted to the dominance statistic
    u_a = r_a - n * (n + 1) / 2.0
    return float(2.0 * u_a / (n * m) - 1.0)


def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d on the differences. Reported where the design is genuinely paired."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_cohens_d requires equal-length paired samples")
    d = a - b
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def wilcoxon_paired(a: np.ndarray, b: np.ndarray, alternative: str = "less") -> float:
    """Paired Wilcoxon signed-rank p-value.

    Default `alternative="less"` tests that `a` is stochastically smaller than `b` — the natural
    direction when both are error measures and `a` is the proposed method.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("wilcoxon_paired requires equal-length paired samples")
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 5 or np.allclose(a, b):
        return 1.0                        # too few pairs, or no difference at all
    try:
        return float(stats.wilcoxon(a, b, alternative=alternative,
                                    zero_method="zsplit").pvalue)
    except ValueError:
        return 1.0


def holm_bonferroni(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction within one experiment family.

    Returns, per key, the raw p, the adjusted p, and whether it is rejected. Applied *within* a
    family (e.g. all six Gate-1 baselines at one horizon), never across the whole paper.
    """
    if not pvalues:
        return {}
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    n = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (n - i) * p))     # enforce monotone adjusted p-values
        running = adj
        out[k] = {"p_raw": float(p), "p_adjusted": float(adj), "rejected": bool(adj <= alpha)}
    return out


def blocked_bootstrap_ci(values: np.ndarray, blocks: np.ndarray, *, n_boot: int = 2000,
                         level: float = 0.95, seed: int = 0,
                         statistic=np.mean) -> tuple[float, float, float]:
    """Bootstrap CI resampling **whole blocks** (streams), not individual origins.

    Origins within a stream are temporally dependent; resampling them independently would shrink
    the interval to a width the data does not support. Resampling streams respects that dependence.
    Returns `(point, lo, hi)`.
    """
    values = np.asarray(values, dtype=float)
    blocks = np.asarray(blocks)
    m = np.isfinite(values)
    values, blocks = values[m], blocks[m]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")

    unique = np.unique(blocks)
    grouped = [values[blocks == b] for b in unique]
    rng = np.random.default_rng(seed)
    point = float(statistic(values))
    if unique.size < 2:
        return point, float("nan"), float("nan")

    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, unique.size, size=unique.size)
        draws[i] = statistic(np.concatenate([grouped[j] for j in pick]))
    lo = float(np.percentile(draws, 100 * (1 - level) / 2))
    hi = float(np.percentile(draws, 100 * (1 + level) / 2))
    return point, lo, hi


@dataclass(frozen=True)
class ComparisonResult:
    """One method-versus-baseline comparison, with everything the paper must report."""

    method: str
    baseline: str
    p_raw: float
    effect_delta: float
    effect_d: float
    method_mean: float
    baseline_mean: float
    n_pairs: int
    p_adjusted: float | None = None

    @property
    def wins(self) -> bool:
        """The project's single definition of 'beats' — corrected test AND effect size."""
        if self.p_adjusted is None:
            return False
        return bool(self.p_adjusted <= ALPHA and abs(self.effect_delta) >= MIN_EFFECT)

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("method", "baseline", "p_raw", "p_adjusted", "effect_delta", "effect_d",
              "method_mean", "baseline_mean", "n_pairs")}
        d["wins"] = self.wins
        return d


def compare(method_errors: np.ndarray, baseline_errors: np.ndarray,
            method: str, baseline: str) -> ComparisonResult:
    """Paired comparison of two error samples. Lower is better for both."""
    a = np.asarray(method_errors, dtype=float)
    b = np.asarray(baseline_errors, dtype=float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    return ComparisonResult(
        method=method, baseline=baseline,
        p_raw=wilcoxon_paired(a, b, alternative="less"),
        effect_delta=cliffs_delta(a, b),
        effect_d=paired_cohens_d(a, b),
        method_mean=float(np.nanmean(a)) if n else float("nan"),
        baseline_mean=float(np.nanmean(b)) if n else float("nan"),
        n_pairs=int(n),
    )


def beats(results: list[ComparisonResult], alpha: float = ALPHA) -> list[ComparisonResult]:
    """Apply Holm correction across a family and return the results with `p_adjusted` filled in.

    The family is the set of baselines a method is compared against within one experiment, which is
    the level `experimental_protocol.md` section 7 specifies. Correcting across the whole paper
    instead would be over-conservative; not correcting at all would inflate the false-positive rate
    across six simultaneous Gate-1 comparisons.
    """
    corrected = holm_bonferroni({r.baseline: r.p_raw for r in results}, alpha=alpha)
    return [dataclasses.replace(r, p_adjusted=corrected[r.baseline]["p_adjusted"])
            for r in results]
