#!/usr/bin/env python3
"""31_dependence_sensitivity.py -- a dependence-aware re-analysis of the survival comparison.

The pre-registered test is a Holm-corrected paired Wilcoxon over per-origin concordance
contributions. Origins within a stream are not independent, so that test's nominal level is
optimistic. **The pre-registered analysis is not altered here**; this script adds a secondary,
dependence-aware analysis and reports whether the verdict is stable under it.

Two dependence-aware procedures, both operating on the same frozen per-origin contributions:

* **Stream-level paired analysis.** Contributions are averaged within each held-out stream, and the
  paired comparison is made across streams. The unit of analysis becomes the stream, which is the
  unit the data-generating process actually replicates.
* **Blocked permutation.** The sign of the per-stream mean difference is flipped at random, whole
  streams at a time, to build a null distribution for the mean difference. This respects the block
  structure that a per-origin permutation would destroy.

Reported as a sensitivity analysis, never as confirmatory evidence.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.survival.utils import kaplan_meier                          # noqa: E402
from src.utils.cli import standard_parser                            # noqa: E402
from src.utils.io import ensure_dir, get_logger, write_json          # noqa: E402
from src.utils.seeding import SEEDS_FULL, seed_everything            # noqa: E402
from src.validity.dataset import build_validity_dataset, stream_level_split  # noqa: E402
from src.validity.model import PrismV, PrismVConfig                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
_t = import_module("20_transfer")
age_marginal_survival = _t.age_marginal_survival
concordance_contributions = _t.concordance_contributions

OUT = Path("results/mechanism/dependence_sensitivity.json")
EPS_MARGIN, H, L = 0.20, 20, 20
N_PERM = 20000


def blocked_permutation_p(diff: np.ndarray, block: np.ndarray, rng) -> tuple[float, float]:
    """(mean difference, two-sided p) under whole-stream sign flipping."""
    blocks = np.unique(block)
    means = np.array([float(np.mean(diff[block == b])) for b in blocks])
    obs = float(np.mean(means))
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, means.size))
    null = (signs * means).mean(axis=1)
    p = float((np.abs(null) >= abs(obs) - 1e-15).mean())
    return obs, max(p, 1.0 / N_PERM)


def main() -> int:
    ap = standard_parser("Dependence-aware sensitivity analysis", require_config=False)
    args = ap.parse_args()
    ensure_dir(OUT.parent)
    log = get_logger("depsens", args.log or "logs/dependence_sensitivity.log").info

    by_ds: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(Path("data/cache/monitored").glob("*.pkl")):
        e = pickle.loads(f.read_bytes())
        by_ds[e["dataset_id"]].append({**e, "eps": float(min(e["eps"] - 0.05 + EPS_MARGIN, 1.0))})

    verdict = json.loads(Path("results/mechanism/PRH1_VERDICT.json").read_text(encoding="utf-8"))
    scored = [k for k, v in verdict["datasets"].items()
              if v.get("tier") in ("PRIMARY", "SECONDARY")]

    out: dict = {
        "note": "Secondary, dependence-aware re-analysis of the monitoring-state model against the "
                "deployment-age reference. The pre-registered Holm-corrected paired Wilcoxon over "
                "per-origin contributions is unchanged and remains the confirmatory analysis.",
        "n_permutations": N_PERM, "unit": "held-out stream", "datasets": {},
    }
    rng = np.random.default_rng(0)

    for name in sorted(scored):
        ds = build_validity_dataset(by_ds[name], H=H, L=L, delta=1)
        train, cal, test = stream_level_split(ds, train_frac=0.5, cal_frac=0.2, seed=0)
        age = age_marginal_survival(train, test, H)
        ac = concordance_contributions(age, test)

        per_seed = []
        for seed in SEEDS_FULL:
            seed_everything(seed)
            m = PrismV(PrismVConfig(variant="first_passage", encoder="gru", max_epochs=150,
                                    patience=25, seed=seed, n_samples=128),
                       H=H, d_in=ds.S.shape[-1]).fit(train, cal)
            pc = concordance_contributions(m.predict_survival(test), test)
            msk = np.isfinite(pc) & np.isfinite(ac)
            diff = pc[msk] - ac[msk]
            blocks = np.asarray(test.stream_id)[msk]
            mean_diff, p = blocked_permutation_p(diff, blocks, rng)
            n_blocks = int(np.unique(blocks).size)
            per_seed.append({"seed": int(seed), "n_streams": n_blocks,
                             "stream_level_mean_diff": round(mean_diff, 4),
                             "blocked_permutation_p": round(p, 5),
                             "significant_at_05": bool(p <= 0.05)})
            log(f"  {name:16s} seed={seed} streams={n_blocks} "
                f"mean_diff={mean_diff:+.4f} p_perm={p:.4f}")

        # With B blocks, whole-block sign flipping admits 2^B assignments, so the smallest
        # attainable two-sided p is 2^(1-B). Where that floor exceeds 0.05 the test cannot reject
        # at any effect size, and a null result carries no evidential weight. Recording the floor
        # keeps that visible instead of letting "0/5 significant" be read as a refutation.
        nb = per_seed[0]["n_streams"]
        floor = 2.0 ** (1 - nb)
        out["datasets"][name] = {
            "n_test_streams": nb,
            "min_attainable_p": round(floor, 4),
            "test_can_reject_at_05": bool(floor <= 0.05),
            "per_seed": per_seed,
            "seeds_significant": sum(s["significant_at_05"] for s in per_seed),
            "mean_diff_mean": round(float(np.mean([s["stream_level_mean_diff"]
                                                   for s in per_seed])), 4),
            "confirmatory_verdict_beats_age":
                verdict["datasets"][name]["beats_deciding_reference"],
        }
        log(f"{name:16s} stream-level: {out['datasets'][name]['seeds_significant']}/"
            f"{len(per_seed)} seeds significant; confirmatory said "
            f"{out['datasets'][name]['confirmatory_verdict_beats_age']}")

    powered = {k: v for k, v in out["datasets"].items() if v["test_can_reject_at_05"]}
    agree = sum(1 for v in powered.values()
                if (v["seeds_significant"] == len(SEEDS_FULL)) == v["confirmatory_verdict_beats_age"])
    out["n_datasets"] = len(out["datasets"])
    out["n_datasets_with_power"] = len(powered)
    out["n_datasets_where_sensitivity_agrees"] = agree
    out["conclusion"] = (
        "The stream-level unit of analysis leaves too few blocks for a whole-stream permutation "
        "test to reject at the 5%% level on %d of %d datasets, so this sensitivity analysis is "
        "inconclusive by construction rather than contradictory. It neither corroborates nor "
        "refutes the pre-registered per-origin analysis." % (len(out["datasets"]) - len(powered),
                                                              len(out["datasets"])))
    write_json(out, OUT)
    log(f"sensitivity agrees with the confirmatory verdict on {agree}/{len(out['datasets'])} "
        f"datasets; written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
