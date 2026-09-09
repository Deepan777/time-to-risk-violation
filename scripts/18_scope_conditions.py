#!/usr/bin/env python3
"""18_scope_conditions.py -- measure the evidence behind the excluded datasets.

Section 6.5 of the manuscript reports two datasets that were acquired, monitored and then excluded
because the estimand is not defined on them. Those claims carry numbers, and every number in the
manuscript must come from a result file rather than from a sentence someone typed. This script
produces that file.

Writes results/scope_conditions/huffpost_exclusion.json.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.wildtime import load_huffpost          # noqa: E402
from src.predictors.base import fit_on_pre_deployment    # noqa: E402
from src.utils.cli import standard_parser                # noqa: E402
from src.utils.io import ensure_dir, get_logger, should_skip, write_json  # noqa: E402
from src.utils.manifest import run_manifest              # noqa: E402
from src.validity.targets import build_targets           # noqa: E402

OUT = Path("results/scope_conditions")
EPS_MARGIN, H = 0.20, 20


def main() -> int:
    ap = standard_parser("Measure the scope conditions behind the excluded datasets",
                         require_config=False)
    args = ap.parse_args()
    ensure_dir(OUT)
    log = get_logger("scope", args.log or "logs/scope.log").info
    target = OUT / "huffpost_exclusion.json"
    if should_skip(target, args.force, logger=None):
        log(f"{target} exists; pass --force to recompute")
        return 0

    with run_manifest("scope_huffpost", out_path=OUT / "manifest_huffpost.json",
                      config={"scale": "full", "eps_margin": EPS_MARGIN},
                      dataset_id="huffpost", seed=0, scale="full",
                      predictor_id="xgboost") as man:
        st = load_huffpost()
        y, yr = st.y(), st.df["year"].to_numpy()
        years = sorted(np.unique(yr).tolist())

        # class prevalence per year: the mechanism behind the collapse
        prev = {int(u): (np.bincount(y[yr == u], minlength=int(y.max()) + 1)
                         / max(int((yr == u).sum()), 1)).round(4).tolist() for u in years}
        first, last = prev[years[0]], prev[years[-1]]
        drops = [(i, first[i], last[i]) for i in range(len(first))]
        biggest_fall = max(drops, key=lambda d: d[1] - d[2])
        biggest_rise = max(drops, key=lambda d: d[2] - d[1])

        handle, pre, dep = fit_on_pre_deployment(
            st, family="xgboost", seed=0, pre_deployment_frac=0.20, window_size=300)
        acc_pre = float(1.0 - handle.per_sample_loss(pre.X(), pre.y()).mean())
        acc_dep = float(1.0 - handle.per_sample_loss(dep.X(), dep.y()).mean())

        # how much of deployment is valid, at the pre-committed margin
        cached = sorted(Path("data/cache/monitored").glob("huffpost*.pkl"))
        valid_frac, n_origins = [], []
        for f in cached:
            e = pickle.loads(f.read_bytes())
            eps = min(e["eps"] - 0.05 + EPS_MARGIN, 1.0)
            r = np.asarray(e["risks"], dtype=float)
            valid_frac.append(float((r <= eps).mean()))
            n_origins.append(len(build_targets(r, eps, H=H)))

        res = {
            "dataset_id": "huffpost",
            "scale": "full",
            "n_rows": int(st.n_rows),
            "n_classes": int(len(np.unique(y))),
            "years": years,
            "accuracy_pre_deployment": round(acc_pre, 4),
            "accuracy_deployment": round(acc_dep, 4),
            "accuracy_drop_points": round(100.0 * (acc_pre - acc_dep), 1),
            "chance_accuracy": round(1.0 / len(np.unique(y)), 4),
            "class_prevalence_by_year": prev,
            "largest_prevalence_fall": {
                "class": int(biggest_fall[0]),
                "from_pct": round(100.0 * biggest_fall[1], 1),
                "to_pct": round(100.0 * biggest_fall[2], 1)},
            "largest_prevalence_rise": {
                "class": int(biggest_rise[0]),
                "from_pct": round(100.0 * biggest_rise[1], 1),
                "to_pct": round(100.0 * biggest_rise[2], 1)},
            "mean_valid_window_fraction": round(float(np.mean(valid_frac)), 4),
            "mean_invalid_window_pct": round(100.0 * (1.0 - float(np.mean(valid_frac))), 1),
            "origins_per_stream": n_origins,
            "mean_origins_per_stream": round(float(np.mean(n_origins)), 1),
            "eps_margin": EPS_MARGIN,
            "exclusion_reason":
                "frozen predictor invalid for the great majority of deployment windows; too few "
                "valid origins per stream to support a chronological split with a gap of H",
        }
        man.metrics = {"accuracy_drop_points": res["accuracy_drop_points"],
                       "mean_invalid_window_pct": res["mean_invalid_window_pct"]}
        write_json(res, target)

    log(f"pre-deployment accuracy {res['accuracy_pre_deployment']:.3f} -> "
        f"deployment {res['accuracy_deployment']:.3f} "
        f"(drop {res['accuracy_drop_points']} points, chance {res['chance_accuracy']:.3f})")
    log(f"invalid for {res['mean_invalid_window_pct']}% of deployment windows; "
        f"{res['mean_origins_per_stream']} valid origins per stream")
    log(f"class {res['largest_prevalence_fall']['class']}: "
        f"{res['largest_prevalence_fall']['from_pct']}% -> "
        f"{res['largest_prevalence_fall']['to_pct']}%; class "
        f"{res['largest_prevalence_rise']['class']}: "
        f"{res['largest_prevalence_rise']['from_pct']}% -> "
        f"{res['largest_prevalence_rise']['to_pct']}%")
    log(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
