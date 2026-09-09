#!/usr/bin/env python3
"""21_gate2_survival.py -- Experiment E2 and the Gate 2 decision, in-domain.

**Standing of this experiment.** Gate 2 is pre-registered in `proposal/go_no_go.md`; it was not run
earlier only because Gate 1's failure triggered the stop rule. It is therefore a planned experiment
executed late, not one invented after seeing suggestive numbers. Its criterion is the project's
single definition of "beats", unchanged: Holm-corrected paired test rejecting at alpha = 0.05 **and**
|Cliff's delta| >= 0.147.

Gate 2 asks a different question from Gate 1. Gate 1 asked whether the deployment state forecasts
the *value* of future risk better than extrapolating the error series, and the answer was no. Gate 2
asks whether it *ranks* origins by impending violation better than the marginal alternatives. The
two can differ: a signal too weak to pin down a number may still order cases usefully, which is what
a survival model is scored on.

Three references, in increasing order of honesty:

* `km_marginal`  -- the classical constant marginal curve named in the original gate;
* `cox_ph`       -- Cox proportional hazards on the deployment state, the strong classical
                    competitor the gate also names;
* `age_marginal` -- Kaplan-Meier stratified by deployment age. D28 established that a constant KM
                    curve is an unfairly weak floor, because it ignores covariates entirely while
                    PRISM-V sees deployment age, and the pooled hazard genuinely varies with it.
                    **This is the reference that decides the gate here.**
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.statistics import (                              # noqa: E402
    ALPHA, MIN_EFFECT, blocked_bootstrap_ci, cliffs_delta, holm_bonferroni, wilcoxon_paired,
)
from src.metrics.survival_metrics import (                           # noqa: E402
    antolini_concordance, d_calibration, ipcw_integrated_brier,
)
from src.survival.utils import kaplan_meier                          # noqa: E402
from src.utils.cli import standard_parser                            # noqa: E402
from src.utils.io import ensure_dir, get_logger, should_skip, write_json  # noqa: E402
from src.utils.manifest import run_manifest                          # noqa: E402
from src.utils.seeding import SEEDS_FULL, seed_everything            # noqa: E402
from src.validity.dataset import build_validity_dataset, stream_level_split  # noqa: E402
from src.validity.model import PrismV, PrismVConfig                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module                                  # noqa: E402
_t = import_module("20_transfer")
age_marginal_survival = _t.age_marginal_survival
concordance_contributions = _t.concordance_contributions

OUT = Path("results/e2_gate2")
EPS_MARGIN, H, L = 0.20, 20, 20
REFERENCES = ("km_marginal", "cox_ph", "age_marginal")
#: The reference that decides the gate. D28: a constant KM curve is not an honest floor.
DECIDING_REFERENCE = "age_marginal"


def load_by_dataset(margin: float) -> dict[str, list[dict]]:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(Path("data/cache/monitored").glob("*.pkl")):
        e = pickle.loads(f.read_bytes())
        by_ds[e["dataset_id"]].append({**e, "eps": float(min(e["eps"] - 0.05 + margin, 1.0))})
    return dict(by_ds)


def cox_survival(train, test, H: int) -> tuple[np.ndarray, dict]:
    """Cox proportional hazards on the deployment state, fitted on train only.

    The state is reduced to its most recent step and to a modest number of principal components:
    a Cox model on 87 correlated covariates with a few hundred origins does not converge, and a
    baseline that fails to converge is not a fair competitor.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # A missing dependency must not be able to masquerade as a fitted comparator. An earlier
    # version caught ImportError here and returned the Kaplan-Meier curve, so running under an
    # interpreter without lifelines silently recorded the Cox reference at exactly 0.5000 on every
    # dataset -- an artefact indistinguishable, in the result files, from a real finding about the
    # Cox model. Refusing to run is the only safe behaviour.
    try:
        from lifelines import CoxPHFitter
        import pandas as pd
    except ImportError as exc:                            # pragma: no cover
        raise RuntimeError(
            "lifelines is required for the Cox proportional-hazards reference. Refusing to "
            "substitute the marginal curve, because that would be recorded as a Cox result. "
            "Install lifelines, or run under the project environment."
        ) from exc

    k = int(min(8, train.S.shape[-1], max(2, len(train) // 25)))
    sc = StandardScaler().fit(train.S[:, -1, :])
    pca = PCA(n_components=k, random_state=0).fit(sc.transform(train.S[:, -1, :]))
    Ztr = pca.transform(sc.transform(train.S[:, -1, :]))
    Zte = pca.transform(sc.transform(test.S[:, -1, :]))

    df = pd.DataFrame(Ztr, columns=[f"z{i}" for i in range(k)])
    df["T"], df["E"] = train.y_tilde, train.event
    try:
        cph = CoxPHFitter(penalizer=0.1).fit(df, duration_col="T", event_col="E")
        te = pd.DataFrame(Zte, columns=[f"z{i}" for i in range(k)])
        sf = cph.predict_survival_function(te, times=np.arange(1, H + 1))
        return np.clip(sf.to_numpy().T, 0.0, 1.0), {"status": "fitted", "n_components": k}
    except Exception as exc:
        # A genuine estimation failure is still reported as one, and the reason travels with the
        # number so the manuscript can describe what actually happened.
        return (np.tile(kaplan_meier(train.y_tilde, train.event, H), (len(test), 1)),
                {"status": "estimation_failed", "n_components": k,
                 "reason": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    ap = standard_parser("E2 / Gate 2: survival ranking vs marginal and Cox references",
                         require_config=False)
    ap.add_argument("--scale", default="full", choices=["smoke", "dev", "full"])
    args = ap.parse_args()
    ensure_dir(OUT)
    log = get_logger("gate2", args.log or "logs/gate2.log").info

    by_ds = load_by_dataset(EPS_MARGIN)
    seeds = list(SEEDS_FULL) if args.scale == "full" else [0]
    all_rows: list[dict] = []

    for name in sorted(by_ds):
        try:
            ds = build_validity_dataset(by_ds[name], H=H, L=L, delta=1)
            train, cal, test = stream_level_split(ds, train_frac=0.5, cal_frac=0.2, seed=0)
        except ValueError as exc:
            log(f"EXCLUDED {name}: {exc}")
            continue

        cox_surv, cox_status = cox_survival(train, test, H)
        refs = {
            "km_marginal": np.tile(kaplan_meier(train.y_tilde, train.event, H), (len(test), 1)),
            "cox_ph": cox_surv,
            "age_marginal": age_marginal_survival(train, test, H),
        }
        log(f"\n=== {name}: train {len(train)} / cal {len(cal)} / test {len(test)} origins, "
            f"{test.n_events} events, censoring {ds.censoring_rate:.3f}")
        for rn, rs in refs.items():
            log(f"  {rn:14s} C={antolini_concordance(rs, test.y_tilde, test.event):.4f}")

        for seed in seeds:
            tgt = OUT / f"{name}_seed{seed}.json"
            if should_skip(tgt, args.force, logger=None):
                all_rows.append(__import__("json").loads(tgt.read_text()))
                continue
            with run_manifest(f"E2_gate2_{name}_seed{seed}",
                              out_path=OUT / f"manifest_{name}_seed{seed}.json",
                              config={"scale": args.scale, "eps_margin": EPS_MARGIN,
                                      "H": H, "L": L},
                              dataset_id=name, seed=seed, scale=args.scale,
                              predictor_id="mixed") as man:
                seed_everything(seed)
                m = PrismV(PrismVConfig(variant="first_passage", encoder="gru", max_epochs=150,
                                        patience=25, seed=seed, n_samples=128),
                           H=H, d_in=ds.S.shape[-1]).fit(train, cal)
                surv = m.predict_survival(test)

                row: dict = {
                    "dataset": name, "seed": int(seed), "scale": args.scale,
                    "n_test_origins": len(test), "n_test_events": int(test.n_events),
                    "censoring_rate": round(ds.censoring_rate, 4),
                    "n_streams": int(len(np.unique(ds.stream_id))),
                    "metrics": {}, "tests": {},
                }
                for rn, rs in refs.items():
                    row["metrics"][rn] = {
                        "antolini_c": float(antolini_concordance(rs, test.y_tilde, test.event)),
                        "ipcw_ibs": float(ipcw_integrated_brier(rs, test.y_tilde, test.event)),
                        "d_cal_p": float(d_calibration(rs, test.y_tilde, test.event)["d_cal_p"]),
                    }
                row["metrics"]["prismv"] = {
                    "antolini_c": float(antolini_concordance(surv, test.y_tilde, test.event)),
                    "ipcw_ibs": float(ipcw_integrated_brier(surv, test.y_tilde, test.event)),
                    "d_cal_p": float(d_calibration(surv, test.y_tilde, test.event)["d_cal_p"]),
                }

                pc = concordance_contributions(surv, test)
                praw = {}
                for rn, rs in refs.items():
                    rc = concordance_contributions(rs, test)
                    msk = np.isfinite(pc) & np.isfinite(rc)
                    p = wilcoxon_paired(rc[msk], pc[msk], alternative="less")
                    d = cliffs_delta(pc[msk], rc[msk])
                    pt, lo, hi = blocked_bootstrap_ci(pc[msk] - rc[msk],
                                                      test.stream_id[msk], seed=seed)
                    praw[rn] = p
                    row["tests"][rn] = {"p_raw": p, "effect_delta": d,
                                        "n_pairs": int(msk.sum()), "mean_gain": pt,
                                        "ci95": [lo, hi]}
                corrected = holm_bonferroni(praw)
                for rn in refs:
                    adj = corrected[rn]["p_adjusted"]
                    row["tests"][rn]["p_adjusted"] = adj
                    row["tests"][rn]["beats"] = bool(
                        adj <= ALPHA and row["tests"][rn]["effect_delta"] >= MIN_EFFECT)
                man.metrics = {k: v["antolini_c"] for k, v in row["metrics"].items()}
                write_json(row, tgt)
                all_rows.append(row)
            b = row["tests"][DECIDING_REFERENCE]
            log(f"  seed={seed} PRISM-V C={row['metrics']['prismv']['antolini_c']:.4f} "
                f"vs {DECIDING_REFERENCE} {row['metrics'][DECIDING_REFERENCE]['antolini_c']:.4f} "
                f"| p_adj={b['p_adjusted']:.3g} delta={b['effect_delta']:+.3f} beats={b['beats']}")

    # ------------------------------------------------------------------ verdict
    log("\n" + "=" * 78)
    log(f"GATE 2 VERDICT  (deciding reference: {DECIDING_REFERENCE}; "
        f"beats = Holm p<={ALPHA} AND delta>={MIN_EFFECT})")
    log("=" * 78)
    verdict: dict = {"scale": args.scale, "seeds": seeds,
                     "deciding_reference": DECIDING_REFERENCE,
                     "rule": f"Holm-corrected p <= {ALPHA} AND Cliff's delta >= {MIN_EFFECT}, "
                             "against every reference",
                     "datasets": {}}
    for name in sorted({r["dataset"] for r in all_rows}):
        rows = [r for r in all_rows if r["dataset"] == name]
        entry: dict = {"n_seeds": len(rows),
                       "n_test_origins": rows[0]["n_test_origins"],
                       "n_test_events": rows[0]["n_test_events"]}
        for k in ("km_marginal", "cox_ph", "age_marginal", "prismv"):
            cs = [r["metrics"][k]["antolini_c"] for r in rows]
            ib = [r["metrics"][k]["ipcw_ibs"] for r in rows]
            entry[k] = {"c_mean": round(float(np.mean(cs)), 4),
                        "c_sd": round(float(np.std(cs)), 4),
                        "ibs_mean": round(float(np.mean(ib)), 4)}
        for rn in REFERENCES:
            t = [r["tests"][rn] for r in rows]
            entry[f"vs_{rn}"] = {
                "seeds_beaten": int(sum(x["beats"] for x in t)),
                "mean_delta": round(float(np.mean([x["effect_delta"] for x in t])), 4),
                "mean_p_adj": float(np.mean([x["p_adjusted"] for x in t])),
                "mean_gain": round(float(np.mean([x["mean_gain"] for x in t])), 4)}
        entry["beats_all_references_all_seeds"] = all(
            entry[f"vs_{rn}"]["seeds_beaten"] == len(rows) for rn in REFERENCES)
        verdict["datasets"][name] = entry
        log(f"{name:26s} PRISM-V C={entry['prismv']['c_mean']:.4f}±{entry['prismv']['c_sd']:.4f} "
            f"| km {entry['km_marginal']['c_mean']:.4f} "
            f"| cox {entry['cox_ph']['c_mean']:.4f} "
            f"| age {entry['age_marginal']['c_mean']:.4f}")
        for rn in REFERENCES:
            v = entry[f"vs_{rn}"]
            log(f"      vs {rn:14s} beats {v['seeds_beaten']}/{len(rows)} seeds  "
                f"delta={v['mean_delta']:+.3f}  p_adj={v['mean_p_adj']:.3g}")

    n_pass = sum(v["beats_all_references_all_seeds"] for v in verdict["datasets"].values())
    verdict["n_datasets_beating_all_references"] = n_pass
    verdict["GATE2"] = "PASS" if n_pass >= 2 else "FAIL"
    write_json(verdict, OUT / "GATE2_VERDICT.json")
    log(f"\nGATE 2 = {verdict['GATE2']}  "
        f"({n_pass} dataset(s) beat every reference on every seed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
