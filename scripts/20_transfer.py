#!/usr/bin/env python3
"""20_transfer.py -- Experiments E9/E10 and Gate 5: do validity dynamics transfer?

This asks a question Gate 1 did not. Gate 1 asked whether the deployment state forecasts the risk
path *within* a deployment, and the answer was no. Transfer asks whether the relationship between
deployment state and impending violation, learned on some domains and architectures, carries to a
domain the model has never seen. The two can come apart: a signal too weak to beat an online
smoother at point forecasting may still rank origins usefully, and ranking is what a survival model
is scored on.

Gate 5 rule (`proposal/go_no_go.md`): transfer performance must be **strictly above the marginal
baseline**. It does not need to match in-domain training.

**Two lower references, not one.** D28 established that a plain Kaplan-Meier curve is an unfairly
weak reference here, because it ignores covariates entirely while PRISM-V can see deployment age,
and pooled hazard genuinely varies with age. So transfer is judged against:

* `km_marginal`     -- the classical constant marginal curve, reported for continuity; and
* `age_marginal`    -- a hazard model given **deployment age alone**, which is the honest floor.
                       Only skill in excess of this can be attributed to the deployment state.

Leave-one-domain-out over the usable dataset families, five seeds, with the same statistical
protocol as every other experiment.
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.statistics import (                              # noqa: E402
    ALPHA, blocked_bootstrap_ci, cliffs_delta, wilcoxon_paired,
)
from src.metrics.survival_metrics import (                           # noqa: E402
    antolini_concordance, ipcw_integrated_brier,
)
from src.survival.utils import kaplan_meier                          # noqa: E402
from src.utils.cli import standard_parser                            # noqa: E402
from src.utils.io import ensure_dir, get_logger, should_skip, write_json  # noqa: E402
from src.utils.manifest import run_manifest                          # noqa: E402
from src.utils.seeding import SEEDS_FULL, seed_everything            # noqa: E402
from src.validity.dataset import build_validity_dataset, stream_level_split  # noqa: E402
from src.validity.model import PrismV, PrismVConfig                  # noqa: E402

OUT = Path("results/e9e10_transfer")
EPS_MARGIN, H, L = 0.20, 20, 20
AGE_FEATURE = "ctx_deployment_age"


def load_by_dataset(margin: float) -> dict[str, list[dict]]:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(Path("data/cache/monitored").glob("*.pkl")):
        e = pickle.loads(f.read_bytes())
        e = {**e, "eps": float(min(e["eps"] - 0.05 + margin, 1.0))}
        by_ds[e["dataset_id"]].append(e)
    return dict(by_ds)


def age_marginal_survival(train, test, H: int, n_bins: int = 5) -> np.ndarray:
    """Kaplan-Meier stratified by deployment age -- the honest lower reference (D28).

    Age bins are cut on the TRAINING split only and then applied to test, like every other
    reference statistic in the project. A test origin whose age falls outside the training range is
    assigned the nearest bin rather than dropped.
    """
    def age_of(ds):
        idx = list(ds.feature_names).index(AGE_FEATURE)
        return ds.S[:, -1, idx]            # most recent step of the context window

    tr_age, te_age = age_of(train), age_of(test)
    edges = np.quantile(tr_age, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    tr_bin = np.clip(np.digitize(tr_age, edges[1:-1]), 0, n_bins - 1)
    te_bin = np.clip(np.digitize(te_age, edges[1:-1]), 0, n_bins - 1)

    curves = []
    for b in range(n_bins):
        m = tr_bin == b
        if m.sum() >= 10 and train.event[m].sum() > 0:
            curves.append(kaplan_meier(train.y_tilde[m], train.event[m], H))
        else:                              # too few origins in this bin: fall back to the pooled curve
            curves.append(kaplan_meier(train.y_tilde, train.event, H))
    return np.stack([curves[b] for b in te_bin], axis=0)


def evaluate(surv: np.ndarray, ds) -> dict:
    return {"antolini_c": float(antolini_concordance(surv, ds.y_tilde, ds.event)),
            "ipcw_ibs": float(ipcw_integrated_brier(surv, ds.y_tilde, ds.event))}


def concordance_contributions(surv: np.ndarray, ds) -> np.ndarray:
    """Per-origin concordance contribution, for paired testing against a reference.

    For each origin with an observed event, the fraction of later-surviving origins it is correctly
    ordered against. Averaging these reproduces the Antolini statistic, and the per-origin values
    are what the paired Wilcoxon test consumes.
    """
    S = np.clip(surv, 0.0, 1.0)
    out = []
    for i in range(len(ds.y_tilde)):
        if ds.event[i] != 1:
            out.append(np.nan)
            continue
        h = ds.y_tilde[i] - 1
        others = np.flatnonzero(ds.y_tilde > ds.y_tilde[i])
        if others.size == 0:
            out.append(np.nan)
            continue
        si, sj = S[i, h], S[others, h]
        out.append(float((np.sum(si < sj) + 0.5 * np.sum(si == sj)) / others.size))
    return np.asarray(out, dtype=float)


def main() -> int:
    ap = standard_parser("E9/E10 + Gate 5: cross-domain transfer of validity dynamics",
                         require_config=False)
    ap.add_argument("--scale", default="full", choices=["smoke", "dev", "full"])
    args = ap.parse_args()
    ensure_dir(OUT)
    log = get_logger("transfer", args.log or "logs/transfer.log").info

    by_ds = load_by_dataset(EPS_MARGIN)
    usable: dict[str, list[dict]] = {}
    for name, entries in by_ds.items():
        try:
            probe = build_validity_dataset(entries, H=H, L=L, delta=1)
            if len(probe) >= 60 and probe.n_events > 0:
                usable[name] = entries
            else:
                log(f"EXCLUDED {name}: only {len(probe)} origins")
        except ValueError as exc:
            log(f"EXCLUDED {name}: {exc}")
    log(f"usable domains: {sorted(usable)}")
    if len(usable) < 2:
        raise SystemExit("need at least two usable domains for leave-one-domain-out")

    seeds = list(SEEDS_FULL) if args.scale == "full" else [0]
    all_rows: list[dict] = []

    for held_out in sorted(usable):
        source = [e for n, es in usable.items() if n != held_out for e in es]
        target_entries = usable[held_out]
        src_ds = build_validity_dataset(source, H=H, L=L, delta=1)
        tgt_ds = build_validity_dataset(target_entries, H=H, L=L, delta=1)

        # target is split so that an in-domain model has its own train/cal, and every method is
        # scored on exactly the same held-out origins
        try:
            tgt_train, tgt_cal, tgt_test = stream_level_split(
                tgt_ds, train_frac=0.5, cal_frac=0.2, seed=0)
        except ValueError as exc:
            log(f"SKIP {held_out}: target cannot be split ({exc})")
            continue
        src_train, src_cal, _ = stream_level_split(src_ds, train_frac=0.6, cal_frac=0.2, seed=0)

        # the two lower references, both fitted on the TARGET's own training split
        km = np.tile(kaplan_meier(tgt_train.y_tilde, tgt_train.event, H), (len(tgt_test), 1))
        age = age_marginal_survival(tgt_train, tgt_test, H)
        refs = {"km_marginal": km, "age_marginal": age}
        log(f"\n=== held-out domain: {held_out} "
            f"(train {len(tgt_train)} / test {len(tgt_test)} origins, "
            f"source {len(src_train)} origins from {sorted(set(n for n in usable if n != held_out))})")
        for rname, rsurv in refs.items():
            log(f"  {rname:14s} {evaluate(rsurv, tgt_test)}")

        for seed in seeds:
            target_file = OUT / f"{held_out}_seed{seed}.json"
            if should_skip(target_file, args.force, logger=None):
                all_rows.append(__import__("json").loads(target_file.read_text()))
                continue
            with run_manifest(f"transfer_{held_out}_seed{seed}",
                              out_path=OUT / f"manifest_{held_out}_seed{seed}.json",
                              config={"scale": args.scale, "eps_margin": EPS_MARGIN,
                                      "H": H, "L": L, "held_out": held_out},
                              dataset_id=held_out, seed=seed, scale=args.scale,
                              predictor_id="mixed") as man:
                seed_everything(seed)
                row: dict = {"held_out_domain": held_out, "seed": int(seed),
                             "scale": args.scale,
                             "source_domains": sorted(n for n in usable if n != held_out),
                             "n_source_origins": len(src_train),
                             "n_target_test_origins": len(tgt_test),
                             "n_target_test_events": int(tgt_test.n_events),
                             "censoring_rate": round(tgt_ds.censoring_rate, 4),
                             "metrics": {}, "tests": {}}
                for rname, rsurv in refs.items():
                    row["metrics"][rname] = evaluate(rsurv, tgt_test)

                cfg = dict(encoder="gru", max_epochs=150, patience=25, seed=seed, n_samples=128)
                # TRANSFER: trained only on the other domains, never on the target
                tm = PrismV(PrismVConfig(variant="first_passage", **cfg),
                            H=H, d_in=src_ds.S.shape[-1]).fit(src_train, src_cal)
                transfer_surv = tm.predict_survival(tgt_test)
                row["metrics"]["prismv_transfer"] = evaluate(transfer_surv, tgt_test)

                # IN-DOMAIN upper reference: same architecture, trained on the target's own data
                im = PrismV(PrismVConfig(variant="first_passage", **cfg),
                            H=H, d_in=tgt_ds.S.shape[-1]).fit(tgt_train, tgt_cal)
                row["metrics"]["prismv_in_domain"] = evaluate(
                    im.predict_survival(tgt_test), tgt_test)

                # paired tests of transfer against each lower reference
                tc = concordance_contributions(transfer_surv, tgt_test)
                for rname, rsurv in refs.items():
                    rc = concordance_contributions(rsurv, tgt_test)
                    m = np.isfinite(tc) & np.isfinite(rc)
                    # "greater": transfer should order origins BETTER than the reference
                    p = wilcoxon_paired(rc[m], tc[m], alternative="less")
                    d = cliffs_delta(tc[m], rc[m])
                    pt, lo, hi = blocked_bootstrap_ci(tc[m] - rc[m],
                                                      tgt_test.stream_id[m], seed=seed)
                    row["tests"][f"transfer_vs_{rname}"] = {
                        "p_raw": p, "effect_delta": d, "n_pairs": int(m.sum()),
                        "mean_gain": pt, "ci95": [lo, hi],
                        "above_reference": bool(p <= ALPHA and d > 0)}
                man.metrics = {k: v["antolini_c"] for k, v in row["metrics"].items()}
                write_json(row, target_file)
                all_rows.append(row)
            log(f"  seed={seed}  transfer C={row['metrics']['prismv_transfer']['antolini_c']:.4f} "
                f"| in-domain C={row['metrics']['prismv_in_domain']['antolini_c']:.4f} "
                f"| age-marginal C={row['metrics']['age_marginal']['antolini_c']:.4f}")

    # ------------------------------------------------------------------ verdict
    log("\n" + "=" * 78)
    log("GATE 5 VERDICT  (transfer must be strictly above the marginal baseline)")
    log("=" * 78)
    verdict: dict = {"scale": args.scale, "seeds": seeds, "domains": {},
                     "rule": "transfer concordance strictly above the marginal; the honest floor "
                             "is age_marginal, not km_marginal (D28)"}
    for held_out in sorted({r["held_out_domain"] for r in all_rows}):
        rows = [r for r in all_rows if r["held_out_domain"] == held_out]
        entry: dict = {"n_seeds": len(rows)}
        for k in ("km_marginal", "age_marginal", "prismv_transfer", "prismv_in_domain"):
            cs = [r["metrics"][k]["antolini_c"] for r in rows]
            entry[k] = {"c_mean": round(float(np.mean(cs)), 4),
                        "c_sd": round(float(np.std(cs)), 4)}
        for ref in ("km_marginal", "age_marginal"):
            t = [r["tests"][f"transfer_vs_{ref}"] for r in rows]
            entry[f"vs_{ref}"] = {
                "seeds_above": int(sum(x["above_reference"] for x in t)),
                "mean_delta": round(float(np.mean([x["effect_delta"] for x in t])), 4),
                "mean_gain": round(float(np.mean([x["mean_gain"] for x in t])), 4)}
        entry["gate5_vs_age_marginal"] = (
            "PASS" if entry["vs_age_marginal"]["seeds_above"] == len(rows) else "FAIL")
        verdict["domains"][held_out] = entry
        log(f"{held_out:26s} transfer C={entry['prismv_transfer']['c_mean']:.4f} "
            f"| age-marg {entry['age_marginal']['c_mean']:.4f} "
            f"| in-domain {entry['prismv_in_domain']['c_mean']:.4f} "
            f"| above age-marginal {entry['vs_age_marginal']['seeds_above']}/{len(rows)} seeds "
            f"-> {entry['gate5_vs_age_marginal']}")

    n_pass = sum(v["gate5_vs_age_marginal"] == "PASS" for v in verdict["domains"].values())
    verdict["GATE5"] = "PASS" if n_pass == len(verdict["domains"]) and n_pass > 0 else "FAIL"
    verdict["n_domains_passing"] = n_pass
    write_json(verdict, OUT / "GATE5_VERDICT.json")
    log(f"\nGATE 5 = {verdict['GATE5']}  ({n_pass}/{len(verdict['domains'])} domains)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
