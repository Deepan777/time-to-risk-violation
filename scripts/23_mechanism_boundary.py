#!/usr/bin/env python3
"""23_mechanism_boundary.py -- the PR-H1 confirmatory test and the PR-H2 baseline table.

Pre-registered in `proposal/PREREGISTRATION_drift_mechanism.md` and amended once, before any model
was fitted, in `proposal/PREREGISTRATION_drift_mechanism_ADDENDUM_1.md`. Nothing here decides
anything the pre-registration did not already decide; this script only executes it.

**PR-H1.** Deployment-state monitoring carries information about time-to-risk-violation when the
drift has a physical accumulating mechanism, and little or none when it is social. Decided by:
PRISM-V beating the **age-stratified marginal** on at least two of three usable PHYSICAL datasets
*and* failing on the SOCIAL ones. "Beats" is the project's single definition, unchanged:
Holm-corrected paired Wilcoxon at alpha = 0.05 **and** Cliff's delta >= 0.147, on every seed.

**PR-H2.** Deployment age alone is a strong predictor across drift regimes. Descriptive; the
evidence is the table.

Three tiers of dataset, fixed by rule rather than by choice:

* `PRIMARY`   -- usable under D26, classified PHYSICAL or SOCIAL. These and only these decide PR-H1.
* `SECONDARY` -- reported in full, decides nothing: datasets that failed the D26 usability screen,
                 `insects` (which generated the hypothesis and is disqualified as confirmatory
                 evidence), and `elec2` (AMBIGUOUS).
* refused     -- never monitored at all.

The secondary tier is reported rather than dropped because a reader checking the sample-size
falsifier in section 3 of the pre-registration needs to see every origin count, including the ones
that did not qualify.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import yaml

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
_t = import_module("20_transfer")
age_marginal_survival = _t.age_marginal_survival
concordance_contributions = _t.concordance_contributions
_g2 = import_module("21_gate2_survival")
cox_survival = _g2.cox_survival

OUT = Path("results/mechanism")
CONFIG = Path("configs/drift_mechanism.yaml")
SCREEN = Path("results/processed/mechanism_usability_screen.json")
EPS_MARGIN, H, L = 0.20, 20, 20
REFERENCES = ("km_marginal", "cox_ph", "age_marginal")
DECIDING_REFERENCE = "age_marginal"

#: Disqualified as confirmatory evidence by section 1 of the pre-registration: it generated the
#: hypothesis. Reported in the secondary tier so the reader can see what the hunch rested on.
HYPOTHESIS_GENERATING = ("insects",)


def load_classes() -> dict[str, dict]:
    cfg = yaml.safe_load(CONFIG.read_text())
    out: dict[str, dict] = {}
    for section in ("incumbent", "confirmatory", "confirmatory_addendum"):
        for name, entry in (cfg.get(section) or {}).items():
            out[name] = {"drift_class": entry.get("class"),
                         "measured": entry.get("measured"),
                         "role": entry.get("role"),
                         "config_section": section}
    return out


def load_by_dataset(margin: float) -> dict[str, list[dict]]:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(Path("data/cache/monitored").glob("*.pkl")):
        e = pickle.loads(f.read_bytes())
        by_ds[e["dataset_id"]].append({**e, "eps": float(min(e["eps"] - 0.05 + margin, 1.0))})
    return dict(by_ds)


def assign_tier(name: str, classes: dict, usable: dict[str, bool]) -> tuple[str, str]:
    """(tier, reason). Applies the pre-registered rules; makes no judgement of its own."""
    cls = (classes.get(name) or {}).get("drift_class")
    if name in HYPOTHESIS_GENERATING:
        return "SECONDARY", "hypothesis-generating; disqualified as confirmatory evidence"
    if cls not in ("PHYSICAL", "SOCIAL"):
        return "SECONDARY", f"drift class {cls}; excluded from the confirmatory test"
    if not usable.get(name, False):
        return "SECONDARY", "fails the D26 usability screen"
    return "PRIMARY", "usable and classified"


def evaluate(name: str, entries: list[dict], seeds, scale: str, log) -> list[dict]:
    ds = build_validity_dataset(entries, H=H, L=L, delta=1)
    train, cal, test = stream_level_split(ds, train_frac=0.5, cal_frac=0.2, seed=0)
    cox_surv, cox_status = cox_survival(train, test, H)
    refs = {
        "km_marginal": np.tile(kaplan_meier(train.y_tilde, train.event, H), (len(test), 1)),
        "cox_ph": cox_surv,
        "age_marginal": age_marginal_survival(train, test, H),
    }
    log(f"  cox reference: {cox_status}")
    log(f"\n=== {name}: train {len(train)} / cal {len(cal)} / test {len(test)} origins, "
        f"{test.n_events} events, censoring {ds.censoring_rate:.3f}")
    for rn, rs in refs.items():
        log(f"  {rn:14s} C={antolini_concordance(rs, test.y_tilde, test.event):.4f}")

    rows = []
    for seed in seeds:
        tgt = OUT / f"{name}_seed{seed}.json"
        if should_skip(tgt, False, logger=None):
            rows.append(json.loads(tgt.read_text()))
            continue
        with run_manifest(f"PRH1_{name}_seed{seed}",
                          out_path=OUT / f"manifest_{name}_seed{seed}.json",
                          config={"scale": scale, "eps_margin": EPS_MARGIN, "H": H, "L": L},
                          dataset_id=name, seed=seed, scale=scale,
                          predictor_id="mixed") as man:
            seed_everything(seed)
            m = PrismV(PrismVConfig(variant="first_passage", encoder="gru", max_epochs=150,
                                    patience=25, seed=seed, n_samples=128),
                       H=H, d_in=ds.S.shape[-1]).fit(train, cal)
            surv = m.predict_survival(test)

            row: dict = {
                "dataset": name, "seed": int(seed), "scale": scale,
                "n_test_origins": len(test), "n_test_events": int(test.n_events),
                "n_origins_total": len(ds),
                "censoring_rate": round(ds.censoring_rate, 4),
                "n_streams": int(len(np.unique(ds.stream_id))),
                "cox_status": cox_status,
                "metrics": {}, "tests": {},
            }
            for rn, rs in {**refs, "prismv": surv}.items():
                row["metrics"][rn] = {
                    "antolini_c": float(antolini_concordance(rs, test.y_tilde, test.event)),
                    "ipcw_ibs": float(ipcw_integrated_brier(rs, test.y_tilde, test.event)),
                    "d_cal_p": float(d_calibration(rs, test.y_tilde, test.event)["d_cal_p"]),
                }
            pc = concordance_contributions(surv, test)
            praw = {}
            for rn, rs in refs.items():
                rc = concordance_contributions(rs, test)
                msk = np.isfinite(pc) & np.isfinite(rc)
                p = wilcoxon_paired(rc[msk], pc[msk], alternative="less")
                d = cliffs_delta(pc[msk], rc[msk])
                pt, lo, hi = blocked_bootstrap_ci(pc[msk] - rc[msk], test.stream_id[msk], seed=seed)
                praw[rn] = p
                row["tests"][rn] = {"p_raw": p, "effect_delta": d, "n_pairs": int(msk.sum()),
                                    "mean_gain": pt, "ci95": [lo, hi]}
            corrected = holm_bonferroni(praw)
            for rn in refs:
                adj = corrected[rn]["p_adjusted"]
                row["tests"][rn]["p_adjusted"] = adj
                row["tests"][rn]["beats"] = bool(
                    adj <= ALPHA and row["tests"][rn]["effect_delta"] >= MIN_EFFECT)
            man.metrics = {k: v["antolini_c"] for k, v in row["metrics"].items()}
            write_json(row, tgt)
            rows.append(row)
        b = row["tests"][DECIDING_REFERENCE]
        log(f"  seed={seed} PRISM-V C={row['metrics']['prismv']['antolini_c']:.4f} "
            f"vs age {row['metrics'][DECIDING_REFERENCE]['antolini_c']:.4f} "
            f"| p_adj={b['p_adjusted']:.3g} delta={b['effect_delta']:+.3f} beats={b['beats']}")
    return rows


def summarise(name: str, rows: list[dict]) -> dict:
    entry: dict = {"n_seeds": len(rows),
                   "n_origins_total": rows[0]["n_origins_total"],
                   "n_test_origins": rows[0]["n_test_origins"],
                   "n_test_events": rows[0]["n_test_events"],
                   "censoring_rate": rows[0]["censoring_rate"]}
    for k in (*REFERENCES, "prismv"):
        cs = [r["metrics"][k]["antolini_c"] for r in rows]
        ib = [r["metrics"][k]["ipcw_ibs"] for r in rows]
        dp = [r["metrics"][k]["d_cal_p"] for r in rows]
        entry[k] = {"c_mean": round(float(np.mean(cs)), 4),
                    "c_sd": round(float(np.std(cs)), 4),
                    "ibs_mean": round(float(np.mean(ib)), 4),
                    "ibs_sd": round(float(np.std(ib)), 4),
                    # D-calibration is a goodness-of-fit test: a small p rejects calibration.
                    # Reported as the number of seeds rejected, which is what a reader needs.
                    "d_cal_p_median": round(float(np.median(dp)), 4),
                    "d_cal_rejected_seeds": int(sum(1 for p in dp if p < 0.05))}
    for rn in REFERENCES:
        t = [r["tests"][rn] for r in rows]
        entry[f"vs_{rn}"] = {
            "seeds_beaten": int(sum(x["beats"] for x in t)),
            "mean_delta": round(float(np.mean([x["effect_delta"] for x in t])), 4),
            "mean_p_adj": float(np.mean([x["p_adjusted"] for x in t])),
            "mean_gain": round(float(np.mean([x["mean_gain"] for x in t])), 4)}
    entry["beats_deciding_reference"] = bool(
        entry[f"vs_{DECIDING_REFERENCE}"]["seeds_beaten"] == len(rows))
    return entry


def main() -> int:
    ap = standard_parser("PR-H1 drift-mechanism boundary test and PR-H2 baseline table",
                         require_config=False)
    ap.add_argument("--scale", default="full", choices=["smoke", "dev", "full"])
    args = ap.parse_args()
    ensure_dir(OUT)
    log = get_logger("mechanism", args.log or "logs/mechanism.log").info

    classes = load_classes()
    usable = {r["dataset"]: bool(r.get("usable")) for r in json.loads(SCREEN.read_text())["screen"]}
    by_ds = load_by_dataset(EPS_MARGIN)
    seeds = list(SEEDS_FULL) if args.scale == "full" else [0]

    results: dict[str, dict] = {}
    for name in sorted(by_ds):
        tier, reason = assign_tier(name, classes, usable)
        info = classes.get(name) or {}
        try:
            rows = evaluate(name, by_ds[name], seeds, args.scale, log)
        except ValueError as exc:
            log(f"EXCLUDED {name}: {exc}")
            results[name] = {"tier": "REFUSED", "reason": str(exc),
                             "drift_class": info.get("drift_class"),
                             "measured": info.get("measured")}
            continue
        entry = summarise(name, rows)
        entry.update({"tier": tier, "tier_reason": reason,
                      "drift_class": info.get("drift_class"),
                      "measured": info.get("measured")})
        results[name] = entry

    # ------------------------------------------------------------------ the pre-registered verdict
    prim = {k: v for k, v in results.items() if v.get("tier") == "PRIMARY"}
    phys = sorted(k for k, v in prim.items() if v["drift_class"] == "PHYSICAL")
    soc = sorted(k for k, v in prim.items() if v["drift_class"] == "SOCIAL")
    phys_meas = [k for k in phys if results[k].get("measured") is True]

    beaten_phys = [k for k in phys if prim[k]["beats_deciding_reference"]]
    beaten_soc = [k for k in soc if prim[k]["beats_deciding_reference"]]

    if len(phys) < 3:
        verdict = "UNDECIDED"
        why = (f"the pre-registration requires three usable PHYSICAL datasets and only "
               f"{len(phys)} could be obtained ({', '.join(phys)}). The evidence below is "
               "reported in full but does not decide PR-H1.")
    elif beaten_soc:
        verdict = "REFUTED"
        why = (f"PRISM-V beats the age-stratified marginal on SOCIAL dataset(s) "
               f"{', '.join(beaten_soc)}, which the boundary claim forbids.")
    elif len(beaten_phys) >= 2:
        verdict = "SUPPORTED"
        why = (f"PRISM-V beats the age-stratified marginal on {len(beaten_phys)} of {len(phys)} "
               f"PHYSICAL datasets and on no SOCIAL dataset.")
    else:
        verdict = "REFUTED"
        why = (f"PRISM-V beats the age-stratified marginal on only {len(beaten_phys)} of "
               f"{len(phys)} PHYSICAL datasets, a minority.")

    # The measured-only reading, fixed by addendum 1 before either archive was downloaded.
    beaten_phys_meas = [k for k in phys_meas if prim[k]["beats_deciding_reference"]]
    measured_only = {
        "physical_measured": phys_meas,
        "beaten": beaten_phys_meas,
        "note": "Addendum 1: where this reading disagrees with the all-PHYSICAL reading, this one "
                "governs the claim, because C-MAPSS is simulator-generated.",
    }

    out = {
        "scale": args.scale, "seeds": seeds, "eps_margin": EPS_MARGIN, "H": H, "L": L,
        "deciding_reference": DECIDING_REFERENCE,
        "beats_rule": f"Holm-corrected p <= {ALPHA} AND Cliff's delta >= {MIN_EFFECT}, every seed",
        "primary_physical": phys, "primary_social": soc,
        "physical_beaten": beaten_phys, "social_beaten": beaten_soc,
        "measured_only_reading": measured_only,
        "PR_H1": verdict, "PR_H1_reason": why,
        "datasets": results,
    }
    write_json(out, OUT / "PRH1_VERDICT.json")

    log("\n" + "=" * 86)
    log(f"{'dataset':18s} {'tier':10s} {'class':9s} {'N':>5s} {'cens':>6s} "
        f"{'PRISM-V':>8s} {'age':>7s} {'km':>7s} {'cox':>7s}  beats_age")
    log("=" * 86)
    for name in sorted(results):
        v = results[name]
        if v.get("tier") == "REFUSED":
            log(f"{name:18s} {'REFUSED':10s} {str(v.get('drift_class')):9s} {v['reason'][:60]}")
            continue
        log(f"{name:18s} {v['tier']:10s} {str(v['drift_class']):9s} {v['n_origins_total']:5d} "
            f"{v['censoring_rate']:6.3f} {v['prismv']['c_mean']:8.4f} "
            f"{v['age_marginal']['c_mean']:7.4f} {v['km_marginal']['c_mean']:7.4f} "
            f"{v['cox_ph']['c_mean']:7.4f}  "
            f"{v[f'vs_{DECIDING_REFERENCE}']['seeds_beaten']}/{v['n_seeds']}")
    log("=" * 86)
    log(f"PR-H1 = {verdict}")
    log(f"  {why}")
    log(f"  measured-only PHYSICAL: {phys_meas} -> beaten {beaten_phys_meas}")
    log(f"written to {OUT / 'PRH1_VERDICT.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
