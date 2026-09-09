#!/usr/bin/env python3
"""16_run_residual.py -- Experiment E16: does deployment state help ON TOP of the online level?

**Gate 1 has already failed and that verdict stands permanently.** This is not an attempt to
re-run it. It is a separate, separately-labelled experiment testing the specific mechanism the
Gate 1 diagnosis identified: the naive baselines re-read each stream's own recent error level at
every origin and extrapolate online, while PRISM-V applied a fixed mapping learned on earlier
origins.

E16 therefore asks a different question. Instead of predicting absolute risk, the model predicts a
**correction to that online level**. A zero correction reproduces the anchor exactly, so the
comparison isolates whether the deployment state carries information *on top of* the best simple
extrapolator rather than in competition with it.

Two honest outcomes, both stated in advance:
  * the residual model beats the baselines -> the state does add information, but only once the
    online level is accounted for, and the paper's claim becomes conditional on that;
  * it does not -> the Gate 1 negative result is strengthened, because even the variant designed
    around the diagnosed weakness fails.

Nothing here is tuned toward either outcome: the eps margin is the one fixed by D26, the baselines
keep their calibration-split budgets, and the anchor is a pure function of the matured history.

Writes one JSON per (family, model seed) into results/e16_residual/ plus a run manifest.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.naive_temporal import build_naive_baselines           # noqa: E402
from src.evaluation.statistics import (  # noqa: E402
    ALPHA, MIN_EFFECT, beats, blocked_bootstrap_ci, compare,
)
from src.metrics.risk_path_metrics import (                              # noqa: E402
    GATE1_HORIZONS, horizon_mae, per_origin_abs_error, risk_path_metrics,
)
from src.utils.cli import standard_parser                                # noqa: E402
from src.utils.io import ensure_dir, get_logger, should_skip, write_json  # noqa: E402
from src.utils.manifest import run_manifest                              # noqa: E402
from src.utils.seeding import SEEDS_FULL, seed_everything                # noqa: E402
from src.validity.dataset import build_validity_dataset, chronological_split  # noqa: E402
from src.validity.model import PrismV, PrismVConfig                      # noqa: E402

#: Fixed by decision D26 before this experiment was first run. Do not change it here.
EPS_MARGIN = 0.20
EPS_QUANTILE = 0.75
H, L = 20, 20


def load_cached_entries(margin: float) -> dict[str, list[dict]]:
    """Cached monitored streams, grouped by dataset family, with eps recomputed at `margin`."""
    import pickle

    by_family: dict[str, list[dict]] = defaultdict(list)
    files = sorted(Path("data/cache/monitored").glob("*.pkl"))
    if not files:
        raise FileNotFoundError(
            "no monitored streams cached; run scripts/_build_streams.py first")
    for f in files:
        e = pickle.loads(f.read_bytes())
        # the cached eps used the build-time margin of 0.05; recover the frozen pre-deployment
        # quantile and re-apply the margin fixed by D26
        e = {**e, "eps": float(min(e["eps"] - 0.05 + margin, 1.0))}
        by_family[e["family"]].append(e)
    return dict(by_family)


def run_family(family: str, entries: list[dict], model_seed: int, log) -> dict:
    """One family, one model seed: train both variants, run all baselines, compare."""
    seed_everything(model_seed)
    ds = build_validity_dataset(entries, H=H, L=L, delta=1)
    train, cal, test = chronological_split(ds, train_frac=0.5, cal_frac=0.2)

    log(f"  {family} seed={model_seed}: N={len(ds)} cens={ds.censoring_rate:.3f} | "
        f"train={len(train)}({train.n_events}ev) cal={len(cal)}({cal.n_events}ev) "
        f"test={len(test)}({test.n_events}ev)")

    preds: dict[str, np.ndarray] = {}
    budgets: dict[str, int] = {}

    # -- baselines, tuned on the CALIBRATION split only
    for b in build_naive_baselines(seed=model_seed):
        b.fit(cal.risk_hist, cal.risk_path)
        preds[b.name] = b.predict_path(test.risk_hist, H)
        budgets[b.name] = int(b.tuning_budget)

    # -- PRISM-V, both variants, early-stopped on the same calibration split
    # the anchor alone, as an explicit competitor: this is what the model must improve on
    preds["anchor_only"] = test.anchor
    budgets["anchor_only"] = 0

    for variant in ("two_head", "first_passage"):
        cfg = PrismVConfig(variant=variant, encoder="gru", max_epochs=150, patience=25,
                           seed=model_seed, n_samples=128, early_stop_metric="risk_mae",
                           use_anchor=True)
        m = PrismV(cfg, H=H, d_in=ds.S.shape[-1]).fit(train, cal)
        preds[f"prismv_{variant}"] = m.predict_risk_path(test)
        budgets[f"prismv_{variant}"] = 1
        log(f"     {variant}(residual): best_epoch={m.best_epoch}")

    baseline_names = [k for k in preds if not k.startswith("prismv_")]
    ref = preds["persistence"]

    out: dict = {
        "family": family,
        "model_seed": int(model_seed),
        "eps_margin": EPS_MARGIN,
        "eps_quantile": EPS_QUANTILE,
        "n_origins": {"train": len(train), "cal": len(cal), "test": len(test)},
        "n_events": {"train": train.n_events, "cal": cal.n_events, "test": test.n_events},
        "censoring_rate": ds.censoring_rate,
        "n_streams": int(len(np.unique(ds.stream_id))),
        "datasets": sorted({e["dataset_id"] for e in entries}),
        "tuning_budgets": budgets,
        "metrics": {},
        "e16": {},
    }
    for name, p in preds.items():
        out["metrics"][name] = risk_path_metrics(p, test.risk_path, reference=ref)
        # 95% CI on mean absolute error, resampling WHOLE STREAMS rather than origins:
        # origins inside one deployment are temporally dependent, so resampling them
        # independently would report an interval narrower than the data supports.
        for h in GATE1_HORIZONS:
            mask = np.isfinite(test.risk_path[:, h - 1])
            err = np.abs(p[mask, h - 1] - test.risk_path[mask, h - 1])
            point, lo, hi = blocked_bootstrap_ci(err, test.stream_id[mask], n_boot=2000,
                                                 seed=model_seed)
            out["metrics"][name][f"mae_h{h}_ci95"] = [point, lo, hi]

    # -- the Gate 1 comparison
    for variant in ("two_head", "first_passage"):
        mk = f"prismv_{variant}"
        per_h: dict[str, dict] = {}
        for h in GATE1_HORIZONS:
            me = per_origin_abs_error(preds[mk], test.risk_path, h)
            results = beats([compare(me, per_origin_abs_error(preds[b], test.risk_path, h), mk, b)
                             for b in baseline_names])
            per_h[f"h{h}"] = {
                "comparisons": [r.as_dict() for r in results],
                "n_beaten": int(sum(r.wins for r in results)),
                "n_baselines": len(results),
                "all_beaten": bool(all(r.wins for r in results)),
            }
        out["e16"][mk] = {
            "per_horizon": per_h,
            "beats_all_this_family": bool(all(v["all_beaten"] for v in per_h.values())),
        }
    return out


def main() -> int:
    ap = standard_parser("E16: residual/anchored PRISM-V vs naive temporal baselines",
                         require_config=False)
    ap.add_argument("--scale", default="full", choices=["smoke", "dev", "full"])
    args = ap.parse_args()

    out_dir = Path(args.out or "results/e16_residual")
    ensure_dir(out_dir)
    log = get_logger("e16", args.log or "logs/e16.log").info

    seeds = list(SEEDS_FULL) if args.scale == "full" else [0]
    by_family = load_cached_entries(EPS_MARGIN)
    log(f"families: { {k: len(v) for k, v in by_family.items()} }  seeds={seeds}  "
        f"eps_margin={EPS_MARGIN} (D26)")

    all_results: list[dict] = []
    excluded: dict[str, str] = {}
    for family, entries in sorted(by_family.items()):
        # A family whose frozen predictor is invalid for most of deployment cannot support the
        # protocol's gapped chronological split: there are too few origins at which the model is
        # still valid. That is a scope condition on the estimand, not a tuning knob, so it is
        # recorded with its reason and the family is excluded rather than rescued.
        try:
            probe = build_validity_dataset(entries, H=H, L=L, delta=1)
            chronological_split(probe, train_frac=0.5, cal_frac=0.2)
        except ValueError as exc:
            excluded[family] = str(exc)
            log(f"EXCLUDED {family}: {exc}")
            continue
        for seed in seeds:
            target = out_dir / f"{family}_seed{seed}.json"
            if should_skip(target, args.force, logger=None):
                all_results.append(__import__("json").loads(target.read_text()))
                continue
            with run_manifest(f"E16_residual_{family}_seed{seed}", out_path=out_dir /
                              f"manifest_{family}_seed{seed}.json",
                              config={"scale": args.scale, "eps_margin": EPS_MARGIN,
                                      "eps_quantile": EPS_QUANTILE, "H": H, "L": L},
                              dataset_id=family, seed=seed, scale=args.scale,
                              predictor_id="mixed") as man:
                res = run_family(family, entries, seed, log)
                res["scale"] = args.scale
                man.metrics = {"n_test": res["n_origins"]["test"],
                               "censoring_rate": res["censoring_rate"]}
                write_json(res, target)
                all_results.append(res)

    # ---------------------------------------------------------------- verdict
    log("\n" + "=" * 78)
    log("E16 SUMMARY (Gate 1 verdict is unchanged and still FAIL)")
    log("=" * 78)
    verdict: dict = {"rule": "beats EVERY naive baseline at h in {1,5,10} on >= 2 families; "
                             f"'beats' = Holm-corrected p <= {ALPHA} AND |delta| >= {MIN_EFFECT}",
                     "scale": args.scale, "seeds": seeds, "eps_margin": EPS_MARGIN,
                     "families": {}, "excluded_families": excluded}

    for variant in ("two_head", "first_passage"):
        mk = f"prismv_{variant}"
        fam_pass: dict[str, dict] = {}
        for family in sorted(f for f in by_family if f not in excluded):
            rows = [r for r in all_results if r["family"] == family]
            if not rows:
                continue
            passes = [r["e16"][mk]["beats_all_this_family"] for r in rows]
            beaten = {f"h{h}": [r["e16"][mk]["per_horizon"][f"h{h}"]["n_beaten"] for r in rows]
                      for h in GATE1_HORIZONS}
            maes = {f"h{h}": [r["metrics"][mk][f"mae_h{h}"] for r in rows]
                    for h in GATE1_HORIZONS}
            fam_pass[family] = {
                "seeds_passing": int(sum(passes)), "n_seeds": len(passes),
                "family_passes": bool(all(passes)) if passes else False,
                "n_beaten_by_horizon": beaten,
                "mae_mean_by_horizon": {k: float(np.mean(v)) for k, v in maes.items()},
                "mae_sd_by_horizon": {k: float(np.std(v)) for k, v in maes.items()},
            }
            log(f"{mk:22s} {family:18s} passes {sum(passes)}/{len(passes)} seeds | "
                f"beaten/6: " + " ".join(f"h{h}={beaten[f'h{h}']}" for h in GATE1_HORIZONS))
        n_fam = sum(v["family_passes"] for v in fam_pass.values())
        verdict["families"][mk] = fam_pass
        verdict[f"{mk}_families_passing"] = n_fam
        verdict[f"{mk}_beats_all_on_n_families"] = n_fam
        log(f"  -> {mk}: {n_fam} family/families pass; GATE 1 = "
            f"{'PASS' if n_fam >= 2 else 'FAIL'}\n")

    write_json(verdict, out_dir / "E16_SUMMARY.json")
    log(f"verdict written to {out_dir / 'E16_SUMMARY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
