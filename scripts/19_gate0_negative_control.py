#!/usr/bin/env python3
"""19_gate0_negative_control.py -- Gate 0 at full scale, as a result file.

Gate 0 (`proposal/go_no_go.md`): on class-U streams, where the conditional distribution is replaced
at a time drawn independently of every observable, PRISM-V must be statistically indistinguishable
from the marginal Kaplan-Meier curve. If it beats the marginal there, the pipeline is presumed leaky
and everything else is suspect.

This had previously been run only at smoke scale. The manuscript quotes it, so it must exist as a
`full`-scale result file with a manifest like every other number in the paper.

`tau` is drawn per stream (decision D22): with `tau` pinned, time-to-violation is exactly `tau - t`
and the state's own deployment-age feature predicts it perfectly, which would make the control
vacuous while appearing to pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import make_synthetic_stream                      # noqa: E402
from src.metrics.survival_metrics import (                          # noqa: E402
    antolini_concordance, ipcw_integrated_brier,
)
from src.monitoring import build_state_table, fit_reference         # noqa: E402
from src.predictors.base import fit_on_pre_deployment               # noqa: E402
from src.shifts import make_u_stream                                # noqa: E402
from src.survival.utils import kaplan_meier                         # noqa: E402
from src.utils.cli import standard_parser                           # noqa: E402
from src.utils.io import ensure_dir, get_logger, should_skip, write_json  # noqa: E402
from src.utils.manifest import run_manifest                         # noqa: E402
from src.utils.seeding import SEEDS_FULL, seed_everything           # noqa: E402
from src.validity.dataset import build_validity_dataset, stream_level_split  # noqa: E402
from src.validity.model import PrismV, PrismVConfig                 # noqa: E402

OUT = Path("results/gate0")
WINDOW, DEPLOY, H, L = 200, 8000, 20, 20
N_STREAMS = 16
#: Concordance this far from 0.5 counts as distinguishable from the marginal.
EQUIVALENCE_MARGIN = 0.10


def build_u_entries(log) -> list[dict]:
    entries = []
    for i in range(N_STREAMS):
        base = make_synthetic_stream("sea", n_rows=60000, seed=i)
        rng = np.random.default_rng(5000 + i)
        stream, mans, tau = make_u_stream(base, rng, deployment_start=DEPLOY,
                                          tau_range=(0.25, 0.80), severity=0.45)
        handle, pre, dep = fit_on_pre_deployment(
            stream, family="xgboost", seed=0, pre_deployment_frac=0.15, window_size=WINDOW)
        ref = fit_reference(pre.X(), pre.y(), handle, window_size=WINDOW, seed=0)
        eps = ref.eps_from_quantile(0.75, margin=0.05)
        states, risks = build_state_table(dep, handle, ref, window_size=WINDOW,
                                          stride=WINDOW, delta=1)
        entries.append({"stream_id": f"U{i}", "dataset_id": "synthetic_sea_U",
                        "family": "tier_d_negative_control", "seed": i,
                        "states": states, "risks": risks, "eps": float(eps),
                        "forecastability": mans[0].forecastability_type, "tau": int(tau)})
        log(f"  U{i}: {len(risks)} windows, eps={eps:.4f}, tau={tau}")
    return entries


def main() -> int:
    ap = standard_parser("Gate 0: negative control on class-U streams", require_config=False)
    ap.add_argument("--scale", default="full", choices=["smoke", "dev", "full"])
    args = ap.parse_args()
    ensure_dir(OUT)
    log = get_logger("gate0", args.log or "logs/gate0.log").info

    verdict_path = OUT / "GATE0_VERDICT.json"
    if should_skip(verdict_path, args.force, logger=None):
        log(f"{verdict_path} exists; pass --force to recompute")
        return 0

    entries = build_u_entries(log)
    ds = build_validity_dataset(entries, H=H, L=L, delta=1)
    # Stream-level, not chronological: the U streams are independent draws with
    # independently drawn change points, so events cluster around each stream's own tau
    # and a time-banded split can leave calibration with no events at all. Splitting by
    # stream also asks the question the control actually poses - does a model fitted on
    # some unforecastable streams show skill on unseen ones.
    train, cal, test = stream_level_split(ds, train_frac=0.5, cal_frac=0.2, seed=0)
    log(f"U dataset: N={len(ds)} cens={ds.censoring_rate:.3f} | train={len(train)} "
        f"cal={len(cal)}({cal.n_events}ev) test={len(test)}({test.n_events}ev)")

    km_curve = kaplan_meier(train.y_tilde, train.event, H)
    km_surv = np.tile(km_curve, (len(test), 1))
    km = {"antolini_c": float(antolini_concordance(km_surv, test.y_tilde, test.event)),
          "ipcw_ibs": float(ipcw_integrated_brier(km_surv, test.y_tilde, test.event))}
    log(f"Kaplan-Meier: C={km['antolini_c']:.4f} IBS={km['ipcw_ibs']:.4f}")

    seeds = list(SEEDS_FULL) if args.scale == "full" else [0]
    per_variant: dict[str, dict] = {}
    for variant in ("two_head", "first_passage"):
        cs, ibs = [], []
        for seed in seeds:
            seed_everything(seed)
            with run_manifest(f"gate0_{variant}_seed{seed}",
                              out_path=OUT / f"manifest_{variant}_seed{seed}.json",
                              config={"scale": args.scale, "H": H, "L": L,
                                      "n_streams": N_STREAMS},
                              dataset_id="synthetic_sea_U", seed=seed, scale=args.scale,
                              predictor_id="xgboost") as man:
                m = PrismV(PrismVConfig(variant=variant, encoder="gru", max_epochs=150,
                                        patience=25, seed=seed, n_samples=128),
                           H=H, d_in=ds.S.shape[-1]).fit(train, cal)
                surv = m.predict_survival(test)
                c = float(antolini_concordance(surv, test.y_tilde, test.event))
                b = float(ipcw_integrated_brier(surv, test.y_tilde, test.event))
                cs.append(c)
                ibs.append(b)
                man.metrics = {"antolini_c": c, "ipcw_ibs": b}
            log(f"  {variant} seed={seed}: C={c:.4f} IBS={b:.4f}")
        mean_c = float(np.mean(cs))
        per_variant[f"prismv_{variant}"] = {
            "antolini_c_mean": round(mean_c, 4),
            "antolini_c_sd": round(float(np.std(cs)), 4),
            "antolini_c_per_seed": [round(v, 4) for v in cs],
            "ipcw_ibs_mean": round(float(np.mean(ibs)), 4),
            "ipcw_ibs_sd": round(float(np.std(ibs)), 4),
            "distinguishable_from_marginal": bool(abs(mean_c - 0.5) > EQUIVALENCE_MARGIN),
        }

    passes = not any(v["distinguishable_from_marginal"] for v in per_variant.values())
    verdict = {
        "scale": args.scale, "seeds": seeds, "n_streams": N_STREAMS,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "n_origins": {"train": len(train), "cal": len(cal), "test": len(test)},
        "censoring_rate": round(ds.censoring_rate, 4),
        "kaplan_meier": {k: round(v, 4) for k, v in km.items()},
        "variants": per_variant,
        "GATE0": "PASS" if passes else "FAIL -- LEAKAGE PRESUMED",
        "rule": "on class-U streams PRISM-V must be statistically indistinguishable from the "
                "marginal Kaplan-Meier curve; beating it implies leakage",
    }
    write_json(verdict, verdict_path)
    log(f"GATE 0 = {verdict['GATE0']}  (written to {verdict_path})")
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
