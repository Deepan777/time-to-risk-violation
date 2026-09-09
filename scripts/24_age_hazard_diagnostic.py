#!/usr/bin/env python3
"""24_age_hazard_diagnostic.py -- why an age-blind marginal is not an honest floor.

Decisions D28 and D29 both rest on one empirical claim: **the hazard of a risk violation varies
with how long the deployment has been running**, so a reference curve that cannot see deployment
age is beaten by anything that can, and beating it means very little.

That claim was originally established during the Gate 0 post-mortem and recorded only in the
project log. The manuscript quotes it, so it must exist as a result file with a manifest like every
other number in the paper — which is what this script produces. It computes, on the class-U
negative-control streams and on every real monitored stream:

* the empirical discrete hazard within each deployment-age quintile, and
* the point-biserial correlation between deployment age at the origin and whether that origin's
  horizon contains a violation, with its p-value.

Nothing here fits a model or touches a method's accuracy. It describes the estimand.
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import make_synthetic_stream                      # noqa: E402
from src.monitoring import build_state_table, fit_reference         # noqa: E402
from src.predictors.base import fit_on_pre_deployment               # noqa: E402
from src.shifts import make_u_stream                                # noqa: E402
from src.utils.cli import standard_parser                           # noqa: E402
from src.utils.io import ensure_dir, get_logger, should_skip, write_json  # noqa: E402
from src.utils.manifest import run_manifest                         # noqa: E402
from src.validity.dataset import build_validity_dataset             # noqa: E402

OUT = Path("results/gate0/age_hazard_diagnostic.json")
N_QUANTILES = 5
#: Mirrors scripts/19_gate0_negative_control.py so the diagnosis describes the streams the gate ran.
WINDOW, DEPLOY, H, L, N_STREAMS = 200, 8000, 20, 20, 16
EPS_MARGIN = 0.20


def age_hazard(ds) -> dict:
    """Hazard by deployment-age quintile, plus the age/event correlation.

    Deployment age is taken from the origin index within its own stream, which is exactly what the
    model's `ctx_deployment_age` feature reports, so the diagnostic and the feature describe the
    same quantity.
    """
    age, event = [], []
    for sid in np.unique(ds.stream_id):
        m = ds.stream_id == sid
        o = np.asarray(ds.origin)[m]
        age.append(o - o.min())
        event.append(np.asarray(ds.event)[m])
    age = np.concatenate(age).astype(float)
    event = np.concatenate(event).astype(float)

    edges = np.quantile(age, np.linspace(0.0, 1.0, N_QUANTILES + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(age, edges[1:-1], right=True), 0, len(edges) - 2)
    quintiles = []
    for q in range(len(edges) - 1):
        sel = idx == q
        quintiles.append({
            "quintile": q + 1,
            "n_origins": int(sel.sum()),
            "age_range": [float(edges[q]), float(edges[q + 1])],
            "hazard": round(float(event[sel].mean()), 4) if sel.any() else None,
        })
    # Where every origin is an event the correlation is undefined, not zero. Record it as null so
    # that a reader -- and the manuscript audit -- sees an absent quantity rather than a NaN that
    # could be mistaken for a measurement.
    if np.unique(event).size < 2:
        r, p = None, None
    else:
        rr, pp = stats.pointbiserialr(event.astype(int), age)
        r, p = (None, None) if not np.isfinite(rr) else (round(float(rr), 4), float(pp))
    return {
        "n_origins": int(age.size),
        "n_streams": int(np.unique(ds.stream_id).size),
        "quintiles": quintiles,
        "point_biserial_r": r,
        "point_biserial_p": p,
        "correlation_defined": r is not None,
        "hazard_min": round(float(min(q["hazard"] for q in quintiles if q["hazard"] is not None)), 4),
        "hazard_max": round(float(max(q["hazard"] for q in quintiles if q["hazard"] is not None)), 4),
    }


def u_stream_dataset(log):
    """Rebuild the class-U negative-control streams exactly as Gate 0 built them."""
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
                        "forecastability": mans[0].forecastability_type})
        log(f"  U{i}: {len(risks)} windows, eps={eps:.4f}")
    return build_validity_dataset(entries, H=H, L=L, delta=1)


def real_datasets(log) -> dict:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(Path("data/cache/monitored").glob("*.pkl")):
        e = pickle.loads(f.read_bytes())
        by_ds[e["dataset_id"]].append({**e, "eps": float(min(e["eps"] - 0.05 + EPS_MARGIN, 1.0))})
    out = {}
    for name in sorted(by_ds):
        try:
            out[name] = q = age_hazard(build_validity_dataset(by_ds[name], H=H, L=L, delta=1))
            corr = ("undefined -- every origin is an event" if q["point_biserial_r"] is None
                    else f"r={q['point_biserial_r']:+.4f} p={q['point_biserial_p']:.3g}")
            log(f"  {name:18s} hazard {q['hazard_min']:.3f}-{q['hazard_max']:.3f}  {corr}")
        except ValueError as exc:
            log(f"  {name:18s} skipped: {exc}")
    return out


def regime_comparability(log) -> dict:
    """Is the Gate 1 test split drawn from the same risk regime as its training split?

    A chronological split on a monotonically rising risk series puts training origins in a low-risk
    regime and test origins in a high-risk one, and then the comparison measures extrapolation to
    an unseen risk level rather than prognosis. An earlier version of Gate 1 failed for exactly
    that reason (D23). The ratio below is how we check that the accepted verdict is not the same
    artefact, so it belongs in a result file rather than in prose.

    The split reproduced here is Gate 1's own: the gapped chronological split over pooled origins.
    """
    from src.evaluation.pipeline import DATASET_FAMILY
    from src.validity.dataset import chronological_split

    by_family: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(Path("data/cache/monitored").glob("*.pkl")):
        e = pickle.loads(f.read_bytes())
        fam = DATASET_FAMILY.get(e["dataset_id"])
        if fam is None:
            continue
        by_family[fam].append({**e, "eps": float(min(e["eps"] - 0.05 + EPS_MARGIN, 1.0))})

    out = {}
    for fam in sorted(by_family):
        try:
            ds = build_validity_dataset(by_family[fam], H=H, L=L, delta=1)
            train, _cal, test = chronological_split(ds, train_frac=0.5, cal_frac=0.2)
        except ValueError as exc:
            log(f"  {fam:24s} skipped: {exc}")
            continue
        tr = float(np.nanmean(train.risk_path))
        te = float(np.nanmean(test.risk_path))
        out[fam] = {
            "train_mean_risk": round(tr, 4),
            "test_mean_risk": round(te, 4),
            "test_over_train_ratio": round(te / tr, 4) if tr else None,
            "n_train_origins": len(train),
            "n_test_origins": len(test),
        }
        log(f"  {fam:24s} train {tr:.4f} test {te:.4f} ratio {te / tr:.4f} "
            f"({len(train)}/{len(test)} origins)")
    return out


def main() -> int:
    ap = standard_parser("Age/hazard diagnostic behind D28 and D29", require_config=False)
    args = ap.parse_args()
    ensure_dir(OUT.parent)
    log = get_logger("age_hazard", args.log or "logs/age_hazard.log").info

    if should_skip(OUT, args.force, logger=None):
        log(f"{OUT} exists; pass --force to recompute")
        return 0

    with run_manifest("age_hazard_diagnostic",
                      out_path=OUT.parent / "manifest_age_hazard.json",
                      config={"H": H, "L": L, "n_quantiles": N_QUANTILES,
                              "eps_margin": EPS_MARGIN, "n_u_streams": N_STREAMS},
                      dataset_id="mixed", seed=0, scale="full",
                      predictor_id="xgboost") as man:
        log("class-U negative-control streams:")
        u = age_hazard(u_stream_dataset(log))
        log(f"  U pooled: hazard {u['hazard_min']:.3f}-{u['hazard_max']:.3f} "
            f"r={u['point_biserial_r']:+.4f} p={u['point_biserial_p']:.3g}")
        log("real monitored streams:")
        real = real_datasets(log)
        log("Gate 1 regime comparability:")
        regimes = regime_comparability(log)
        out = {
            "claim": "The hazard of a risk violation varies with deployment age, so a marginal "
                     "reference that cannot see age is not an honest floor (D28, D29).",
            "note": "Descriptive only. No model is fitted and no method's accuracy appears here.",
            "n_quantiles": N_QUANTILES, "H": H, "L": L, "eps_margin": EPS_MARGIN,
            "negative_control_u_streams": u,
            "real_datasets": real,
            "gate1_regime_comparability": regimes,
        }
        man.metrics = {"u_point_biserial_r": u["point_biserial_r"]}
        write_json(out, OUT)
    log(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
