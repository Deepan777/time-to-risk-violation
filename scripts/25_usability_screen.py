#!/usr/bin/env python3
"""25_usability_screen.py -- can each candidate stream support the estimand at all?

The rule is D26, fixed before the binding Gate 1 run: a dataset is usable only if it yields at
least 100 valid origins **and** a censoring rate strictly between 0 and 0.6, at the operational
margin 0.20. It reads only the structure of the estimand -- how many origins exist at which the
frozen model is still valid, and how many of those survive the horizon -- and never any method's
accuracy. It runs after the frozen base predictor is fitted, because valid origins and
censoring status are properties of that predictor's realised risk series, and before any
prospective monitoring model exists. It is method-independent rather than outcome-blind.

The screen was originally run inline while the drift-mechanism datasets were being built. It lives
here as a script because the manuscript quotes it, and a number the manuscript quotes must be
reproducible by a command rather than by a session.

Candidates refused *before* monitoring are included too, with the reason. A dataset that cannot
yield a single origin is the strongest possible case of the scope condition this screen exists to
expose, and leaving it out of the table would make the screen look kinder than it is.
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.cli import standard_parser                           # noqa: E402
from src.utils.io import ensure_dir, get_logger, write_json         # noqa: E402
from src.validity.dataset import build_validity_dataset             # noqa: E402

OUT = Path("results/processed/mechanism_usability_screen.json")
CONFIG = Path("configs/drift_mechanism.yaml")
EPS_MARGIN, H, L = 0.20, 20, 20
MIN_ORIGINS, MAX_CENSORING = 100, 0.6

#: Candidates that never reached the monitoring stage. Each is refused by a rule that fires at load
#: time, and the refusal is reproduced live below rather than asserted here.
PRE_MONITORING_CANDIDATES = ("hydraulic",)


def classes() -> dict[str, dict]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for section in ("incumbent", "confirmatory", "confirmatory_addendum"):
        for name, entry in (cfg.get(section) or {}).items():
            out[name] = {"drift_class": entry.get("class"), "measured": entry.get("measured")}
    return out


def main() -> int:
    ap = standard_parser("D26 usability screen over every candidate stream", require_config=False)
    args = ap.parse_args()
    ensure_dir(OUT.parent)
    log = get_logger("usability", args.log or "logs/usability.log").info
    cls = classes()

    by_ds: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(Path("data/cache/monitored").glob("*.pkl")):
        e = pickle.loads(f.read_bytes())
        by_ds[e["dataset_id"]].append({**e, "eps": float(min(e["eps"] - 0.05 + EPS_MARGIN, 1.0))})

    rows: list[dict] = []
    for name in sorted(by_ds):
        info = cls.get(name, {})
        try:
            ds = build_validity_dataset(by_ds[name], H=H, L=L, delta=1)
        except ValueError as exc:
            rows.append({"dataset": name, **info, "usable": False, "refused": str(exc)})
            log(f"{name:18s} REFUSED: {exc}")
            continue
        n, cens = len(ds), float(ds.censoring_rate)
        usable = (n >= MIN_ORIGINS) and (0.0 < cens < MAX_CENSORING)
        rows.append({"dataset": name, **info, "n_origins": n,
                     "n_events": int(ds.event.sum()), "censoring_rate": round(cens, 4),
                     "n_streams": int(np.unique(ds.stream_id).size), "usable": bool(usable)})
        log(f"{name:18s} {str(info.get('drift_class')):10s} N={n:5d} cens={cens:.3f} "
            f"{'USABLE' if usable else 'unusable'}")

    # Candidates refused before they could be monitored. The refusal is produced by running the
    # loader, so the recorded reason is the one the code actually raises.
    from src.datasets.mechanism import load_mechanism
    for name in PRE_MONITORING_CANDIDATES:
        if any(r["dataset"] == name for r in rows):
            continue
        info = cls.get(name, {})
        try:
            load_mechanism(name)
            log(f"{name:18s} loads after all; it is no longer a pre-monitoring refusal")
            continue
        except Exception as exc:
            # The declared row count is carried as a number, not only inside the refusal message,
            # so the manuscript can quote it and the audit can trace it.
            cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
            declared = None
            for section in ("confirmatory", "confirmatory_addendum"):
                entry = (cfg.get(section) or {}).get(name)
                if entry:
                    declared = entry.get("n_instances_declared")
            rows.append({"dataset": name, **info, "usable": False,
                         "n_rows_declared": declared,
                         "refused": str(exc), "refused_before_monitoring": True})
            log(f"{name:18s} REFUSED BEFORE MONITORING: {str(exc)[:110]}")

    out = {
        "rule": f"D26: >= {MIN_ORIGINS} valid origins AND censoring strictly in "
                f"(0, {MAX_CENSORING}), at eps margin {EPS_MARGIN}",
        "note": "Computed before any PRISM-V or reference model was fitted. Reads only the "
                "structure of the estimand, never a method's accuracy.",
        "eps_margin": EPS_MARGIN, "H": H, "L": L,
        "min_origins": MIN_ORIGINS, "max_censoring": MAX_CENSORING,
        "n_candidates": len(rows),
        "n_usable": sum(1 for r in rows if r["usable"]),
        "screen": rows,
    }
    write_json(out, OUT)
    log(f"{out['n_usable']}/{out['n_candidates']} candidates support the estimand; wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
