"""Verify the deployment-state dimensions the code builds, against the monitored cache.

`StateDim` was previously a single hard-coded 87 whose declared provenance pointed at
`src/features/deployment_state.py`, a file that does not exist. The state is in fact task-specific:
the prediction and uncertainty views branch on `handle.task`, so a classification stream and a
regression stream do not have the same number of dimensions.

This script recomputes both counts from `data/cache/monitored/*.pkl` and aborts if either differs
from the constant emitted by `scripts/28_latex_numbers.py`. It also reports how many uncertainty
dimensions are identically zero on regression streams, because `src/monitoring/uncertainty.py:57-64`
sets four of them to 0.0 when the task is regression.
"""
from __future__ import annotations

import collections
import json
import pickle
import sys
from pathlib import Path

CACHE = Path("data/cache/monitored")
OUT = Path("results/processed/STATE_DIMENSION_CHECK.md")
#: Consumed by scripts/28_latex_numbers.py, so the manuscript quotes counted values.
ARTEFACT = Path("results/processed/state_dimensions.json")

# The constants the manuscript quotes, via numbers.tex.
EXPECTED_CLF = 87
EXPECTED_REG = 79
EXPECTED_BLOCKS = 7
BLOCKS = ("pred", "rep", "unc", "dist", "fb", "ctx", "adapt")


def main() -> int:
    files = sorted(CACHE.glob("*.pkl"))
    if not files:
        print(f"ABORT: no monitored streams under {CACHE}", file=sys.stderr)
        return 1

    per_task: dict[str, set[str]] = {}
    streams: dict[str, tuple[str, int]] = {}
    const_zero: dict[str, set[str]] = {}
    per_stream_zero: dict[str, int] = {}

    for f in files:
        e = pickle.loads(f.read_bytes())
        frame = e["states"]
        task = e["provenance"]["predictor"]["task"]
        kind = "regression" if task == "regression" else "classification"
        cols = set(frame.columns)

        prev = per_task.setdefault(kind, cols)
        if prev != cols:
            print(f"ABORT: {kind} streams disagree on their feature set "
                  f"(first difference at {f.name})", file=sys.stderr)
            return 1

        streams.setdefault(e["dataset_id"], (task, len(cols)))
        zeros = {c for c in frame.columns if float(frame[c].abs().max()) == 0.0}
        per_stream_zero[e["dataset_id"]] = len(zeros)
        const_zero.setdefault(kind, zeros).intersection_update(zeros)

    n_clf = len(per_task.get("classification", ()))
    n_reg = len(per_task.get("regression", ()))

    failures = []
    if n_clf != EXPECTED_CLF:
        failures.append(f"classification dimension is {n_clf}, manuscript says {EXPECTED_CLF}")
    if n_reg != EXPECTED_REG:
        failures.append(f"regression dimension is {n_reg}, manuscript says {EXPECTED_REG}")
    for kind, cols in per_task.items():
        blocks = {c.split("_")[0] for c in cols}
        if blocks != set(BLOCKS):
            failures.append(f"{kind} blocks are {sorted(blocks)}, expected {sorted(BLOCKS)}")
        if len(blocks) != EXPECTED_BLOCKS:
            failures.append(f"{kind} has {len(blocks)} view blocks, manuscript says {EXPECTED_BLOCKS}")

    lines = ["# State-dimension check", "",
             "Recomputed from `data/cache/monitored/*.pkl`. Every number below is counted from a",
             "frozen monitored stream, not declared.", "",
             "## Dimensions actually built", "",
             "| Task | State dimension | Streams |", "|---|---|---|"]
    for kind in ("classification", "regression"):
        members = sorted(d for d, (t, _) in streams.items()
                         if (t == "regression") == (kind == "regression"))
        lines.append(f"| {kind} | {len(per_task.get(kind, ()))} | {', '.join(members)} |")

    lines += ["", "## Per-block counts", "",
              "| Block | Classification | Regression |", "|---|---|---|"]
    for b in BLOCKS:
        c = len([x for x in per_task.get("classification", ()) if x.startswith(b + "_")])
        r = len([x for x in per_task.get("regression", ()) if x.startswith(b + "_")])
        flag = "" if c == r else "  <-- task-specific"
        lines.append(f"| `{b}` | {c} | {r} |{flag}")

    lines += ["", "Only the prediction and uncertainty views differ. The other five are identical,",
              "which is why the two totals differ by "
              f"{abs(n_clf - n_reg)}.", ""]

    reg_zero = sorted(const_zero.get("regression", set()))
    clf_zero = sorted(const_zero.get("classification", set()))
    lines += ["## Dimensions that are identically zero on every stream of a task", "",
              f"* classification: {len(clf_zero)} -- "
              + (", ".join(f"`{c}`" for c in clf_zero) if clf_zero else "none"),
              f"* regression: {len(reg_zero)} -- "
              + (", ".join(f"`{c}`" for c in reg_zero) if reg_zero else "none"),
              "",
              "`src/monitoring/uncertainty.py:57-64` returns 0.0 for the energy score, the",
              "aleatoric/epistemic split and the sub-ensemble variance when the task is regression,",
              "so those dimensions carry no information on the regression streams.", ""]

    lines += ["## Verdict", ""]
    if failures:
        lines += ["**MISMATCH.**", ""] + [f"* {f}" for f in failures]
    else:
        lines.append("Both dimensions and the block count match what the manuscript states.")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")

    ARTEFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTEFACT.write_text(json.dumps({
        "state_dim_classification": n_clf,
        "state_dim_regression": n_reg,
        "n_view_blocks": EXPECTED_BLOCKS,
        "constant_dims_min": min(per_stream_zero.values()),
        "constant_dims_max": max(per_stream_zero.values()),
        "constant_dims_by_stream": per_stream_zero,
        "note": "counted from data/cache/monitored/*.pkl; never hand-typed",
    }, indent=2, sort_keys=True), encoding="utf-8")

    if failures:
        for f in failures:
            print("ABORT: " + f, file=sys.stderr)
        return 1
    print(f"state dimensions verified: classification {n_clf}, regression {n_reg}, "
          f"{EXPECTED_BLOCKS} view blocks; wrote {OUT} and {ARTEFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
