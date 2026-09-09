#!/usr/bin/env python3
"""12_run_ablations.py — Full ablation grid over state blocks, encoders, objectives and horizons.

STATUS: NOT YET IMPLEMENTED. The research-design phase is complete; implementation begins with
Phase II of the implementation plan. This file is the executable specification, so that
no methodological decision has to be invented later. It intentionally refuses to run rather than
producing placeholder output, because fabricated results are prohibited by the project's integrity
rules.

Inputs   : validity_dataset.parquet + configs/experiments/ablations.yaml
Outputs  : results/raw/<run>/ablations/*.json
Main API : src.evaluation.ablation.run_grid

Unit tests that must pass before this script is considered done:
  - every configuration writes a manifest
  - removal of a block changes the feature schema, not values

See proposal/implementation_plan.md for the full component specification, and
proposal/go_no_go.md for the gate this stage feeds.
"""
from __future__ import annotations
import sys


def main() -> int:
    sys.exit(
        "12_run_ablations.py is NOT YET IMPLEMENTED.\n"
        "This is deliberate: the design phase produces no experimental output.\n"
        "See proposal/implementation_plan.md for the build order."
    )


if __name__ == "__main__":
    raise SystemExit(main())
