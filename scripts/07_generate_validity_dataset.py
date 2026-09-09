#!/usr/bin/env python3
"""07_generate_validity_dataset.py — Build deployment-state tables and the censored validity dataset.

STATUS: NOT YET IMPLEMENTED. The research-design phase is complete; implementation begins with
Phase II of the implementation plan. This file is the executable specification, so that
no methodological decision has to be invented later. It intentionally refuses to run rather than
producing placeholder output, because fabricated results are prohibited by the project's integrity
rules.

Inputs   : streams + frozen predictors + epsilon + H
Outputs  : results/raw/<run>/state.parquet, data/processed/<run>/validity_dataset.parquet
Main API : src.monitoring.assembler.build_state; src.validity.targets.build_targets

Unit tests that must pass before this script is considered done:
  - no future column reaches the feature tensor (schema assertion)
  - s_fb is masked for windows younger than the label delay
  - origins with R_t > epsilon are excluded
  - censoring flags correct on hand-built fixtures

See proposal/implementation_plan.md for the full component specification, and
proposal/go_no_go.md for the gate this stage feeds.
"""
from __future__ import annotations
import sys


def main() -> int:
    sys.exit(
        "07_generate_validity_dataset.py is NOT YET IMPLEMENTED.\n"
        "This is deliberate: the design phase produces no experimental output.\n"
        "See proposal/implementation_plan.md for the build order."
    )


if __name__ == "__main__":
    raise SystemExit(main())
