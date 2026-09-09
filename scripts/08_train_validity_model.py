#!/usr/bin/env python3
"""08_train_validity_model.py — Train the encoder plus risk and hazard heads, then calibrate.

STATUS: NOT YET IMPLEMENTED. The research-design phase is complete; implementation begins with
Phase II of the implementation plan. This file is the executable specification, so that
no methodological decision has to be invented later. It intentionally refuses to run rather than
producing placeholder output, because fabricated results are prohibited by the project's integrity
rules.

Inputs   : validity_dataset.parquet + configs/validity/<id>.yaml
Outputs  : checkpoints, calibration objects, results/raw/<run>/train_log.json
Main API : src.validity.model.PrismV.fit

Unit tests that must pass before this script is considered done:
  - survival curve monotone non-increasing
  - constant-hazard fixture recovered within tolerance
  - refuse to fit below the minimum uncensored-event count
  - hyperparameters selected on the calibration split only

See proposal/implementation_plan.md for the full component specification, and
proposal/go_no_go.md for the gate this stage feeds.
"""
from __future__ import annotations
import sys


def main() -> int:
    sys.exit(
        "08_train_validity_model.py is NOT YET IMPLEMENTED.\n"
        "This is deliberate: the design phase produces no experimental output.\n"
        "See proposal/implementation_plan.md for the build order."
    )


if __name__ == "__main__":
    raise SystemExit(main())
