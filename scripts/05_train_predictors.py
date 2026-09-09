#!/usr/bin/env python3
"""05_train_predictors.py — Fit each base predictor on the pre-deployment segment and freeze it.

STATUS: NOT YET IMPLEMENTED. The research-design phase is complete; implementation begins with
Phase II of the implementation plan. This file is the executable specification, so that
no methodological decision has to be invented later. It intentionally refuses to run rather than
producing placeholder output, because fabricated results are prohibited by the project's integrity
rules.

Inputs   : data/processed/<id>/stream.parquet + configs/predictors/<id>.yaml
Outputs  : frozen model artefacts + results/raw/predictors.json
Main API : src.predictors.base.train_predictor

Unit tests that must pass before this script is considered done:
  - deterministic under a fixed seed
  - deployment segment never read during fit
  - embed() dimensionality stable
  - skip inapplicable model families with a logged reason

See proposal/implementation_plan.md for the full component specification, and
proposal/go_no_go.md for the gate this stage feeds.
"""
from __future__ import annotations
import sys


def main() -> int:
    sys.exit(
        "05_train_predictors.py is NOT YET IMPLEMENTED.\n"
        "This is deliberate: the design phase produces no experimental output.\n"
        "See proposal/implementation_plan.md for the build order."
    )


if __name__ == "__main__":
    raise SystemExit(main())
