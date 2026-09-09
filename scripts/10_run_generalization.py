#!/usr/bin/env python3
"""10_run_generalization.py — Leave-one-shift-type-out, cross-model and cross-domain transfer.

STATUS: NOT YET IMPLEMENTED. The research-design phase is complete; implementation begins with
Phase II of the implementation plan. This file is the executable specification, so that
no methodological decision has to be invented later. It intentionally refuses to run rather than
producing placeholder output, because fabricated results are prohibited by the project's integrity
rules.

Inputs   : trained models + held-out streams
Outputs  : results/raw/<run>/generalization/*.json
Main API : src.evaluation.transfer.run_lolo

Unit tests that must pass before this script is considered done:
  - held-out group never appears in training
  - marginal Kaplan-Meier reference always reported
  - negative transfer is reported, not retried with target-domain tuning

See proposal/implementation_plan.md for the full component specification, and
proposal/go_no_go.md for the gate this stage feeds.
"""
from __future__ import annotations
import sys


def main() -> int:
    sys.exit(
        "10_run_generalization.py is NOT YET IMPLEMENTED.\n"
        "This is deliberate: the design phase produces no experimental output.\n"
        "See proposal/implementation_plan.md for the build order."
    )


if __name__ == "__main__":
    raise SystemExit(main())
