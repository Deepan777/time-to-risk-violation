#!/usr/bin/env python3
"""11_run_adaptation.py — Anticipatory versus reactive and proactive adaptation policies.

STATUS: NOT YET IMPLEMENTED. The research-design phase is complete; implementation begins with
Phase II of the implementation plan. This file is the executable specification, so that
no methodological decision has to be invented later. It intentionally refuses to run rather than
producing placeholder output, because fabricated results are prohibited by the project's integrity
rules.

Inputs   : trained validity model + action costs + configs/experiments/adaptation.yaml
Outputs  : results/raw/<run>/adaptation/*.json
Main API : src.adaptation.policy.Policy.decide; src.adaptation.actions.apply_action

Unit tests that must pass before this script is considered done:
  - no policy consults future labels
  - action set restricted per predictor family
  - cost accounting monotone
  - oracle clearly labelled as an upper reference

See proposal/implementation_plan.md for the full component specification, and
proposal/go_no_go.md for the gate this stage feeds.
"""
from __future__ import annotations
import sys


def main() -> int:
    sys.exit(
        "11_run_adaptation.py is NOT YET IMPLEMENTED.\n"
        "This is deliberate: the design phase produces no experimental output.\n"
        "See proposal/implementation_plan.md for the build order."
    )


if __name__ == "__main__":
    raise SystemExit(main())
