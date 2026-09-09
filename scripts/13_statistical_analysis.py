#!/usr/bin/env python3
"""13_statistical_analysis.py — Aggregate across seeds with CIs, paired tests and effect sizes.

STATUS: NOT YET IMPLEMENTED. The research-design phase is complete; implementation begins with
Phase II of the implementation plan. This file is the executable specification, so that
no methodological decision has to be invented later. It intentionally refuses to run rather than
producing placeholder output, because fabricated results are prohibited by the project's integrity
rules.

Inputs   : results/raw/**
Outputs  : results/statistical/*.json and *.csv
Main API : src.evaluation.stats.aggregate

Unit tests that must pass before this script is considered done:
  - bootstrap blocked by stream
  - Holm-Bonferroni applied within each experiment family
  - effect size reported with every p-value
  - all seeds included; selection is impossible by design

See proposal/implementation_plan.md for the full component specification, and
proposal/go_no_go.md for the gate this stage feeds.
"""
from __future__ import annotations
import sys


def main() -> int:
    sys.exit(
        "13_statistical_analysis.py is NOT YET IMPLEMENTED.\n"
        "This is deliberate: the design phase produces no experimental output.\n"
        "See proposal/implementation_plan.md for the build order."
    )


if __name__ == "__main__":
    raise SystemExit(main())
