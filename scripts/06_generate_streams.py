#!/usr/bin/env python3
"""06_generate_streams.py — Generate controlled shift streams with ground-truth manifests.

STATUS: NOT YET IMPLEMENTED. The research-design phase is complete; implementation begins with
Phase II of the implementation plan. This file is the executable specification, so that
no methodological decision has to be invented later. It intentionally refuses to run rather than
producing placeholder output, because fabricated results are prohibited by the project's integrity
rules.

Inputs   : a base stream + configs/shifts/<id>.yaml
Outputs  : data/streams/<sid>/{stream.parquet, shift_manifest.json}
Main API : src.shifts.generators.generate_stream

Unit tests that must pass before this script is considered done:
  - covariate-only generators leave P(Y|X) invariant
  - conditional generators leave P(X) invariant
  - U-class streams pass the permutation independence test
  - manifest round-trips

See proposal/implementation_plan.md for the full component specification, and
proposal/go_no_go.md for the gate this stage feeds.
"""
from __future__ import annotations
import sys


def main() -> int:
    sys.exit(
        "06_generate_streams.py is NOT YET IMPLEMENTED.\n"
        "This is deliberate: the design phase produces no experimental output.\n"
        "See proposal/implementation_plan.md for the build order."
    )


if __name__ == "__main__":
    raise SystemExit(main())
