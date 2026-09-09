#!/usr/bin/env python3
"""04_prepare_datasets.py — Convert every raw source into a canonical, time-ordered stream.

STATUS: NOT YET IMPLEMENTED. The research-design phase is complete; implementation begins with
Phase II of the implementation plan. This file is the executable specification, so that
no methodological decision has to be invented later. It intentionally refuses to run rather than
producing placeholder output, because fabricated results are prohibited by the project's integrity
rules.

Inputs   : data/raw/<id>/ + configs/datasets/<id>.yaml
Outputs  : data/processed/<id>/stream.parquet, data/metadata/<id>.json (rows, span, licence, checksum)
Main API : src.datasets.loaders.load_stream; src.datasets.windowing.windows

Unit tests that must pass before this script is considered done:
  - timestamps strictly increasing
  - no window crosses the pre-deployment boundary
  - refuse a stream shorter than window_size*(H+2)
  - licence field present or refuse

See proposal/implementation_plan.md for the full component specification, and
proposal/go_no_go.md for the gate this stage feeds.
"""
from __future__ import annotations
import sys


def main() -> int:
    sys.exit(
        "04_prepare_datasets.py is NOT YET IMPLEMENTED.\n"
        "This is deliberate: the design phase produces no experimental output.\n"
        "See proposal/implementation_plan.md for the build order."
    )


if __name__ == "__main__":
    raise SystemExit(main())
