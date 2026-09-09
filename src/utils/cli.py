"""The standard script interface.

Every script in `scripts/` takes `--config`, `--seed`, `--out`, `--force`. Defining the parser once
means the fourteen stages cannot drift apart, and a resumed session can rely on the same flags
working everywhere.
"""
from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["standard_parser", "REPO_ROOT"]

#: Repository root, derived from this file's location (src/utils/cli.py -> two levels up).
REPO_ROOT = Path(__file__).resolve().parents[2]


def standard_parser(description: str, *, require_config: bool = True) -> argparse.ArgumentParser:
    """Build the common parser. Scripts add their own stage-specific flags on top."""
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=require_config,
                   help="path to the YAML config for this run")
    p.add_argument("--seed", type=int, default=None,
                   help="single seed to run; omit to run every seed the config's scale defines")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory or file; defaults to the path the config specifies")
    p.add_argument("--force", action="store_true",
                   help="regenerate outputs that already exist instead of skipping them")
    p.add_argument("--log", type=Path, default=None,
                   help="tee log output to this file (long jobs should always set it)")
    return p
