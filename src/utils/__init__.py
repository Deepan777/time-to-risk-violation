"""Shared infrastructure: seeding, config, manifests, IO, hardware capture, leakage guards."""
from __future__ import annotations

from .cli import REPO_ROOT, standard_parser
from .config import SCALES, Config, Scale, deep_merge, load_config
from .hardware import capture_hardware, format_hardware
from .io import (
    ensure_dir, get_logger, read_json, read_parquet, sha256_bytes, sha256_dir, sha256_file,
    should_skip, write_json, write_parquet,
)
from .leakage import FORBIDDEN_EXACT, FORBIDDEN_PREFIXES, assert_no_future_columns
from .manifest import REQUIRED_FIELDS, RunManifest, git_commit, git_is_dirty, run_manifest
from .seeding import SEEDS_DEV, SEEDS_FULL, SEEDS_SMOKE, make_rng, seed_everything

__all__ = [
    "REPO_ROOT", "standard_parser",
    "Config", "Scale", "SCALES", "load_config", "deep_merge",
    "capture_hardware", "format_hardware",
    "ensure_dir", "get_logger", "read_json", "read_parquet", "sha256_bytes", "sha256_dir",
    "sha256_file", "should_skip", "write_json", "write_parquet",
    "FORBIDDEN_EXACT", "FORBIDDEN_PREFIXES", "assert_no_future_columns",
    "REQUIRED_FIELDS", "RunManifest", "git_commit", "git_is_dirty", "run_manifest",
    "SEEDS_DEV", "SEEDS_FULL", "SEEDS_SMOKE", "make_rng", "seed_everything",
]
