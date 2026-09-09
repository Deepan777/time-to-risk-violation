"""Typed configuration loading and resolution.

Every run resolves its config to a single serialised dict written next to the results
(implementation_plan.md, preamble). Two rules are enforced here rather than by convention:

* `scale` is mandatory and must be one of smoke / dev / full. It travels with every result record,
  and `scripts/14_generate_tables_figures.py` refuses any record whose scale is not `full`.
* the resolved dict is hashed, so a table can be traced back to the exact configuration that
  produced it even if the YAML on disk later changes.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

__all__ = ["Scale", "SCALES", "Config", "load_config", "deep_merge"]

Scale = Literal["smoke", "dev", "full"]
SCALES: tuple[str, ...] = get_args(Scale)

_MAX_EXTENDS_DEPTH = 16


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge. `override` wins at every leaf. Neither input is mutated."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass(frozen=True)
class Config:
    """A fully resolved configuration. Immutable: a run cannot edit its own config mid-flight."""

    data: dict
    source_path: Path | None = None

    # -- access ---------------------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """`cfg.get("model.encoder.hidden")`. Returns `default` if any level is missing."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """As `get`, but raises if absent. Use for anything whose absence would silently change
        the science (thresholds, horizons, split boundaries)."""
        sentinel = object()
        val = self.get(dotted, sentinel)
        if val is sentinel:
            raise KeyError(
                f"required config key '{dotted}' is missing"
                + (f" (source: {self.source_path})" if self.source_path else "")
            )
        return val

    def __getitem__(self, dotted: str) -> Any:
        return self.require(dotted)

    def __contains__(self, dotted: str) -> bool:
        sentinel = object()
        return self.get(dotted, sentinel) is not sentinel

    # -- identity -------------------------------------------------------------------------
    @property
    def scale(self) -> str:
        s = self.require("scale")
        if s not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}, got {s!r}")
        return str(s)

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)

    def canonical_json(self) -> str:
        return json.dumps(self.data, sort_keys=True, separators=(",", ":"), default=str)

    def hash(self) -> str:
        """sha256 of the canonical form. Recorded in every manifest."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def merged(self, override: dict) -> "Config":
        return Config(deep_merge(self.data, override), self.source_path)


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Load a YAML config, resolving an optional `extends:` chain, then apply `overrides`.

    `extends` is resolved relative to the extending file's directory, so a config tree can be
    moved without rewriting paths. Cycles are caught by a depth cap rather than left to recurse.
    """
    path = Path(path)
    data = _load_with_extends(path, depth=0)
    if overrides:
        data = deep_merge(data, overrides)

    if "scale" not in data:
        raise KeyError(
            f"config {path} does not declare `scale`. Every config must declare one of "
            f"{SCALES}: results carry their scale, and only `full` may reach the manuscript."
        )
    if data["scale"] not in SCALES:
        raise ValueError(f"config {path}: scale must be one of {SCALES}, got {data['scale']!r}")
    return Config(data, path)


def _load_with_extends(path: Path, depth: int) -> dict:
    if depth > _MAX_EXTENDS_DEPTH:
        raise RecursionError(f"`extends` chain deeper than {_MAX_EXTENDS_DEPTH} at {path}; "
                             "this is almost certainly a cycle")
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise TypeError(f"config {path} must contain a mapping at the top level, got "
                        f"{type(loaded).__name__}")

    parent_ref = loaded.pop("extends", None)
    if parent_ref is None:
        return loaded

    parent_path = (path.parent / str(parent_ref)).resolve()
    parent = _load_with_extends(parent_path, depth + 1)
    return deep_merge(parent, loaded)
