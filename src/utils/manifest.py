"""Run manifests — the provenance record that makes a number defensible.

experimental_protocol.md section 9 fixes the required contents: git commit hash, dataset id and
checksum, predictor id, seed, fully resolved config, hyperparameters, hardware description, start
and end timestamps, scale, all metrics, captured warnings, captured exceptions, and the output file
path. `REQUIRED_FIELDS` below is that list, and `tests/test_manifest.py` asserts against it, so the
requirement cannot drift away from the specification.

Warnings and exceptions are captured rather than allowed to scroll past: a run that emitted a
RuntimeWarning about a degenerate covariance is a different run from one that did not, and the
difference must survive into the record.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
import traceback
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .hardware import capture_hardware

__all__ = ["RunManifest", "REQUIRED_FIELDS", "git_commit", "git_is_dirty", "run_manifest"]

#: Every field experimental_protocol.md section 9 requires. Asserted by the test suite.
REQUIRED_FIELDS: tuple[str, ...] = (
    "run_id", "experiment", "git_commit", "dataset_id", "dataset_checksum",
    "predictor_id", "seed", "scale", "config", "config_hash", "hyperparameters",
    "hardware", "started_at", "ended_at", "duration_seconds", "metrics",
    "warnings", "exceptions", "output_path", "status",
)

_NOT_A_REPO = "NOT A GIT REPOSITORY"


def _git(args: list[str], repo: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, timeout=30,
            cwd=str(repo) if repo else None,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def git_commit(repo: Path | None = None) -> str:
    """Full commit hash, or an explicit marker. Never an empty string, never a guess."""
    return _git(["rev-parse", "HEAD"], repo) or _NOT_A_REPO


def git_is_dirty(repo: Path | None = None) -> bool:
    """True if the working tree has uncommitted changes. A dirty tree means the commit hash does
    not fully describe the code that ran, which the manifest must say out loud."""
    status = _git(["status", "--porcelain"], repo)
    return bool(status) if status is not None else False


@dataclass
class RunManifest:
    """One record per executed run. Serialised next to the run's output."""

    experiment: str
    dataset_id: str = "UNSET"
    dataset_checksum: str = "UNSET"
    predictor_id: str = "UNSET"
    seed: int = -1
    scale: str = "UNSET"
    config: dict = field(default_factory=dict)
    config_hash: str = "UNSET"
    hyperparameters: dict = field(default_factory=dict)
    output_path: str | None = None

    # filled in automatically
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    git_commit: str = field(default_factory=git_commit)
    git_dirty: bool = field(default_factory=git_is_dirty)
    hardware: dict = field(default_factory=capture_hardware)
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    status: str = "created"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def missing_fields(self) -> list[str]:
        d = self.to_dict()
        return [f for f in REQUIRED_FIELDS if f not in d]

    def write(self, path: str | Path) -> Path:
        """Atomic JSON write: a half-written manifest is worse than none."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True, default=str)
        tmp.replace(path)
        return path


@contextmanager
def run_manifest(
    experiment: str,
    *,
    out_path: str | Path,
    config: Any = None,
    **kwargs: Any,
) -> Iterator[RunManifest]:
    """Wrap a run so that timing, warnings and exceptions land in the manifest automatically.

    The manifest is written on the way out whether the run succeeded or failed. A crashed run that
    leaves a manifest with `status == "failed"` and the traceback attached is a recorded failure;
    a crashed run that leaves nothing is an invisible one, and section 7 of the integrity rules
    forbids invisible failures.
    """
    man = RunManifest(experiment=experiment, **kwargs)
    if config is not None:
        # accepts either a Config object or a plain dict
        man.config = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        if hasattr(config, "hash"):
            man.config_hash = config.hash()
        if man.scale == "UNSET":
            scale = man.config.get("scale")
            if scale:
                man.scale = str(scale)

    man.started_at = datetime.now(timezone.utc).isoformat()
    man.status = "running"
    t0 = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield man
            man.status = "completed"
        except BaseException as exc:
            man.status = "failed"
            man.exceptions.append("".join(traceback.format_exception(exc)).strip())
            raise
        finally:
            man.duration_seconds = round(time.perf_counter() - t0, 6)
            man.ended_at = datetime.now(timezone.utc).isoformat()
            man.warnings.extend(
                f"{w.category.__name__}: {w.message} ({Path(str(w.filename)).name}:{w.lineno})"
                for w in caught
            )
            if man.output_path is None:
                man.output_path = str(out_path)
            man.write(out_path)
