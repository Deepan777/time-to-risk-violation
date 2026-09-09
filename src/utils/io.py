"""Result and artefact IO: atomic writes, checksums, and the idempotency helper.

Every script in `scripts/` is required to be idempotent and resumable: skip work whose output
already exists unless `--force` is passed, and never make a long job restart from zero. `should_skip`
is that contract in one place, so the behaviour is identical across all fourteen stages.

Writes are atomic (write to a temp file in the same directory, then replace). A partially written
parquet or JSON that a later stage happily reads is a subtle way to get a wrong number.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "ensure_dir", "sha256_file", "sha256_bytes", "sha256_dir",
    "write_json", "read_json", "write_parquet", "read_parquet",
    "should_skip", "get_logger",
]

_CHUNK = 1 << 20  # 1 MiB


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------- checksums

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Streaming sha256. Used for the dataset provenance record required by section 7 of the
    master prompt: every dataset carries a checksum of the downloaded artefact."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: str | Path, patterns: Iterable[str] = ("**/*",)) -> str:
    """Order-stable checksum over a directory tree: hashes relative paths and file digests.

    Used where a "dataset" is a directory rather than one archive. Directories themselves and
    anything unreadable are skipped, and the relative path is included in the hash so that moving
    a file changes the digest.
    """
    root = Path(path)
    files: list[Path] = []
    for pat in patterns:
        files.extend(p for p in root.glob(pat) if p.is_file())
    h = hashlib.sha256()
    for f in sorted(set(files), key=lambda p: p.relative_to(root).as_posix()):
        h.update(f.relative_to(root).as_posix().encode("utf-8"))
        h.update(sha256_file(f).encode("ascii"))
    return h.hexdigest()


# ---------------------------------------------------------------------------- json / parquet

def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, sort_keys=True, default=str)
    tmp.replace(path)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_parquet(df, path: str | Path) -> Path:
    """Atomic parquet write. Results live in a machine-readable tree, not in prose logs."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def read_parquet(path: str | Path):
    import pandas as pd
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------- idempotency

def should_skip(output: str | Path, force: bool, *, logger: logging.Logger | None = None) -> bool:
    """True when `output` already exists and `--force` was not passed.

    Callers skip their work and return. This is the single definition of the `--force` contract;
    do not reimplement it per script.
    """
    p = Path(output)
    exists = p.exists() and (p.stat().st_size > 0 if p.is_file() else any(p.iterdir()))
    if exists and not force:
        msg = f"skipping: {p} already exists (pass --force to regenerate)"
        (logger.info if logger else print)(msg)
        return True
    return False


# ---------------------------------------------------------------------------- logging

def get_logger(name: str, logfile: str | Path | None = None, level: int = logging.INFO
               ) -> logging.Logger:
    """Console logger, optionally tee'd to a file. Long jobs run detached with output to a log
    file, so a dropped connection does not kill an experiment."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured; do not double-attach
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile is not None:
        lf = Path(logfile)
        ensure_dir(lf.parent)
        fh = logging.FileHandler(lf, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
