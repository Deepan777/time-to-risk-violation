"""Canonical stream construction: one validated `Stream` type for every data source."""
from __future__ import annotations

from .stream import REQUIRED_META, Stream, Task, Window
from .synthetic import GENERATORS, RIVER_LICENCE, make_synthetic_stream

__all__ = [
    "WILDTIME_DATASETS", "WILDTIME_LICENCES", "load_huffpost", "load_yearbook",
    "wildtime_metadata",
    "Stream", "Window", "Task", "REQUIRED_META",
    "make_synthetic_stream", "GENERATORS", "RIVER_LICENCE",
]

from .wildtime import (  # noqa: E402
    WILDTIME_DATASETS, WILDTIME_LICENCES, load_huffpost, load_yearbook, wildtime_metadata,
)
