"""Deployment-state computation: the seven blocks of `s_t` and the frozen reference statistics."""
from __future__ import annotations

from .adaptation_state import adaptation_stats
from .assembler import ALL_BLOCKS, BLOCK_PREFIX, build_state, build_state_table
from .context import context_stats
from .discrepancy import discrepancy_stats
from .feedback import FEEDBACK_KEYS, feedback_stats
from .history import History
from .prediction_stats import prediction_stats
from .reference import MAX_REFERENCE_POINTS, ReferenceStats, fit_reference
from .representation_stats import representation_stats
from .uncertainty import uncertainty_stats

__all__ = [
    "ALL_BLOCKS", "BLOCK_PREFIX", "build_state", "build_state_table",
    "History", "ReferenceStats", "fit_reference", "MAX_REFERENCE_POINTS",
    "prediction_stats", "representation_stats", "uncertainty_stats", "discrepancy_stats",
    "feedback_stats", "FEEDBACK_KEYS", "context_stats", "adaptation_stats",
]
