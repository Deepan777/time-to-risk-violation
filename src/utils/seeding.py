"""Deterministic seeding across every RNG this project touches.

Seed policy (experimental_protocol.md section 7): five independent seeds per configuration, all
reported, aggregation rule fixed in advance. Never select a seed. This module exists so that a
seed means the same thing in every stage of the pipeline.
"""
from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["seed_everything", "make_rng", "SEEDS_FULL", "SEEDS_DEV", "SEEDS_SMOKE"]

# The pre-registered seed sets. These are fixed here, in code, so that no stage can quietly
# choose a different set. `full` is the protocol's five seeds.
SEEDS_FULL: tuple[int, ...] = (0, 1, 2, 3, 4)
SEEDS_DEV: tuple[int, ...] = (0, 1)
SEEDS_SMOKE: tuple[int, ...] = (0,)


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed python, numpy and torch (CPU and CUDA), and request deterministic kernels.

    `deterministic_torch` trades throughput for reproducibility. It is on by default because a
    result that cannot be reproduced cannot be defended, and every training job in this project is
    small enough that the cost is acceptable.
    """
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # torch is optional for the pure-numpy stages
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        # cuBLAS needs this set before the first CUDA matmul to make GEMMs reproducible.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - depends on torch build
            pass


def make_rng(seed: int) -> np.random.Generator:
    """A local Generator. Prefer this to the global numpy RNG wherever a stream is generated:
    global state makes stage ordering silently affect results."""
    return np.random.default_rng(int(seed))
