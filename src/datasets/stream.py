"""The canonical stream type and its windowing.

Every raw source — synthetic generator, Wild-Time benchmark, tabular stream — is reduced to a
`Stream`: a time-ordered table with a declared feature set, target, task type, and provenance.
Everything downstream (predictors, monitoring, targets) sees only this type, so a new dataset costs
one loader and nothing else.

The validation in `__post_init__` is deliberately unforgiving. implementation_plan.md section 1
requires refusal, not repair, for: unsorted or duplicate timestamps, a stream too short to contain a
horizon, and a missing licence. Each of those, if silently patched, produces a plausible number
that means nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd

__all__ = ["Task", "Window", "Stream", "REQUIRED_META"]

Task = Literal["binary", "multiclass", "regression"]

#: Provenance the master prompt (section 7) requires for every dataset. A stream that cannot state
#: its licence is dropped, not assumed to be permissive.
REQUIRED_META: tuple[str, ...] = ("dataset_id", "licence", "source_url")


@dataclass(frozen=True)
class Window:
    """One evaluation window: the unit at which risk is defined."""

    index: int          # window number within the stream, 0-based
    start: int          # row index, inclusive
    stop: int           # row index, exclusive
    X: np.ndarray       # (n, d) features
    y: np.ndarray       # (n,) targets
    t_start: Any        # timestamp of the first row
    t_stop: Any         # timestamp of the last row

    def __len__(self) -> int:
        return self.stop - self.start


@dataclass
class Stream:
    """A validated, time-ordered stream."""

    df: pd.DataFrame
    time_col: str
    feature_cols: list[str]
    target_col: str
    task: Task
    meta: dict = field(default_factory=dict)
    #: Optional dense payload for streams whose inputs are not tabular columns (images).
    #: Rows align 1:1 with `df`. `X()` returns it flattened so every downstream block keeps
    #: working on an (n, d) matrix; the original shape lives in `meta["input_shape"]` for
    #: predictors that need it.
    array: np.ndarray | None = None

    # ------------------------------------------------------------------ validation
    def __post_init__(self) -> None:
        missing_meta = [k for k in REQUIRED_META if not self.meta.get(k)]
        if missing_meta:
            raise ValueError(
                f"stream is missing required provenance {missing_meta}. Section 7 of the master "
                "prompt: if a licence cannot be confirmed, the dataset is dropped, not assumed."
            )

        if self.array is not None and self.array.shape[0] != len(self.df):
            raise ValueError(
                f"array has {self.array.shape[0]} rows but the frame has {len(self.df)}")

        for col in (self.time_col, self.target_col, *self.feature_cols):
            if col not in self.df.columns:
                raise ValueError(f"column {col!r} not present in the stream frame")

        if self.task not in ("binary", "multiclass", "regression"):
            raise ValueError(f"unknown task {self.task!r}")

        t = self.df[self.time_col].to_numpy()
        if t.size == 0:
            raise ValueError("stream is empty")
        order = np.argsort(t, kind="stable")
        if not np.array_equal(order, np.arange(t.size)):
            raise ValueError(
                f"stream {self.meta['dataset_id']!r} is not sorted by {self.time_col!r}. "
                "Sort at load time and record the tie-breaking rule; never sort silently here, and "
                "never shuffle a temporal stream."
            )
        diffs = np.diff(t)
        if diffs.size and not np.all(diffs > 0):
            n_dup = int(np.sum(diffs <= 0))
            raise ValueError(
                f"stream {self.meta['dataset_id']!r} has {n_dup} non-increasing timestamp step(s). "
                "The canonical stream requires a strictly increasing time index; resolve ties in "
                "the loader with a documented rule."
            )

        # feature/target dtypes must be numeric: the monitoring blocks compute moments on them
        bad = [c for c in self.feature_cols
               if not pd.api.types.is_numeric_dtype(self.df[c])]
        if bad:
            raise ValueError(f"non-numeric feature columns {bad}; encode them in the loader")

    # ------------------------------------------------------------------ accessors
    @property
    def dataset_id(self) -> str:
        return str(self.meta["dataset_id"])

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    def X(self) -> np.ndarray:
        if self.array is not None:
            return self.array.reshape(self.array.shape[0], -1).astype(np.float32)
        return self.df[self.feature_cols].to_numpy(dtype=float)

    def y(self) -> np.ndarray:
        arr = self.df[self.target_col].to_numpy()
        return arr.astype(float) if self.task == "regression" else arr.astype(int)

    def timestamps(self) -> np.ndarray:
        return self.df[self.time_col].to_numpy()

    # ------------------------------------------------------------------ windowing
    def n_windows(self, size: int, stride: int) -> int:
        if size < 1 or stride < 1:
            raise ValueError("window size and stride must be >= 1")
        if self.n_rows < size:
            return 0
        return 1 + (self.n_rows - size) // stride

    def assert_long_enough(self, size: int, stride: int, H: int) -> None:
        """Refuse a stream that cannot contain a horizon.

        implementation_plan.md section 1: a stream shorter than `window_size * (H + 2)` is refused
        with an explicit message, because no origin in it has H windows of future to observe.
        """
        need_rows = size * (H + 2)
        if self.n_rows < need_rows:
            raise ValueError(
                f"stream {self.dataset_id!r} has {self.n_rows} rows but needs at least "
                f"{need_rows} (= window_size {size} * (H {H} + 2)) to contain a horizon. "
                "The stream cannot support this H; shorten H or drop the dataset."
            )
        if self.n_windows(size, stride) < H + 2:
            raise ValueError(
                f"stream {self.dataset_id!r} yields {self.n_windows(size, stride)} windows at "
                f"size={size}, stride={stride}, but needs at least H+2 = {H + 2}."
            )

    def windows(self, size: int, stride: int, *, start: int = 0, stop: int | None = None
                ) -> Iterator[Window]:
        """Emit fixed-size windows over `[start, stop)`.

        Windows never straddle the `[start, stop)` boundary, which is how the pre-deployment /
        deployment split is kept clean: a window that spanned it would mix data the predictor was
        fitted on with data it was not.
        """
        if size < 1 or stride < 1:
            raise ValueError("window size and stride must be >= 1")
        stop = self.n_rows if stop is None else stop
        if not (0 <= start <= stop <= self.n_rows):
            raise ValueError(f"invalid window range [{start}, {stop}) for {self.n_rows} rows")

        X_all = self.X()
        y_all = self.y()
        t_all = self.timestamps()

        idx = 0
        s = start
        while s + size <= stop:
            e = s + size
            yield Window(index=idx, start=s, stop=e,
                         X=X_all[s:e], y=y_all[s:e],
                         t_start=t_all[s], t_stop=t_all[e - 1])
            idx += 1
            s += stride

    # ------------------------------------------------------------------ deployment split
    def head_rows(self, n: int) -> "Stream":
        """A new Stream containing only the first `n` rows, metadata preserved.

        Used by the temporal-leakage suite: recomputing an early window inside a truncated stream
        must reproduce exactly the value it had inside the full stream. Any state feature that
        could see past its own window fails that comparison.
        """
        if n < 1 or n > self.n_rows:
            raise ValueError(f"n must lie in [1, {self.n_rows}], got {n}")
        return Stream(df=self.df.iloc[:n].copy(), time_col=self.time_col,
                      feature_cols=list(self.feature_cols), target_col=self.target_col,
                      task=self.task, meta=dict(self.meta),
                      array=None if self.array is None else self.array[:n])

    def head_windows(self, k: int, window_size: int) -> "Stream":
        """The first `k` whole windows, as a Stream."""
        return self.head_rows(int(k) * int(window_size))

    def pre_deployment_boundary(self, frac: float, *, window_size: int) -> int:
        """Row index where deployment begins, snapped down to a whole number of windows.

        Snapping matters: if the boundary fell inside a window, the first deployment window would
        contain rows the predictor had already been fitted on.
        """
        if not 0.0 < frac < 1.0:
            raise ValueError(f"pre_deployment_frac must be in (0, 1), got {frac}")
        raw = int(self.n_rows * frac)
        snapped = (raw // window_size) * window_size
        if snapped < window_size:
            raise ValueError(
                f"pre_deployment_frac={frac} gives {snapped} rows, less than one window "
                f"({window_size}); the reference statistics would have nothing to be fitted on."
            )
        if snapped >= self.n_rows:
            raise ValueError("pre-deployment segment would consume the entire stream")
        return snapped

    def split_deployment(self, frac: float, *, window_size: int) -> tuple["Stream", "Stream"]:
        """(pre_deployment, deployment) as two Streams sharing this stream's provenance."""
        b = self.pre_deployment_boundary(frac, window_size=window_size)
        pre = self.df.iloc[:b].reset_index(drop=True)
        dep = self.df.iloc[b:].reset_index(drop=True)
        common = dict(time_col=self.time_col, feature_cols=list(self.feature_cols),
                      target_col=self.target_col, task=self.task)
        arr_pre = None if self.array is None else self.array[:b]
        arr_dep = None if self.array is None else self.array[b:]
        return (
            Stream(df=pre, meta={**self.meta, "segment": "pre_deployment"},
                   array=arr_pre, **common),
            Stream(df=dep, meta={**self.meta, "segment": "deployment"},
                   array=arr_dep, **common),
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"Stream(id={self.dataset_id!r}, rows={self.n_rows}, "
                f"features={self.n_features}, task={self.task!r})")
