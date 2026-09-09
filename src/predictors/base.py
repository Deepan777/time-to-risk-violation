"""The base predictor `f_theta`: trained once on the pre-deployment window, then frozen.

Freezing is what makes the estimand well posed. `R_t` is the risk of *a particular deployed model*
under a changing distribution; if the model kept learning, a rise in `R_t` would confound drift in
the data with drift in the model, and the time-to-violation event would no longer be a property of
the environment. Adaptation experiments (Phase VII) unfreeze the predictor deliberately and under a
named policy, and that is the only place it happens.

`PredictorHandle` is the single interface the rest of the pipeline sees, so adding a model family
costs one `train_*` function and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..datasets.stream import Stream
from ..metrics.losses import bounded_absolute_error, zero_one_loss

__all__ = ["PredictorHandle", "fit_on_pre_deployment", "EMBED_DIM"]

#: Dimensionality of the projected leaf-index representation for tree models. Fixed here so that
#: `embed()` returns a stable width across datasets and seeds, which the monitoring blocks assume.
EMBED_DIM = 32


@dataclass
class PredictorHandle:
    """A frozen predictor plus everything the monitor needs to interrogate it."""

    model: Any
    predictor_id: str
    family: str
    task: str
    feature_cols: list[str]
    seed: int
    hyperparameters: dict = field(default_factory=dict)
    fitted_on: dict = field(default_factory=dict)
    #: fixed random projection for tree leaf indices; None for models with a real penultimate layer
    embed_projection: np.ndarray | None = None
    #: frozen target scale for regression losses; None for classification
    target_scale: float | None = None
    classes_: np.ndarray | None = None

    # ------------------------------------------------------------------ inference
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.task == "regression":
            return np.asarray(self.model.predict(X), dtype=float)
        return np.asarray(self.model.predict(X)).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, n_classes) probabilities. Raises for regression, which has no such object."""
        if self.task == "regression":
            raise NotImplementedError("predict_proba is undefined for a regression predictor")
        return np.asarray(self.model.predict_proba(np.asarray(X, dtype=float)), dtype=float)

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Pre-softmax scores (logits / margins), shape (n,) or (n, n_classes).

        The energy score in the `s^unc` block needs genuine logits: computed on *normalised*
        probabilities it collapses to a constant, because logsumexp(log p) = log(1) = 0. Every
        family used here can supply a real margin, so this raises rather than silently returning
        a degenerate quantity.
        """
        X = np.asarray(X, dtype=float)
        m = self.model
        if hasattr(m, "prismv_raw"):
            return np.asarray(m.prismv_raw(X), dtype=float)
        mod = type(m).__module__
        if mod.startswith("xgboost"):
            return np.asarray(m.predict(X, output_margin=True), dtype=float)
        if mod.startswith("lightgbm"):
            return np.asarray(m.predict(X, raw_score=True), dtype=float)
        raise NotImplementedError(
            f"no raw-score access for {type(m).__name__}; the energy feature would be degenerate"
        )

    def embed(self, X: np.ndarray) -> np.ndarray:
        """A fixed-width representation of each input, for the `s^rep` monitoring block.

        For neural models this is the penultimate layer. For tree ensembles there is no such layer:
        what is returned instead is the vector of per-tree leaf indices passed through a **fixed**
        random projection. That is a legitimate representation of where the ensemble places a point
        in its own partition, but it is not the same kind of object as a learned embedding, and the
        manuscript says so rather than blurring the two.
        """
        X = np.asarray(X, dtype=float)
        if hasattr(self.model, "prismv_embed"):          # neural families provide their own
            return np.asarray(self.model.prismv_embed(X), dtype=float)
        if self.embed_projection is None:
            raise RuntimeError(f"predictor {self.predictor_id!r} has no embedding mechanism")
        leaves = self._leaf_indices(X).astype(float)
        return leaves @ self.embed_projection

    def _leaf_indices(self, X: np.ndarray) -> np.ndarray:
        m = self.model
        if hasattr(m, "apply"):                           # sklearn / lightgbm / xgboost sklearn API
            out = np.asarray(m.apply(X))
            return out.reshape(out.shape[0], -1)
        if hasattr(m, "predict") and hasattr(m, "get_booster"):   # xgboost fallback
            import xgboost as xgb
            return np.asarray(m.get_booster().predict(xgb.DMatrix(X), pred_leaf=True))
        raise RuntimeError(f"cannot extract leaf indices from {type(m).__name__}")

    # ------------------------------------------------------------------ risk
    def per_sample_loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The bounded loss `l` of equation (1), evaluated per sample."""
        y = np.asarray(y)
        if self.task == "regression":
            if self.target_scale is None:
                raise RuntimeError("regression predictor has no frozen target scale")
            return bounded_absolute_error(y, self.predict(X), self.target_scale)
        return zero_one_loss(y.astype(int), self.predict(X))

    def params(self) -> dict:
        """Provenance for the run manifest."""
        return {
            "predictor_id": self.predictor_id,
            "family": self.family,
            "task": self.task,
            "seed": self.seed,
            "n_features": len(self.feature_cols),
            "hyperparameters": dict(self.hyperparameters),
            "fitted_on": dict(self.fitted_on),
            "target_scale": self.target_scale,
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"PredictorHandle(id={self.predictor_id!r}, family={self.family!r}, task={self.task!r})"


def fit_on_pre_deployment(
    stream: Stream,
    *,
    family: str,
    seed: int,
    pre_deployment_frac: float,
    window_size: int,
    hyperparameters: dict | None = None,
) -> tuple[PredictorHandle, Stream, Stream]:
    """Split off the pre-deployment segment, fit there, and return (handle, pre, deployment).

    The split happens inside this function and the deployment segment is never passed to the
    trainer. That is a structural guarantee rather than a convention, and
    `tests/test_predictor_isolation.py` pins it by poisoning the deployment rows and checking the
    fitted model is bit-identical.
    """
    pre, dep = stream.split_deployment(pre_deployment_frac, window_size=window_size)

    if family in ("cnn", "vision"):
        from .vision import train_vision
        handle = train_vision(pre, seed=seed, hyperparameters=hyperparameters or {})
    else:
        from .tabular import train_tabular
        handle = train_tabular(pre, family=family, seed=seed,
                               hyperparameters=hyperparameters or {})
    handle.fitted_on = {
        "dataset_id": stream.dataset_id,
        "segment": "pre_deployment",
        "n_rows": pre.n_rows,
        "row_range": [0, pre.n_rows],
        "pre_deployment_frac": float(pre_deployment_frac),
        "window_size": int(window_size),
    }
    return handle, pre, dep
