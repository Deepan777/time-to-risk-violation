"""Tree-ensemble base predictors (XGBoost, LightGBM) and a scikit-learn MLP.

These are the Phase-I / Tier-B / Tier-D workhorses. No claim attaches to the choice: the framework
is model-agnostic by construction, and the cross-architecture transfer experiment (E9) depends on
having several genuinely different families available.

Determinism is enforced rather than hoped for. Each family is given an explicit seed and
single-threaded histogram construction where the library's multi-threaded path is
order-dependent, because a predictor that differs run to run makes every downstream risk series
irreproducible.
"""
from __future__ import annotations

import numpy as np

from ..datasets.stream import Stream
from .base import EMBED_DIM, PredictorHandle

__all__ = ["FAMILIES", "train_tabular", "DEFAULT_HYPERPARAMETERS"]

FAMILIES: tuple[str, ...] = ("xgboost", "lightgbm", "mlp", "logreg")

#: Modest, fixed defaults. Any tuning happens on the calibration split via the shared budget in
#: `scripts/`, never here, and never on test data.
DEFAULT_HYPERPARAMETERS: dict[str, dict] = {
    "xgboost": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
                "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0},
    "lightgbm": {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.1,
                 "subsample": 0.9, "colsample_bytree": 0.9, "min_child_samples": 20},
    "mlp": {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "max_iter": 300,
            "learning_rate_init": 1e-3},
    "logreg": {"C": 1.0, "max_iter": 1000},
}


def _make_projection(leaf_width: int, seed: int) -> np.ndarray:
    """Fixed random projection from leaf indices to a stable-width representation.

    Fitted once, from the seed alone, and never refitted — the same rule that governs every other
    reference statistic in the pipeline.

    `leaf_width` must be the width the *fitted* model actually emits, not the configured number of
    trees. A multiclass gradient-boosted model grows one tree per class per round, so `apply()`
    returns `n_estimators * n_classes` columns; sizing this from `n_estimators` alone works for
    binary tasks and fails the moment a multiclass dataset appears.
    """
    rng = np.random.default_rng(seed + 999_983)
    return rng.normal(scale=1.0 / np.sqrt(max(leaf_width, 1)), size=(leaf_width, EMBED_DIM))


def _probe_leaf_width(model, X: np.ndarray) -> int:
    """Width of the leaf-index matrix the fitted model emits, measured on a few real rows."""
    probe = np.asarray(X[: min(8, len(X))], dtype=float)
    out = np.asarray(model.apply(probe))
    return int(out.reshape(out.shape[0], -1).shape[1])


def train_tabular(
    pre: Stream,
    *,
    family: str,
    seed: int,
    hyperparameters: dict | None = None,
) -> PredictorHandle:
    """Fit one tabular predictor on the pre-deployment segment and freeze it."""
    family = family.lower()
    if family not in FAMILIES:
        raise ValueError(f"unknown predictor family {family!r}; expected one of {FAMILIES}")

    hp = {**DEFAULT_HYPERPARAMETERS[family], **(hyperparameters or {})}
    X, y = pre.X(), pre.y()
    if X.shape[0] < 20:
        raise ValueError(f"pre-deployment segment has only {X.shape[0]} rows; refusing to fit")

    if pre.task != "regression" and len(np.unique(y)) < 2:
        raise ValueError(
            "pre-deployment segment contains a single class; the predictor would be degenerate "
            "and every risk value downstream would be an artefact of that, not of drift."
        )

    model, projection, target_scale = _build_and_fit(family, pre.task, X, y, hp, seed)

    return PredictorHandle(
        model=model,
        predictor_id=f"{family}_seed{seed}",
        family=family,
        task=pre.task,
        feature_cols=list(pre.feature_cols),
        seed=int(seed),
        hyperparameters=hp,
        embed_projection=projection,
        target_scale=target_scale,
        classes_=np.unique(y) if pre.task != "regression" else None,
    )


def _build_and_fit(family: str, task: str, X: np.ndarray, y: np.ndarray, hp: dict, seed: int):
    projection: np.ndarray | None = None
    target_scale: float | None = None

    if task == "regression":
        # frozen normalising scale for the bounded loss; never recomputed on deployment data
        target_scale = float(np.std(y))
        if not np.isfinite(target_scale) or target_scale <= 0:
            raise ValueError("pre-deployment target has zero variance; bounded loss is undefined")

    if family == "xgboost":
        import xgboost as xgb
        cls = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
        model = cls(**hp, random_state=seed, n_jobs=1, tree_method="hist", verbosity=0)
        model.fit(X, y)
        projection = _make_projection(_probe_leaf_width(model, X), seed)

    elif family == "lightgbm":
        import lightgbm as lgb
        cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
        model = cls(**hp, random_state=seed, n_jobs=1, verbose=-1, deterministic=True,
                    force_row_wise=True)
        model.fit(X, y)
        projection = _make_projection(_probe_leaf_width(model, X), seed)

    elif family == "mlp":
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        cls = MLPRegressor if task == "regression" else MLPClassifier
        # the scaler is fitted here, on pre-deployment data only, and travels frozen inside the
        # pipeline; it is never refitted on the deployment stream.
        model = Pipeline([("scale", StandardScaler()),
                          ("net", cls(**hp, random_state=seed))])
        model.fit(X, y)
        _attach_mlp_embedding(model)

    elif family == "logreg":
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        if task == "regression":
            model = Pipeline([("scale", StandardScaler()), ("net", LinearRegression())])
        else:
            model = Pipeline([("scale", StandardScaler()),
                              ("net", LogisticRegression(**hp, random_state=seed))])
        model.fit(X, y)
        _attach_linear_embedding(model)
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(family)

    return model, projection, target_scale


def _attach_mlp_embedding(pipeline) -> None:
    """Expose the MLP's penultimate activations and its output-layer pre-activations."""
    def _hidden(X: np.ndarray) -> np.ndarray:
        net = pipeline.named_steps["net"]
        h = pipeline.named_steps["scale"].transform(np.asarray(X, dtype=float))
        # forward through all but the output layer, replicating sklearn's ReLU MLP
        for W, b in zip(net.coefs_[:-1], net.intercepts_[:-1]):
            h = np.maximum(h @ W + b, 0.0)
        return h

    def _raw(X: np.ndarray) -> np.ndarray:
        net = pipeline.named_steps["net"]
        return _hidden(X) @ net.coefs_[-1] + net.intercepts_[-1]

    pipeline.prismv_embed = _hidden  # type: ignore[attr-defined]
    pipeline.prismv_raw = _raw       # type: ignore[attr-defined]


def _attach_linear_embedding(pipeline) -> None:
    """For a linear model the standardised inputs *are* the representation: there is no hidden
    layer, and pretending otherwise would misdescribe the model."""
    def _embed(X: np.ndarray) -> np.ndarray:
        return pipeline.named_steps["scale"].transform(np.asarray(X, dtype=float))

    def _raw(X: np.ndarray) -> np.ndarray:
        net = pipeline.named_steps["net"]
        if hasattr(net, "decision_function"):
            return np.asarray(net.decision_function(_embed(X)), dtype=float)
        return np.asarray(net.predict(_embed(X)), dtype=float)

    pipeline.prismv_embed = _embed  # type: ignore[attr-defined]
    pipeline.prismv_raw = _raw      # type: ignore[attr-defined]
