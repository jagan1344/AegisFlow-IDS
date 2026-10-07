"""Baseline classifiers for Phase 1.

All five models share one interface (scikit-learn's fit / predict /
predict_proba) so the evaluation code never needs to know which is which.

Imbalance handling uses ONE mechanism for every model: per-sample weights
w_i = n_samples / (n_classes * count(class_i)) ("balanced" weights), passed to
fit(). This is equivalent to class_weight="balanced" but works for XGBoost
and the MLP too, so the comparison between models is fair.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

AVAILABLE_MODELS: tuple[str, ...] = ("logreg", "random_forest", "xgboost", "lightgbm", "mlp")


def signed_log1p(X: np.ndarray) -> np.ndarray:
    """sign(x) * log(1 + |x|).

    Flow features are extremely skewed (bytes/s ranges from 0 to 1e9). Linear
    models and neural nets train badly on such ranges, so we compress them.
    The sign is kept because a few CIC features use -1 as "not present".
    Tree models do not need this (they only compare thresholds).
    """
    return np.sign(X) * np.log1p(np.abs(X))


def _scaled(clf: BaseEstimator) -> Pipeline:
    """Wrap a scale-sensitive model as: log-compress -> standardise -> model."""
    return Pipeline(
        [
            ("log", FunctionTransformer(signed_log1p)),
            ("scale", StandardScaler()),
            ("clf", clf),
        ]
    )


def build_model(name: str, seed: int = 42, n_jobs: int = -1) -> BaseEstimator:
    """Create an untrained model by name. Hyper-parameters are modest on
    purpose: they train in minutes on Colab and are a fair baseline."""
    if name == "logreg":
        return _scaled(LogisticRegression(max_iter=1000, C=1.0, random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=25,          # caps model size; full-depth RFs reach GBs
            min_samples_leaf=2,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",    # fast histogram algorithm, CPU friendly
            eval_metric="mlogloss",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=300,
            num_leaves=63,
            learning_rate=0.1,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=-1,
        )
    if name == "mlp":
        return _scaled(
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                batch_size=512,
                learning_rate_init=1e-3,
                max_iter=50,
                early_stopping=True,   # holds out 10% of train internally
                n_iter_no_change=5,
                random_state=seed,
            )
        )
    raise ValueError(f"Unknown model {name!r}; choose from {AVAILABLE_MODELS}")


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """w_i = n / (k * count(y_i)): rare classes count as much as common ones."""
    return compute_sample_weight(class_weight="balanced", y=y)


def fit_model(
    model: BaseEstimator, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None
) -> BaseEstimator:
    """Fit any of the baseline models, routing sample weights correctly.

    A Pipeline needs the weight addressed to its final step ("clf__...").
    """
    if sample_weight is None:
        return model.fit(X, y)
    if isinstance(model, Pipeline):
        return model.fit(X, y, clf__sample_weight=sample_weight)
    return model.fit(X, y, sample_weight=sample_weight)
