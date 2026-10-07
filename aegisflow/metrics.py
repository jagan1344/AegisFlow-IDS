"""Evaluation metrics for intrusion detection.

Accuracy alone is misleading for IDS data (predicting "BENIGN" for everything
already scores ~80%), so we report:

* per-class precision / recall / F1 and macro-F1 (every class counts equally)
* confusion matrix
* PR-AUC (average precision) per class and macro: robust under imbalance
* false-positive rate: share of BENIGN flows raised as any attack
  (this is what creates alert fatigue in a SOC)
* inference latency (ms per flow) for batch and single-flow prediction
* model size (MB) when serialised
"""
from __future__ import annotations

import io
import time
from contextlib import contextmanager

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import label_binarize


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray, benign_idx: int) -> float:
    """FP / (FP + TN) where 'positive' means 'any attack'."""
    benign = y_true == benign_idx
    if benign.sum() == 0:
        return float("nan")
    return float((y_pred[benign] != benign_idx).mean())


def attack_detection_rate(y_true: np.ndarray, y_pred: np.ndarray, benign_idx: int) -> float:
    """Binary recall: share of attack flows NOT predicted as BENIGN
    (counted as detected even if the attack type is wrong)."""
    attack = y_true != benign_idx
    if attack.sum() == 0:
        return float("nan")
    return float((y_pred[attack] != benign_idx).mean())


def pr_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> tuple[float, list[float]]:
    """One-vs-rest average precision per class and its unweighted mean.

    Classes absent from y_true get NaN and are excluded from the mean.
    """
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2:  # label_binarize returns one column for 2 classes
        y_bin = np.hstack([1 - y_bin, y_bin])
    per_class = []
    for k in range(n_classes):
        if y_bin[:, k].sum() == 0:
            per_class.append(float("nan"))
        else:
            per_class.append(float(average_precision_score(y_bin[:, k], proba[:, k])))
    return float(np.nanmean(per_class)), per_class


def model_size_mb(model: BaseEstimator) -> float:
    """Size of the model when saved with joblib (what you would deploy)."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getbuffer().nbytes / 1e6


@contextmanager
def single_threaded(model: BaseEstimator):
    """Temporarily set every `n_jobs` parameter of the model to 1.

    With n_jobs=-1, a one-row prediction spends most of its time starting a
    thread pool. A real-time scorer handles one flow per call on one core,
    so single-flow latency is measured single-threaded.
    """
    keys = [k for k in model.get_params(deep=True) if k.endswith("n_jobs")]
    old = {k: model.get_params(deep=True)[k] for k in keys}
    try:
        if keys:
            model.set_params(**{k: 1 for k in keys})
        yield model
    finally:
        if keys:
            model.set_params(**old)


def measure_latency(
    model: BaseEstimator, X: np.ndarray, batch_rows: int, single_repeats: int
) -> dict[str, float]:
    """Wall-clock inference time.

    * batch: predict_proba on up to `batch_rows` rows, best of 3 runs,
      reported as ms per flow (throughput view).
    * single: predict_proba on one row at a time, median of
      `single_repeats` calls, single-threaded (real-time, one-flow-arrives
      view; includes Python call overhead, which dominates for tree ensembles).
    """
    Xb = X[: min(batch_rows, len(X))]
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        model.predict_proba(Xb)
        runs.append(time.perf_counter() - t0)
    batch_ms = min(runs) * 1000.0 / len(Xb)

    singles = []
    with single_threaded(model):
        for i in range(single_repeats):
            row = X[i % len(X)].reshape(1, -1)
            t0 = time.perf_counter()
            model.predict_proba(row)
            singles.append(time.perf_counter() - t0)
    return {
        "batch_ms_per_flow": float(batch_ms),
        "single_flow_ms": float(np.median(singles) * 1000.0),
        "batch_rows_timed": int(len(Xb)),
    }


def evaluate_classifier(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    benign_idx: int,
    batch_rows: int = 10_000,
    single_repeats: int = 200,
) -> dict:
    """Compute every Phase-1 metric for a fitted model on one split."""
    proba = model.predict_proba(X)
    y_pred = proba.argmax(axis=1)
    k = len(class_names)
    labels = list(range(k))

    report = classification_report(
        y, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    per_class = {
        name: {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": int(report[name]["support"]),
        }
        for name in class_names
    }
    pr_macro, pr_per_class = pr_auc(y, proba, k)
    for name, ap in zip(class_names, pr_per_class):
        per_class[name]["pr_auc"] = ap

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, y_pred, labels=labels, average="weighted", zero_division=0)),
        "pr_auc_macro": pr_macro,
        "false_positive_rate": false_positive_rate(y, y_pred, benign_idx),
        "attack_detection_rate": attack_detection_rate(y, y_pred, benign_idx),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "class_names": list(class_names),
        "latency": measure_latency(model, X, batch_rows, single_repeats),
        "model_size_mb": model_size_mb(model),
    }
