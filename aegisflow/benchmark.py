"""Phase-1 experiment: clean -> sample -> split -> train 5 baselines -> evaluate.

Used by scripts/run_phase1.py and the Colab notebook. Everything the run
did (config, cleaning counts, sample sizes, metrics) is written to one JSON
file so every number in the report can be traced back to code output.
"""
from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from aegisflow.config import BENIGN_LABEL, Phase1Config, set_global_seed
from aegisflow.metrics import evaluate_classifier
from aegisflow.models import balanced_sample_weight, build_model, fit_model
from aegisflow.preprocessing import (
    LABEL_COL,
    FeaturePreprocessor,
    apply_smote,
    clean_dataframe,
    load_csv_folder,
    normalise_columns,
    split_data,
    stratified_sample,
)

log = logging.getLogger(__name__)


def _jsonable(obj):
    """Make numpy scalars / Paths serialisable for json.dump."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def prepare_data(raw: pd.DataFrame, cfg: Phase1Config) -> dict:
    """Clean, sample and split a raw dataframe. Returns splits + reports."""
    # normalise_columns is idempotent, so already-loaded data is unaffected.
    clean, report = clean_dataframe(normalise_columns(raw), cfg)
    sampled = stratified_sample(
        clean, cfg.sample_frac, cfg.min_per_class, cfg.max_per_class, cfg.seed
    )
    train, val, test = split_data(sampled, cfg.val_size, cfg.test_size, cfg.seed)
    return {
        "train": train,
        "val": val,
        "test": test,
        "cleaning": report.to_dict(),
        "sampling": {
            "rows_after_cleaning": len(clean),
            "rows_after_sampling": len(sampled),
            "class_counts_after_sampling": sampled[LABEL_COL].value_counts().to_dict(),
            "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        },
    }


def run_phase1(
    raw: pd.DataFrame,
    dataset_name: str,
    cfg: Phase1Config,
    save_artifacts: bool = True,
) -> dict:
    """Run the full Phase-1 benchmark on one dataset version.

    Args:
        raw: the dataframe from load_csv_folder (or synthetic data in tests).
        dataset_name: e.g. "cicids2017_original" / "cicids2017_improved".
        cfg: settings.
        save_artifacts: write models, preprocessor and splits to disk for
            later phases.
    Returns:
        A dict with everything that is also written to results/phase1_<name>.json.
    """
    set_global_seed(cfg.seed)
    t_start = time.time()
    data = prepare_data(raw, cfg)
    train, val, test = data["train"], data["val"], data["test"]

    # Labels -> integers 0..K-1 (needed by XGBoost). Fitted on all classes
    # present after cleaning; every class is in train thanks to stratification.
    encoder = LabelEncoder().fit(train[LABEL_COL])
    class_names = encoder.classes_.tolist()
    if BENIGN_LABEL not in class_names:
        raise ValueError("No BENIGN class left after cleaning")
    benign_idx = class_names.index(BENIGN_LABEL)

    prep = FeaturePreprocessor().fit(train)
    X_tr, y_tr = prep.transform(train), encoder.transform(train[LABEL_COL])
    X_te, y_te = prep.transform(test), encoder.transform(test[LABEL_COL])
    log.info("Features used (%d): %s", len(prep.feature_names_), prep.feature_names_)

    # Imbalance handling on TRAIN only.
    if cfg.imbalance in {"smote", "both"}:
        X_tr, y_tr = apply_smote(X_tr, y_tr, cfg.smote_target, cfg.seed)
    weights = balanced_sample_weight(y_tr) if cfg.imbalance in {"class_weight", "both"} else None

    results: dict = {
        "dataset": dataset_name,
        "config": {k: v for k, v in asdict(cfg).items()},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "cleaning": data["cleaning"],
        "sampling": data["sampling"],
        "features": {
            "n_features": len(prep.feature_names_),
            "names": prep.feature_names_,
            "constant_columns_dropped_on_train": prep.constant_columns_,
        },
        "train_rows_after_imbalance_handling": int(len(y_tr)),
        "models": {},
    }

    art_dir = cfg.artifacts_dir / "phase1" / dataset_name
    if save_artifacts:
        art_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"preprocessor": prep, "label_encoder": encoder}, art_dir / "preprocessing.joblib")
        # Splits are reused by Phases 3-4 (calibration uses `val`).
        for name, split in (("train", train), ("val", val), ("test", test)):
            split.to_parquet(art_dir / f"{name}.parquet", index=False)

    for name in cfg.models:
        log.info("[%s] training %s on %d rows", dataset_name, name, len(y_tr))
        model = build_model(name, seed=cfg.seed, n_jobs=cfg.n_jobs)
        t0 = time.time()
        fit_model(model, X_tr, y_tr, weights)
        train_s = time.time() - t0
        metrics = evaluate_classifier(
            model, X_te, y_te, class_names, benign_idx,
            cfg.latency_batch_rows, cfg.latency_single_repeats,
        )
        metrics["train_seconds"] = train_s
        results["models"][name] = metrics
        log.info("[%s] %s macro-F1=%.4f FPR=%.4f", dataset_name, name,
                 metrics["macro_f1"], metrics["false_positive_rate"])
        if save_artifacts:
            joblib.dump(model, art_dir / f"{name}.joblib")

    results["total_seconds"] = time.time() - t_start
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.results_dir / f"phase1_{dataset_name}.json"
    out.write_text(json.dumps(results, indent=2, default=_jsonable))
    log.info("Wrote %s", out)
    return results


def run_from_folder(folder: str | Path, dataset_name: str, cfg: Phase1Config) -> dict:
    """Convenience wrapper: load CSVs from a folder, then run_phase1."""
    raw = load_csv_folder(folder, cfg.max_rows_per_file)
    return run_phase1(raw, dataset_name, cfg)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def summary_table(results: dict) -> pd.DataFrame:
    """One row per model with the headline numbers."""
    rows = []
    for name, m in results["models"].items():
        rows.append(
            {
                "dataset": results["dataset"],
                "model": name,
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "pr_auc_macro": m["pr_auc_macro"],
                "fpr": m["false_positive_rate"],
                "attack_detection_rate": m["attack_detection_rate"],
                "batch_ms_per_flow": m["latency"]["batch_ms_per_flow"],
                "single_flow_ms": m["latency"]["single_flow_ms"],
                "size_mb": m["model_size_mb"],
                "train_s": m["train_seconds"],
            }
        )
    return pd.DataFrame(rows)


def per_class_f1_table(results: dict) -> pd.DataFrame:
    """Rows = classes, columns = models, values = F1."""
    return pd.DataFrame(
        {name: {c: v["f1"] for c, v in m["per_class"].items()} for name, m in results["models"].items()}
    )


def compare_versions(original: dict, improved: dict) -> pd.DataFrame:
    """Side-by-side macro-F1 / FPR for original vs corrected CIC-IDS2017.

    Note: the class sets can differ (e.g. 'Attempted' relabelling changes
    what counts as an attack), so per-class numbers are compared only for
    classes present in both versions.
    """
    a = summary_table(original).set_index("model")
    b = summary_table(improved).set_index("model")
    cols = ["macro_f1", "pr_auc_macro", "fpr", "attack_detection_rate"]
    out = a[cols].join(b[cols], lsuffix="_original", rsuffix="_improved", how="inner")
    out["macro_f1_delta"] = out["macro_f1_improved"] - out["macro_f1_original"]
    out["fpr_delta"] = out["fpr_improved"] - out["fpr_original"]
    return out
