"""Tests for baseline models, metrics and the end-to-end Phase-1 runner."""
import json

import numpy as np
import pytest

from aegisflow.benchmark import compare_versions, run_phase1, summary_table
from aegisflow.config import Phase1Config
from aegisflow.metrics import attack_detection_rate, false_positive_rate, pr_auc
from aegisflow.models import AVAILABLE_MODELS, balanced_sample_weight, build_model, fit_model
from aegisflow.synthetic import make_synthetic_cicids


def test_fpr_and_detection_rate():
    y_true = np.array([0, 0, 0, 0, 1, 2])
    y_pred = np.array([0, 1, 0, 0, 1, 0])  # 1 of 4 benign flagged; 1 of 2 attacks missed
    assert false_positive_rate(y_true, y_pred, benign_idx=0) == pytest.approx(0.25)
    assert attack_detection_rate(y_true, y_pred, benign_idx=0) == pytest.approx(0.5)


def test_pr_auc_perfect_and_missing_class():
    y = np.array([0, 1, 0, 1])
    proba = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    macro, per = pr_auc(y, proba, 3)
    assert macro == pytest.approx(1.0)
    assert np.isnan(per[2])  # class 2 absent -> excluded


def test_balanced_weights_equalise_classes():
    y = np.array([0] * 90 + [1] * 10)
    w = balanced_sample_weight(y)
    assert w[y == 0].sum() == pytest.approx(w[y == 1].sum())


@pytest.mark.parametrize("name", AVAILABLE_MODELS)
def test_each_model_learns_easy_problem(name):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (300, 4)), rng.normal(4, 1, (60, 4))]).astype(np.float32)
    y = np.array([0] * 300 + [1] * 60)
    m = build_model(name, seed=0, n_jobs=1)
    fit_model(m, X, y, balanced_sample_weight(y))
    assert (m.predict(X) == y).mean() > 0.95
    assert m.predict_proba(X).shape == (360, 2)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("svm")


def test_run_phase1_end_to_end(tmp_path):
    cfg = Phase1Config(
        results_dir=tmp_path / "results", artifacts_dir=tmp_path / "art",
        sample_frac=1.0, min_per_class=10, min_class_samples=10,
        models=("logreg", "lightgbm"), n_jobs=1,
        latency_batch_rows=200, latency_single_repeats=5,
    )
    res = run_phase1(make_synthetic_cicids(2500, seed=1), "synthetic_test", cfg)
    out = json.loads((tmp_path / "results" / "phase1_synthetic_test.json").read_text())
    assert set(out["models"]) == {"logreg", "lightgbm"}
    m = res["models"]["lightgbm"]
    for key in ("macro_f1", "pr_auc_macro", "false_positive_rate", "confusion_matrix", "model_size_mb"):
        assert key in m
    assert m["latency"]["single_flow_ms"] > 0
    assert "dst_port" not in res["features"]["names"]
    assert (tmp_path / "art" / "phase1" / "synthetic_test" / "lightgbm.joblib").exists()
    assert (tmp_path / "art" / "phase1" / "synthetic_test" / "val.parquet").exists()
    assert len(summary_table(res)) == 2


def test_smote_mode_runs(tmp_path):
    cfg = Phase1Config(
        results_dir=tmp_path, artifacts_dir=tmp_path, sample_frac=1.0, min_per_class=10,
        min_class_samples=10, models=("random_forest",), imbalance="smote", smote_target=300,
        n_jobs=1, latency_batch_rows=100, latency_single_repeats=3,
    )
    res = run_phase1(make_synthetic_cicids(2000, seed=2), "s", cfg, save_artifacts=False)
    train_rows = res["sampling"]["split_sizes"]["train"]
    assert res["train_rows_after_imbalance_handling"] > train_rows


def test_compare_versions(tmp_path):
    cfg = Phase1Config(results_dir=tmp_path, artifacts_dir=tmp_path, sample_frac=1.0,
                       min_per_class=10, min_class_samples=10, models=("lightgbm",), n_jobs=1,
                       latency_batch_rows=100, latency_single_repeats=3)
    a = run_phase1(make_synthetic_cicids(1500, seed=3), "a", cfg, save_artifacts=False)
    b = run_phase1(make_synthetic_cicids(1500, seed=4, improved=True), "b", cfg, save_artifacts=False)
    table = compare_versions(a, b)
    assert "macro_f1_delta" in table.columns and len(table) == 1


def test_single_threaded_restores_n_jobs():
    from aegisflow.metrics import single_threaded

    m = build_model("random_forest", n_jobs=-1)
    with single_threaded(m):
        assert m.n_jobs == 1
    assert m.n_jobs == -1
