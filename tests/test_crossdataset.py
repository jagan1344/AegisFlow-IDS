"""Tests for Phase 2 (aegisflow.crossdataset)."""
import json

import numpy as np
import pandas as pd
import pytest

from aegisflow.config import BENIGN_LABEL, Phase1Config, Phase2Config
from aegisflow.crossdataset import (
    SHARED_FEATURES,
    DatasetNormaliser,
    evaluate_transfer,
    load_unsw_nb15,
    run_phase2,
    summary_table,
    unsw_label,
    unsw_to_shared,
)
from aegisflow.models import build_model, fit_model
from aegisflow.preprocessing import map_label, normalise_columns
from aegisflow.synthetic import make_synthetic_cicids, make_synthetic_unsw


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("DDOS attack-HOIC", "DDoS"),
        ("DDoS attacks-LOIC-HTTP", "DDoS"),
        ("DoS attacks-SlowHTTPTest", "DoS"),
        ("FTP-BruteForce", "BruteForce"),
        ("SSH-Bruteforce", "BruteForce"),
        ("Brute Force -Web", "WebAttack"),   # must NOT become BruteForce
        ("Brute Force -XSS", "WebAttack"),
        ("SQL Injection", "WebAttack"),
        ("Infilteration", "Infiltration"),  # typo in the 2018 files
        ("Botnet Ares", "Bot"),
    ],
)
def test_map_label_2018_spellings(raw, expected):
    assert map_label(raw) == expected


def test_unsw_label():
    assert unsw_label("Normal") == BENIGN_LABEL
    assert unsw_label(" DoS ") == "DoS"
    assert unsw_label("Reconnaissance") == "PortScan"
    assert unsw_label("Exploits") == "UNSW-Exploits"
    assert unsw_label(np.nan) == BENIGN_LABEL  # empty attack_cat means normal


def test_unsw_to_shared_units_and_zero_duration():
    raw = pd.DataFrame({
        "proto": ["tcp", "udp", "ospf"], "dur": [2.0, 0.0, 1.0],
        "spkts": [4, 1, 2], "dpkts": [6, 0, 2], "sbytes": [400, 60, 10], "dbytes": [600, 0, 10],
        "smean": [100, 60, 5], "dmean": [100, 0, 5], "sinpkt": [1.5, 0, 1], "dinpkt": [2.0, 0, 1],
        "swin": [255, 0, 0], "dwin": [255, 0, 0], "attack_cat": ["Normal", "DoS", "Fuzzers"],
    })
    out = unsw_to_shared(raw)
    assert list(out.columns[:len(SHARED_FEATURES)]) == SHARED_FEATURES
    assert out.loc[0, "protocol"] == 6 and out.loc[1, "protocol"] == 17 and out.loc[2, "protocol"] == 0
    assert out.loc[0, "flow_duration"] == pytest.approx(2e6)        # s -> us
    assert out.loc[0, "flow_bytes_s"] == pytest.approx(500.0)       # 1000 B / 2 s
    assert out.loc[0, "fwd_iat_mean"] == pytest.approx(1500.0)      # ms -> us
    assert out.loc[1, "flow_bytes_s"] == 0.0                        # dur 0 -> 0, not inf
    assert list(out["label"]) == [BENIGN_LABEL, "DoS", "UNSW-Fuzzers"]


def test_cic_has_all_shared_features_after_normalisation():
    df = normalise_columns(make_synthetic_cicids(50, improved=True))
    assert all(f in df.columns for f in SHARED_FEATURES)


def test_load_unsw_skips_headerless_files(tmp_path):
    make_synthetic_unsw(100).to_csv(tmp_path / "UNSW_NB15_training-set.csv", index=False)
    (tmp_path / "UNSW-NB15_1.csv").write_text("59.166.0.0,1390,149.171.126.6,53,udp\n")
    df = load_unsw_nb15(tmp_path)
    assert len(df) == 100 and "attack_cat" in df.columns


def test_dataset_normaliser():
    rng = np.random.default_rng(0)
    X = rng.lognormal(3, 1, size=(500, 3)).astype(np.float32)
    Z = DatasetNormaliser().fit(X).transform(X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-5) and np.allclose(Z.std(axis=0), 1, atol=1e-3)
    const = np.ones((10, 2), dtype=np.float32)  # constant column: no division by 0
    assert np.isfinite(DatasetNormaliser().fit(const).transform(const)).all()


def test_evaluate_transfer_counts_unknown_families_only_in_binary():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (100, 2)), rng.normal(5, 1, (100, 2))])
    y = np.array([0] * 100 + [1] * 100)
    m = build_model("lightgbm", n_jobs=1)
    fit_model(m, X, y, None)
    labels = np.array([BENIGN_LABEL] * 100 + ["DoS"] * 50 + ["UNSW-Exploits"] * 50, dtype=object)
    res = evaluate_transfer(m, X, labels, [BENIGN_LABEL, "DoS"])
    assert res["known_family_rows"] == 150             # unknown family excluded
    assert res["binary_detection_rate"] > 0.95         # but counted as attack
    assert res["per_family"]["UNSW-Exploits"]["flagged_as_attack"] > 0.95
    assert "exact_family_recall" not in res["per_family"]["UNSW-Exploits"]


def test_run_phase2_end_to_end(tmp_path):
    src_cfg = Phase1Config(sample_frac=1.0, min_per_class=10, min_class_samples=10, n_jobs=1,
                           results_dir=tmp_path, artifacts_dir=tmp_path)
    cfg = Phase2Config(source=src_cfg, results_dir=tmp_path, models=("lightgbm",),
                       target_max_per_class=2000)
    res = run_phase2(
        cfg,
        source_raw=make_synthetic_cicids(3000, seed=1, improved=True),
        cic_target_raw=make_synthetic_cicids(3000, seed=2, improved=True, labels_2018=True),
        unsw_raw=make_synthetic_unsw(2000, seed=3),
    )
    assert set(res["experiments"]) == set(cfg.experiments)
    allf = res["experiments"]["all_features"]["models"]["lightgbm"]
    assert "unsw_nb15" not in allf                      # UNSW lacks most CIC features
    sh = res["experiments"]["shared_features_norm"]["models"]["lightgbm"]
    assert {"cicids2017_test", "cicids2018_test", "unsw_nb15"} <= set(sh)
    # multi-dataset training data = source train + 2018 adapt part
    assert (res["experiments"]["multi_dataset"]["train_rows"]
            > res["experiments"]["shared_features"]["train_rows"])
    # 2018 test set is the same size for every experiment (fair comparison)
    sizes = {e: m["models"]["lightgbm"]["cicids2018_test"]["n_rows"]
             for e, m in res["experiments"].items()}
    assert len(set(sizes.values())) == 1
    table = summary_table(res)
    assert len(table) == len(cfg.experiments)
    saved = json.loads((tmp_path / "phase2.json").read_text())
    assert "data" in saved and (tmp_path / "phase2_summary.md").exists()


def test_run_phase2_without_targets(tmp_path):
    src_cfg = Phase1Config(sample_frac=1.0, min_per_class=10, min_class_samples=10, n_jobs=1)
    cfg = Phase2Config(source=src_cfg, results_dir=tmp_path, models=("lightgbm",))
    res = run_phase2(cfg, make_synthetic_cicids(2000, seed=1, improved=True), write=False)
    assert "multi_dataset" not in res["experiments"]   # needs 2018 data
    assert "cicids2017_test" in res["experiments"]["all_features"]["models"]["lightgbm"]
