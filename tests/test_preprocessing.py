"""Unit tests for aegisflow.preprocessing (run: pytest -q from the repo root)."""
import numpy as np
import pandas as pd
import pytest

from aegisflow.config import Phase1Config
from aegisflow.preprocessing import (
    LABEL_COL,
    FeaturePreprocessor,
    apply_smote,
    canonical_name,
    clean_dataframe,
    map_label,
    normalise_columns,
    split_data,
    stratified_sample,
)
from aegisflow.synthetic import make_synthetic_cicids


@pytest.mark.parametrize(
    "raw,expected",
    [
        (" Flow Bytes/s", "flow_bytes_s"),
        (" Total Fwd Packets", "total_fwd_packets"),
        ("Total Fwd Packet", "total_fwd_packets"),   # improved-dataset spelling
        (" Destination Port", "dst_port"),
        ("Flow ID", "flow_id"),
        (" CWE Flag Count", "cwr_flag_count"),
    ],
)
def test_canonical_name(raw, expected):
    assert canonical_name(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BENIGN", "BENIGN"),
        ("DDoS", "DDoS"),
        ("DoS Hulk", "DoS"),
        ("DoS slowloris", "DoS"),
        ("PortScan", "PortScan"),
        ("FTP-Patator", "BruteForce"),
        ("SSH-Patator", "BruteForce"),
        ("Web Attack \x96 Brute Force", "WebAttack"),  # must NOT become BruteForce
        ("Web Attack � XSS", "WebAttack"),
        ("Bot", "Bot"),
        ("Infiltration - Portscan", "Infiltration"),
        ("Heartbleed", "Heartbleed"),
        ("DoS Hulk - Attempted", "BENIGN"),
    ],
)
def test_map_label_family(raw, expected):
    assert map_label(raw, "family", attempted_as_benign=True) == expected


def test_map_label_attempted_kept_separate():
    assert map_label("DoS Hulk - Attempted", "family", attempted_as_benign=False) == "Attempted"


def test_normalise_columns_drops_duplicate_column():
    df = normalise_columns(make_synthetic_cicids(100))
    assert "fwd_header_length_1" not in df.columns
    assert "total_fwd_packets" in df.columns and "label" in df.columns


def _clean(n=2000, **cfg_kwargs):
    cfg = Phase1Config(min_class_samples=5, **cfg_kwargs)
    raw = normalise_columns(make_synthetic_cicids(n, improved=True))
    return clean_dataframe(raw, cfg)


def test_clean_removes_leakage_inf_nan_duplicates():
    df, rep = _clean()
    for col in ("flow_id", "src_ip", "timestamp", "dst_port"):
        assert col not in df.columns
    feats = [c for c in df.columns if c not in (LABEL_COL, "source_file")]
    assert np.isfinite(df[feats].to_numpy()).all()
    assert rep.inf_values_found == 5
    assert rep.rows_with_nan_or_inf_dropped >= 5
    assert rep.duplicate_rows_dropped >= 1
    assert not df.duplicated(subset=feats + [LABEL_COL]).any()
    assert rep.rows_out == len(df)


def test_keep_dst_port_option():
    df, _ = _clean(keep_dst_port=True)
    assert "dst_port" in df.columns


def test_conflicting_rows_dropped():
    cfg = Phase1Config(min_class_samples=1)
    df = pd.DataFrame({"a": [1.0, 1.0, 2.0, 3.0], "b": [0.0, 0.0, 1.0, 1.0],
                       "label": ["BENIGN", "DDoS", "BENIGN", "DDoS"]})
    out, rep = clean_dataframe(df, cfg)
    assert rep.conflicting_rows_dropped == 2
    assert len(out) == 2


def test_rare_class_dropped():
    cfg = Phase1Config(min_class_samples=3)
    df = pd.DataFrame({"a": np.arange(10, dtype=float),
                       "label": ["BENIGN"] * 8 + ["Heartbleed"] * 2})
    out, rep = clean_dataframe(df, cfg)
    assert rep.rare_classes_dropped == {"Heartbleed": 2}
    assert set(out[LABEL_COL]) == {"BENIGN"}


def test_stratified_sample_keeps_rare_classes():
    df = pd.DataFrame({"x": np.arange(1100.0), "label": ["BENIGN"] * 1000 + ["Bot"] * 100})
    s = stratified_sample(df, frac=0.1, min_per_class=50, max_per_class=None, seed=0)
    counts = s[LABEL_COL].value_counts()
    assert counts["BENIGN"] == 100       # 10% of 1000
    assert counts["Bot"] == 50           # floor applies
    s2 = stratified_sample(df, frac=0.1, min_per_class=500, max_per_class=None, seed=0)
    assert s2[LABEL_COL].value_counts()["Bot"] == 100  # whole class kept


def test_split_is_stratified_and_disjoint():
    df, _ = _clean(3000)
    df = df.reset_index(drop=True)
    df["row"] = np.arange(len(df))
    tr, va, te = split_data(df, 0.15, 0.15, seed=0)
    assert len(tr) + len(va) + len(te) == len(df)
    assert set(tr.row).isdisjoint(te.row) and set(tr.row).isdisjoint(va.row)
    assert set(te[LABEL_COL]) == set(df[LABEL_COL])


def test_feature_preprocessor_fits_on_train_only():
    tr = pd.DataFrame({"a": [1.0, 2.0], "const": [0.0, 0.0], "label": ["BENIGN", "Bot"]})
    te = pd.DataFrame({"a": [3.0], "const": [5.0], "label": ["Bot"]})
    prep = FeaturePreprocessor().fit(tr)
    assert prep.constant_columns_ == ["const"]
    X = prep.transform(te)
    assert X.shape == (1, 1) and X.dtype == np.float32


def test_smote_only_grows_minority():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3)).astype(np.float32)
    y = np.array([0] * 100 + [1] * 20)
    Xr, yr = apply_smote(X, y, target=60, seed=0)
    assert (yr == 0).sum() == 100 and (yr == 1).sum() == 60


def test_benign_load_frac_keeps_all_attacks(tmp_path):
    from aegisflow.preprocessing import load_csv_folder

    raw = make_synthetic_cicids(2000, seed=5)
    raw.to_csv(tmp_path / "day.csv", index=False)
    full = load_csv_folder(tmp_path)
    part = load_csv_folder(tmp_path, benign_load_frac=0.2, seed=0)
    n_att_full = (full[LABEL_COL] != "BENIGN").sum()
    assert (part[LABEL_COL] != "BENIGN").sum() == n_att_full
    ratio = (part[LABEL_COL] == "BENIGN").sum() / (full[LABEL_COL] == "BENIGN").sum()
    assert 0.1 < ratio < 0.3
