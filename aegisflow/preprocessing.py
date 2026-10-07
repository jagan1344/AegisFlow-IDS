"""Data loading, cleaning, leakage removal, sampling and splitting.

Pipeline (each step is a small function so it can be unit-tested):

    load_csv_folder      read every CSV of one dataset version
    normalise_columns    unify column names between dataset versions
    clean_dataframe      labels -> families, drop leakage, NaN/inf, duplicates
    stratified_sample    shrink to fit RAM while keeping rare attacks
    split_data           stratified train / validation / test
    FeaturePreprocessor  fitted on TRAIN only: column order, constant columns
    apply_smote          optional oversampling, TRAIN only
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from aegisflow.config import BENIGN_LABEL, IDENTIFIER_COLUMNS, Phase1Config

log = logging.getLogger(__name__)

LABEL_COL = "label"

# ---------------------------------------------------------------------------
# 1. Column names
# ---------------------------------------------------------------------------
# The original CIC-IDS2017 CSVs (CICFlowMeter-V3) and the improved version
# (fixed CICFlowMeter) name some features differently, e.g.
# "Total Fwd Packets" vs "Total Fwd Packet". After canonical_name() we map
# known variants to ONE name so both versions can be compared feature by
# feature. Unmatched columns are logged, never silently used.
COLUMN_ALIASES: dict[str, str] = {
    "destination_port": "dst_port",
    "source_port": "src_port",
    "source_ip": "src_ip",
    "destination_ip": "dst_ip",
    "total_fwd_packet": "total_fwd_packets",
    "total_backward_packets": "total_bwd_packets",
    "total_length_of_fwd_packets": "total_length_fwd_packets",
    "total_length_of_fwd_packet": "total_length_fwd_packets",
    "total_length_of_bwd_packets": "total_length_bwd_packets",
    "total_length_of_bwd_packet": "total_length_bwd_packets",
    "min_packet_length": "packet_length_min",
    "max_packet_length": "packet_length_max",
    "average_packet_size": "avg_packet_size",
    "avg_fwd_segment_size": "fwd_segment_size_avg",
    "avg_bwd_segment_size": "bwd_segment_size_avg",
    "fwd_avg_bytes_bulk": "fwd_bytes_bulk_avg",
    "fwd_avg_packets_bulk": "fwd_packet_bulk_avg",
    "fwd_avg_bulk_rate": "fwd_bulk_rate_avg",
    "bwd_avg_bytes_bulk": "bwd_bytes_bulk_avg",
    "bwd_avg_packets_bulk": "bwd_packet_bulk_avg",
    "bwd_avg_bulk_rate": "bwd_bulk_rate_avg",
    "init_win_bytes_forward": "fwd_init_win_bytes",
    "init_win_bytes_backward": "bwd_init_win_bytes",
    "act_data_pkt_fwd": "fwd_act_data_pkts",
    "min_seg_size_forward": "fwd_seg_size_min",
    "cwe_flag_count": "cwr_flag_count",  # typo in CICFlowMeter-V3 ("CWE")
}

# Columns that are exact copies of another column in the original CSVs.
DUPLICATE_COLUMNS: tuple[str, ...] = ("fwd_header_length_1",)


def canonical_name(col: str) -> str:
    """Turn ' Flow Bytes/s' into 'flow_bytes_s' and apply known aliases."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()
    return COLUMN_ALIASES.get(name, name)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename all columns to canonical names and drop duplicate columns."""
    df = df.rename(columns=canonical_name)
    # If two raw names collapse to the same canonical name keep the first.
    df = df.loc[:, ~df.columns.duplicated()]
    return df.drop(columns=[c for c in DUPLICATE_COLUMNS if c in df.columns])


# ---------------------------------------------------------------------------
# 2. Labels
# ---------------------------------------------------------------------------
def clean_label_text(raw: object) -> str:
    """Fix encoding junk: 'Web Attack � Brute Force' -> 'Web Attack - Brute Force'."""
    text = re.sub(r"[^\x20-\x7e]+", "-", str(raw))
    return re.sub(r"\s+", " ", text).strip()


def map_label(raw: object, level: str = "family", attempted_as_benign: bool = True) -> str:
    """Map a raw CIC-IDS2017 label to the label used for training.

    Families (order of checks matters, e.g. 'DDoS' contains 'DoS',
    'Web Attack - Brute Force' contains 'Brute Force'):
      BENIGN, DDoS, DoS, PortScan, BruteForce, WebAttack, Bot, Infiltration,
      Heartbleed.
    """
    text = clean_label_text(raw)
    low = text.lower()
    if "attempted" in low:
        # Improved dataset only: failed/empty attack attempts.
        if attempted_as_benign:
            return BENIGN_LABEL
        return "Attempted" if level == "family" else text
    if low == "benign":
        return BENIGN_LABEL
    if level == "fine":
        return text
    if "ddos" in low:
        return "DDoS"
    if "heartbleed" in low:
        return "Heartbleed"
    if "infiltration" in low:  # includes "Infiltration - Portscan"
        return "Infiltration"
    if "web attack" in low or "xss" in low or "sql injection" in low:
        return "WebAttack"
    if "dos" in low:
        return "DoS"
    if "portscan" in low or "port scan" in low:
        return "PortScan"
    if "patator" in low or "brute force" in low:
        return "BruteForce"
    if "bot" in low:
        return "Bot"
    log.warning("Unknown label %r kept as-is", text)
    return text


# ---------------------------------------------------------------------------
# 3. Loading
# ---------------------------------------------------------------------------
def _read_one_csv(path: Path, max_rows: int | None) -> pd.DataFrame:
    """Read one CSV; the original files are not valid UTF-8, so fall back."""
    try:
        df = pd.read_csv(path, nrows=max_rows, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, nrows=max_rows, low_memory=False, encoding="latin-1")
    df = normalise_columns(df)
    # Halve memory: float64 -> float32 for every numeric feature.
    for col in df.columns:
        if col != LABEL_COL and pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
    df["source_file"] = path.name  # kept for traceability, never a feature
    return df


def load_csv_folder(folder: str | Path, max_rows_per_file: int | None = None) -> pd.DataFrame:
    """Load and concatenate every *.csv under `folder` (recursively)."""
    folder = Path(folder)
    files = sorted(folder.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {folder}. Run data/download.py first "
            "(see data/DATASETS.md)."
        )
    frames = []
    for f in files:
        log.info("Reading %s", f)
        frames.append(_read_one_csv(f, max_rows_per_file))
    df = pd.concat(frames, ignore_index=True)
    if LABEL_COL not in df.columns:
        raise KeyError(f"No 'Label' column found in {folder}; columns: {list(df.columns)[:10]}...")
    return df


# ---------------------------------------------------------------------------
# 4. Cleaning
# ---------------------------------------------------------------------------
@dataclass
class CleaningReport:
    """Counts of what every cleaning step removed (goes into the results JSON)."""

    rows_in: int = 0
    leakage_columns_dropped: list[str] = field(default_factory=list)
    non_numeric_columns_dropped: list[str] = field(default_factory=list)
    inf_values_found: int = 0
    rows_with_nan_or_inf_dropped: int = 0
    duplicate_rows_dropped: int = 0
    conflicting_rows_dropped: int = 0
    rare_classes_dropped: dict[str, int] = field(default_factory=dict)
    rows_out: int = 0
    class_counts_raw: dict[str, int] = field(default_factory=dict)
    class_counts_clean: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns that are allowed to be model inputs."""
    return [c for c in df.columns if c not in (LABEL_COL, "source_file", "raw_label")]


def clean_dataframe(df: pd.DataFrame, cfg: Phase1Config) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply label mapping, leakage removal and data cleaning.

    Returns the cleaned frame (numeric features + 'label' + 'source_file')
    and a report with the number of rows/columns each step removed.
    """
    rep = CleaningReport(rows_in=len(df))
    df = df.copy()

    # --- labels -------------------------------------------------------
    df["raw_label"] = df[LABEL_COL].map(clean_label_text)
    rep.class_counts_raw = df["raw_label"].value_counts().to_dict()
    df[LABEL_COL] = df["raw_label"].map(
        lambda x: map_label(x, cfg.label_level, cfg.attempted_as_benign)
    )

    # --- leakage columns ----------------------------------------------
    leak = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
    if not cfg.keep_dst_port and "dst_port" in df.columns:
        leak.append("dst_port")
    rep.leakage_columns_dropped = leak
    df = df.drop(columns=leak)

    # --- keep only numeric features -----------------------------------
    non_num = [c for c in feature_columns(df) if not pd.api.types.is_numeric_dtype(df[c])]
    # "protocol" is numeric in CIC CSVs; anything else non-numeric is dropped.
    rep.non_numeric_columns_dropped = non_num
    df = df.drop(columns=non_num)
    feats = feature_columns(df)

    # --- NaN / inf ----------------------------------------------------
    # CICFlowMeter writes inf for Flow Bytes/s when the duration is 0.
    values = df[feats].to_numpy(dtype=np.float64, copy=False)
    rep.inf_values_found = int(np.isinf(values).sum())
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=feats)
    rep.rows_with_nan_or_inf_dropped = before - len(df)

    # --- duplicates ---------------------------------------------------
    # Exact duplicates (features + label) would appear in both train and
    # test after a random split and inflate the scores.
    if cfg.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=feats + [LABEL_COL])
        rep.duplicate_rows_dropped = before - len(df)

    # --- conflicting labels -------------------------------------------
    # Same feature vector, different labels -> no model can be right on both.
    if cfg.drop_conflicting:
        # Hash each feature row to one integer: much faster than grouping
        # by ~70 float columns on millions of rows.
        row_hash = pd.util.hash_pandas_object(df[feats], index=False)
        n_labels = df[LABEL_COL].groupby(row_hash.to_numpy()).transform("nunique")
        conflict = n_labels > 1
        rep.conflicting_rows_dropped = int(conflict.sum())
        df = df[~conflict]

    # --- rare classes -------------------------------------------------
    counts = df[LABEL_COL].value_counts()
    rare = counts[counts < cfg.min_class_samples]
    rep.rare_classes_dropped = {str(k): int(v) for k, v in rare.items()}
    if len(rare):
        log.warning("Dropping rare classes (< %d rows): %s", cfg.min_class_samples, dict(rare))
        df = df[~df[LABEL_COL].isin(rare.index)]

    df = df.drop(columns=["raw_label"]).reset_index(drop=True)
    rep.rows_out = len(df)
    rep.class_counts_clean = df[LABEL_COL].value_counts().to_dict()
    return df, rep


# ---------------------------------------------------------------------------
# 5. Sampling and splitting
# ---------------------------------------------------------------------------
def stratified_sample(
    df: pd.DataFrame,
    frac: float,
    min_per_class: int,
    max_per_class: int | None,
    seed: int,
) -> pd.DataFrame:
    """Per-class sampling: n_c = clip(frac * count_c, min_per_class, max_per_class).

    A class smaller than `min_per_class` is kept entirely, so rare attacks
    (e.g. Infiltration, Heartbleed) are never sampled away. Note this changes
    the class ratio compared with the raw data; that is documented in the
    results JSON (class counts before and after).
    """
    parts = []
    for _, group in df.groupby(LABEL_COL, sort=True):
        n = int(round(frac * len(group)))
        n = max(n, min(min_per_class, len(group)))
        if max_per_class is not None:
            n = min(n, max_per_class)
        parts.append(group.sample(n=n, random_state=seed) if n < len(group) else group)
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_data(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split (each class in all three)."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[LABEL_COL], random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)  # val as a fraction of train_val
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val[LABEL_COL], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Feature preprocessor (fit on train only)
# ---------------------------------------------------------------------------
class FeaturePreprocessor:
    """Fixes the feature list and drops constant columns, using TRAIN data only.

    Fitting on train only matters: any statistic computed on the test set
    (even "which columns are constant") is information the model should not
    have. Scaling is done inside each model's own pipeline (models/baselines).
    """

    def __init__(self) -> None:
        self.feature_names_: list[str] = []
        self.constant_columns_: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> "FeaturePreprocessor":
        feats = feature_columns(train_df)
        nunique = train_df[feats].nunique(dropna=False)
        self.constant_columns_ = sorted(nunique[nunique <= 1].index.tolist())
        self.feature_names_ = [c for c in feats if c not in self.constant_columns_]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.feature_names_:
            raise RuntimeError("FeaturePreprocessor is not fitted")
        missing = [c for c in self.feature_names_ if c not in df.columns]
        if missing:
            raise KeyError(f"Input is missing features: {missing}")
        return df[self.feature_names_].to_numpy(dtype=np.float32)


def apply_smote(
    X: np.ndarray, y: np.ndarray, target: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample every class smaller than `target` up to `target` rows.

    Only ever call this on the TRAINING split: synthetic points in the test
    set would make evaluation meaningless.
    """
    from imblearn.over_sampling import SMOTE

    classes, counts = np.unique(y, return_counts=True)
    strategy = {int(c): int(target) for c, n in zip(classes, counts) if n < target}
    if not strategy:
        return X, y
    # k_neighbors must be < size of the smallest class being oversampled.
    smallest = int(min(counts[np.isin(classes, list(strategy))]))
    k = max(1, min(5, smallest - 1))
    sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res.astype(np.float32), y_res
