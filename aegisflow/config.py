"""Central configuration for AegisFlow IDS.

Every tunable number in the project lives here, so experiments are
reproducible and a reviewer can see all choices in one place.
Values can be overridden from the command line (see scripts/) or by
editing this file; nothing else in the package hard-codes settings.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Root of the project (the folder that contains this package).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# One seed for everything: numpy, python's random, and every model.
SEED: int = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fix all random number generators so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Leakage columns
# ---------------------------------------------------------------------------
# These columns identify a *specific* flow/host/time in the lab capture.
# A model that sees them can memorise "attacker IP 205.174.165.73 = attack"
# instead of learning traffic behaviour, so they are NEVER model inputs.
# Names are given in canonical form (see preprocessing.canonical_name).
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "id",                 # row id in the improved dataset
    "flow_id",            # "srcIP-dstIP-srcPort-dstPort-proto" string
    "src_ip",
    "dst_ip",
    "src_port",           # ephemeral client port: pure noise / identifier
    "timestamp",
    "attempted_category",  # label metadata in the improved dataset
)

# Destination port is debatable: it carries real signal (port 80 vs 22) but
# in a single-lab dataset it also lets the model memorise the lab's services.
# Default: drop it (the honest, harder setting). Flip to True for an ablation.
KEEP_DST_PORT: bool = False


@dataclass
class Phase1Config:
    """All settings for the Phase-1 data pipeline and baselines."""

    # ---- data locations -------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    results_dir: Path = PROJECT_ROOT / "results"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    # ---- labels ---------------------------------------------------------
    # "family" groups e.g. DoS Hulk / DoS GoldenEye -> DoS (recommended,
    # needed for leave-one-family-out in Phase 3). "fine" keeps raw labels.
    label_level: str = "family"
    # In the improved dataset, "X - Attempted" flows are attack attempts that
    # carried no malicious payload (e.g. a failed connection). The authors
    # explain they behave like benign traffic. True -> relabel as BENIGN.
    attempted_as_benign: bool = True
    # Classes with fewer rows than this after cleaning are dropped (a class
    # with 3 rows cannot be split into train/val/test meaningfully).
    min_class_samples: int = 20

    # ---- cleaning -------------------------------------------------------
    drop_duplicates: bool = True
    # Rows with identical features but different labels cannot be learned;
    # we report how many there are and drop them if True.
    drop_conflicting: bool = True
    keep_dst_port: bool = KEEP_DST_PORT
    # For laptops: read at most this many rows per CSV (None = all rows).
    max_rows_per_file: int | None = None

    # ---- stratified sampling (to fit Colab RAM) -------------------------
    # Each class keeps `sample_frac` of its rows, but never fewer than
    # `min_per_class` (rare attacks are kept whole) and never more than
    # `max_per_class`. Set sample_frac=1.0 and max_per_class=None for all data.
    sample_frac: float = 0.3
    min_per_class: int = 5_000
    max_per_class: int | None = 200_000

    # ---- split ----------------------------------------------------------
    # Stratified random split. train / validation / test.
    # Validation is reserved for threshold tuning and calibration (Phase 4).
    val_size: float = 0.15
    test_size: float = 0.15

    # ---- imbalance handling ---------------------------------------------
    # "class_weight": weight each sample by 1/frequency of its class.
    # "smote": oversample minority classes in TRAIN ONLY with SMOTE.
    # "both": SMOTE up to `smote_target`, then class weights.
    # "none": nothing (to show why imbalance handling matters).
    imbalance: str = "class_weight"
    smote_target: int = 20_000  # minority classes are oversampled up to this

    # ---- models ---------------------------------------------------------
    models: tuple[str, ...] = ("logreg", "random_forest", "xgboost", "lightgbm", "mlp")
    n_jobs: int = -1
    seed: int = SEED

    # ---- evaluation -----------------------------------------------------
    latency_batch_rows: int = 10_000  # rows used to time batch inference
    latency_single_repeats: int = 200  # single-flow predictions to time

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        self.artifacts_dir = Path(self.artifacts_dir)
        if self.label_level not in {"family", "fine"}:
            raise ValueError("label_level must be 'family' or 'fine'")
        if self.imbalance not in {"class_weight", "smote", "both", "none"}:
            raise ValueError("imbalance must be class_weight|smote|both|none")
        if not 0 < self.val_size + self.test_size < 1:
            raise ValueError("val_size + test_size must be in (0, 1)")


# The class name used for normal traffic everywhere in the project.
BENIGN_LABEL: str = "BENIGN"

# LLM mode is used from Phase 5 on. Read once here so it is in one place.
LLM_MODE: str = os.getenv("AEGIS_LLM_MODE", "offline")  # offline|ollama|api
