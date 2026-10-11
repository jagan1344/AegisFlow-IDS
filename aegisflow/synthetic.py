"""Tiny synthetic CIC-IDS2017-shaped data for unit tests and smoke runs.

THIS IS NOT REAL TRAFFIC. It only has the same column names / label
strings / quirks (inf values, duplicates, an identifier column, a constant
column) so the pipeline can be tested without downloading gigabytes.
Never report metrics computed on this data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Raw label strings exactly as they appear in the original CSVs
# (the en-dash in "Web Attack - ..." is mis-encoded in the real files).
_LABELS = {
    "BENIGN": 0.70,
    "DoS Hulk": 0.08,
    "DDoS": 0.06,
    "PortScan": 0.06,
    "FTP-Patator": 0.03,
    "SSH-Patator": 0.02,
    "Bot": 0.02,
    "Web Attack \x96 Brute Force": 0.02,
    "Infiltration": 0.01,
}


# How the same attacks are spelled in CSE-CIC-IDS2018 (no PortScan family).
_LABELS_2018 = {
    "DoS Hulk": "DoS attacks-Hulk",
    "DDoS": "DDOS attack-HOIC",
    "PortScan": "Infilteration",          # 2018 has no PortScan; typo is real
    "FTP-Patator": "FTP-BruteForce",
    "SSH-Patator": "SSH-Bruteforce",
    "Bot": "Bot",
    "Web Attack \x96 Brute Force": "Brute Force -Web",
    "Infiltration": "Infilteration",
}


def make_synthetic_cicids(
    n_rows: int = 3000, seed: int = 0, improved: bool = False, labels_2018: bool = False
) -> pd.DataFrame:
    """Return a raw-looking dataframe with original (or improved) column names.

    labels_2018=True spells the labels the CSE-CIC-IDS2018 way and shifts the
    feature values a little (a fake "different network").
    """
    rng = np.random.default_rng(seed)
    labels = rng.choice(list(_LABELS), size=n_rows, p=list(_LABELS.values()))
    cls_idx = pd.Series(labels).astype("category").cat.codes.to_numpy()

    # Each class gets its own mean, so the task is learnable.
    shift = 1.5 if labels_2018 else 1.0  # different network -> different scale

    def feat(scale: float) -> np.ndarray:
        return shift * np.abs(rng.normal(loc=(cls_idx + 1) * scale, scale=scale, size=n_rows))

    fwd_name = "Total Fwd Packet" if improved else " Total Fwd Packets"
    df = pd.DataFrame(
        {
            " Destination Port": rng.choice([80, 443, 22, 21], size=n_rows),
            " Protocol": rng.choice([6, 17], size=n_rows),
            " Flow Duration": feat(1000.0),
            fwd_name: feat(5.0),
            " Total Backward Packets": feat(4.0),
            "Flow Bytes/s": feat(200.0),
            " Flow Packets/s": feat(20.0),
            " Fwd Packet Length Mean": feat(50.0),
            " Bwd Packet Length Mean": feat(40.0),
            "Total Length of Fwd Packets": feat(500.0),
            " Total Length of Bwd Packets": feat(400.0),
            " Fwd IAT Mean": feat(100.0),
            " Bwd IAT Mean": feat(90.0),
            "Init_Win_bytes_forward": feat(1000.0),
            " Init_Win_bytes_backward": feat(800.0),
            "Bwd PSH Flags": np.zeros(n_rows),           # constant column
            " Fwd Header Length": feat(30.0),
            " Fwd Header Length.1": None,                # duplicate column
            " Label": labels,
        }
    )
    df[" Fwd Header Length.1"] = df[" Fwd Header Length"]
    if improved:
        df.insert(0, "Flow ID", [f"10.0.0.{i % 250}-1.1.1.1" for i in range(n_rows)])
        df.insert(1, "Src IP", [f"10.0.0.{i % 250}" for i in range(n_rows)])
        df.insert(2, "Timestamp", "2017-07-03 08:00:00")
        # A few "attempted" flows, as in the corrected dataset.
        att = rng.choice(n_rows, size=max(1, n_rows // 100), replace=False)
        df.loc[att, " Label"] = "DoS Hulk - Attempted"
    if labels_2018:
        df[" Label"] = df[" Label"].map(lambda x: _LABELS_2018.get(x, x))
    # Quirks of the real data:
    df.loc[rng.choice(n_rows, 5, replace=False), "Flow Bytes/s"] = np.inf
    df.loc[rng.choice(n_rows, 3, replace=False), " Flow Packets/s"] = np.nan
    df = pd.concat([df, df.iloc[:20]], ignore_index=True)  # exact duplicates
    return df


_UNSW_CATS = {"Normal": 0.75, "Exploits": 0.08, "DoS": 0.05, "Reconnaissance": 0.05,
              "Fuzzers": 0.04, "Generic": 0.03}


def make_synthetic_unsw(n_rows: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Fake UNSW-NB15 partition file (same column names, lower case)."""
    rng = np.random.default_rng(seed)
    cats = rng.choice(list(_UNSW_CATS), size=n_rows, p=list(_UNSW_CATS.values()))
    k = pd.Series(cats).astype("category").cat.codes.to_numpy() + 1

    def feat(scale: float) -> np.ndarray:
        return np.abs(rng.normal(loc=k * scale, scale=scale, size=n_rows))

    dur = feat(0.5)
    dur[rng.choice(n_rows, 20, replace=False)] = 0.0  # zero-length flows exist
    return pd.DataFrame({
        "id": np.arange(n_rows), "dur": dur,
        "proto": rng.choice(["tcp", "udp", "ospf"], size=n_rows),
        "spkts": feat(5.0).round(), "dpkts": feat(4.0).round(),
        "sbytes": feat(500.0).round(), "dbytes": feat(400.0).round(),
        "smean": feat(50.0), "dmean": feat(40.0),
        "sinpkt": feat(0.1), "dinpkt": feat(0.09),
        "swin": feat(255.0).round(), "dwin": feat(255.0).round(),
        "attack_cat": cats, "label": (cats != "Normal").astype(int),
    })
