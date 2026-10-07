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


def make_synthetic_cicids(n_rows: int = 3000, seed: int = 0, improved: bool = False) -> pd.DataFrame:
    """Return a raw-looking dataframe with original (or improved) column names."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(list(_LABELS), size=n_rows, p=list(_LABELS.values()))
    cls_idx = pd.Series(labels).astype("category").cat.codes.to_numpy()

    # Each class gets its own mean, so the task is learnable.
    def feat(scale: float) -> np.ndarray:
        return np.abs(rng.normal(loc=(cls_idx + 1) * scale, scale=scale, size=n_rows))

    fwd_name = "Total Fwd Packet" if improved else " Total Fwd Packets"
    df = pd.DataFrame(
        {
            " Destination Port": rng.choice([80, 443, 22, 21], size=n_rows),
            " Flow Duration": feat(1000.0),
            fwd_name: feat(5.0),
            " Total Backward Packets": feat(4.0),
            "Flow Bytes/s": feat(200.0),
            " Flow Packets/s": feat(20.0),
            " Fwd Packet Length Mean": feat(50.0),
            " Bwd Packet Length Mean": feat(40.0),
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
    # Quirks of the real data:
    df.loc[rng.choice(n_rows, 5, replace=False), "Flow Bytes/s"] = np.inf
    df.loc[rng.choice(n_rows, 3, replace=False), " Flow Packets/s"] = np.nan
    df = pd.concat([df, df.iloc[:20]], ignore_index=True)  # exact duplicates
    return df
