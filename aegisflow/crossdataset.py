"""Phase 2: cross-dataset generalisation.

Question: a model trained on CIC-IDS2017, does it still work on traffic from
a different network and a different capture (CSE-CIC-IDS2018, UNSW-NB15)?

Feature spaces
--------------
* CIC-IDS2017 (corrected) and CSE-CIC-IDS2018 (corrected) were produced by
  the same fixed CICFlowMeter, so they share all ~80 features.
* UNSW-NB15 was produced by Argus/Bro, with different features. We map the
  14 features that both tools measure into one SHARED space, converting
  units (UNSW duration is in seconds, CIC in microseconds, etc.). The mapping
  is an approximation (e.g. byte counts may include headers in one tool and
  not the other); this is a stated limitation, and per-dataset
  normalisation is one of the fixes we test for exactly this reason.

Label spaces
------------
Attack families do not line up across datasets (UNSW has "Exploits",
"Fuzzers"...). Therefore we report:
* binary metrics (attack vs BENIGN), which are always comparable, using
  P(attack) = 1 - P(BENIGN) from the multi-class model;
* macro-F1 only over target rows whose family the model was trained on;
* per-family detection rate (share flagged as any attack).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from aegisflow.config import BENIGN_LABEL, Phase2Config, set_global_seed
from aegisflow.models import balanced_sample_weight, build_model, fit_model
from aegisflow.models.baselines import signed_log1p
from aegisflow.preprocessing import (
    LABEL_COL,
    FeaturePreprocessor,
    clean_dataframe,
    normalise_columns,
    stratified_sample,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Shared feature space (CICFlowMeter <-> UNSW-NB15 / Argus)
# ---------------------------------------------------------------------------
# Canonical CIC names (see preprocessing.canonical_name). Units: time in
# microseconds, sizes in bytes, rates per second.
SHARED_FEATURES: list[str] = [
    "protocol",                  # IANA number: 6 TCP, 17 UDP, 1 ICMP
    "flow_duration",             # us        <- UNSW dur (s) * 1e6
    "total_fwd_packets",         #           <- spkts
    "total_bwd_packets",         #           <- dpkts
    "total_length_fwd_packets",  # bytes     <- sbytes
    "total_length_bwd_packets",  # bytes     <- dbytes
    "fwd_packet_length_mean",    # bytes     <- smean
    "bwd_packet_length_mean",    # bytes     <- dmean
    "flow_bytes_s",              # B/s       <- (sbytes+dbytes)/dur
    "flow_packets_s",            # pkt/s     <- (spkts+dpkts)/dur
    "fwd_iat_mean",              # us        <- sinpkt (ms) * 1000
    "bwd_iat_mean",              # us        <- dinpkt (ms) * 1000
    "fwd_init_win_bytes",        #           <- swin (TCP window)
    "bwd_init_win_bytes",        #           <- dwin
]

_UNSW_PROTO = {"tcp": 6, "udp": 17, "icmp": 1}
# UNSW categories that correspond to a CIC family; all others are attack
# types the CIC-trained model has never seen (only binary metrics apply).
_UNSW_FAMILY = {"normal": BENIGN_LABEL, "dos": "DoS", "reconnaissance": "PortScan"}


def unsw_label(cat: object) -> str:
    """'Normal' -> BENIGN, 'DoS' -> DoS, 'Exploits' -> 'UNSW-Exploits' ..."""
    c = str(cat).strip()
    low = c.lower()
    if low in ("", "nan", "none"):
        return BENIGN_LABEL
    if low in _UNSW_FAMILY:
        return _UNSW_FAMILY[low]
    return "UNSW-" + c.replace(" ", "")


def load_unsw_nb15(folder: str | Path) -> pd.DataFrame:
    """Load the UNSW-NB15 partition CSVs (training-set + testing-set).

    Only files with a header containing 'attack_cat' are used. The four big
    'UNSW-NB15_1..4.csv' files have no header row and are skipped.
    """
    folder = Path(folder)
    frames = []
    for f in sorted(folder.rglob("*.csv")):
        head = pd.read_csv(f, nrows=0)
        cols = [c.strip().lower() for c in head.columns]
        if "attack_cat" not in cols:
            log.warning("Skipping %s (no 'attack_cat' header; use the training/testing-set CSVs)", f.name)
            continue
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        df["source_file"] = f.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No UNSW-NB15 partition CSVs under {folder}. Expected "
            "UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv."
        )
    return pd.concat(frames, ignore_index=True)


def unsw_to_shared(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw UNSW-NB15 columns into SHARED_FEATURES (+ label)."""
    need = ["proto", "dur", "spkts", "dpkts", "sbytes", "dbytes", "smean", "dmean",
            "sinpkt", "dinpkt", "swin", "dwin", "attack_cat"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise KeyError(f"UNSW-NB15 file is missing columns {missing}")
    num = {c: pd.to_numeric(raw[c], errors="coerce").astype(np.float64) for c in need[1:-1]}
    dur = num["dur"]
    # Zero-length flows (single packet) have no defined rate: use 0, not inf.
    safe_dur = dur.where(dur > 0)
    out = pd.DataFrame(
        {
            "protocol": raw["proto"].astype(str).str.strip().str.lower().map(_UNSW_PROTO).fillna(0),
            "flow_duration": dur * 1e6,
            "total_fwd_packets": num["spkts"],
            "total_bwd_packets": num["dpkts"],
            "total_length_fwd_packets": num["sbytes"],
            "total_length_bwd_packets": num["dbytes"],
            "fwd_packet_length_mean": num["smean"],
            "bwd_packet_length_mean": num["dmean"],
            "flow_bytes_s": ((num["sbytes"] + num["dbytes"]) / safe_dur).fillna(0.0),
            "flow_packets_s": ((num["spkts"] + num["dpkts"]) / safe_dur).fillna(0.0),
            "fwd_iat_mean": num["sinpkt"] * 1000.0,
            "bwd_iat_mean": num["dinpkt"] * 1000.0,
            "fwd_init_win_bytes": num["swin"],
            "bwd_init_win_bytes": num["dwin"],
        }
    ).astype(np.float32)
    out[LABEL_COL] = raw["attack_cat"].map(unsw_label).to_numpy()
    if "source_file" in raw.columns:
        out["source_file"] = raw["source_file"].to_numpy()
    return out


# ---------------------------------------------------------------------------
# 2. Per-dataset normalisation (a "fix" we test)
# ---------------------------------------------------------------------------
class DatasetNormaliser:
    """signed-log, then z-score with statistics of ONE dataset.

    Each dataset is normalised with its own mean/std, so a feature that is
    systematically larger in one network (different link speed, different
    tool units) is brought to a comparable scale. For a target dataset the
    statistics come from its UNLABELLED features only (no labels are used);
    this is a transductive, unsupervised domain-adaptation step and is
    reported as such.
    """

    def fit(self, X: np.ndarray) -> "DatasetNormaliser":
        Z = signed_log1p(X.astype(np.float64))
        self.mean_ = Z.mean(axis=0)
        std = Z.std(axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((signed_log1p(X.astype(np.float64)) - self.mean_) / self.std_).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Metrics that are comparable across datasets
# ---------------------------------------------------------------------------
def evaluate_transfer(
    model, X: np.ndarray, labels: np.ndarray, class_names: list[str]
) -> dict:
    """Binary + known-family metrics for one evaluation set.

    labels: string family labels of the evaluation rows (may include
    families the model never saw).
    """
    labels = np.asarray(labels, dtype=object)
    proba = model.predict_proba(X)
    names = np.asarray(class_names, dtype=object)
    pred = names[proba.argmax(axis=1)]
    benign_idx = class_names.index(BENIGN_LABEL)
    attack_score = 1.0 - proba[:, benign_idx]

    y_att = labels != BENIGN_LABEL
    p_att = pred != BENIGN_LABEL
    tp = int((y_att & p_att).sum())
    fp = int((~y_att & p_att).sum())
    fn = int((y_att & ~p_att).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    out = {
        "n_rows": int(len(labels)),
        "n_attack": int(y_att.sum()),
        "binary_detection_rate": recall,
        "binary_fpr": float(p_att[~y_att].mean()) if (~y_att).any() else float("nan"),
        "binary_precision": precision,
        "binary_f1": (2 * precision * recall / (precision + recall))
        if precision + recall > 0 and not np.isnan(recall) else 0.0,
        "binary_pr_auc": float(average_precision_score(y_att, attack_score))
        if 0 < y_att.sum() < len(y_att) else float("nan"),
    }
    # Multi-class quality, only where the true family is one the model knows.
    known = np.isin(labels, names)
    out["known_family_rows"] = int(known.sum())
    if known.any():
        present = sorted(set(labels[known]))
        out["macro_f1_known"] = float(
            f1_score(labels[known], pred[known], labels=present, average="macro", zero_division=0)
        )
    else:
        out["macro_f1_known"] = float("nan")
    # Per-family view: how much of each family is flagged as an attack.
    fam = {}
    for f in sorted(set(labels)):
        m = labels == f
        entry = {"n": int(m.sum()), "flagged_as_attack": float(p_att[m].mean())}
        if f in class_names:
            entry["exact_family_recall"] = float((pred[m] == f).mean())
        fam[str(f)] = entry
    out["per_family"] = fam
    return out


TCP_WINDOW_FEATURES = ("fwd_init_win_bytes", "bwd_init_win_bytes")


def feature_importance(model, feats: list[str], top: int = 10) -> dict[str, float]:
    """Normalised importances (sum = 1) of the top features, if the model has them."""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    imp = np.asarray(imp, dtype=float)
    if imp.sum() <= 0:
        return {}
    imp = imp / imp.sum()
    order = np.argsort(imp)[::-1][:top]
    return {feats[i]: float(imp[i]) for i in order}


def feature_shift(
    src: pd.DataFrame, tgt: pd.DataFrame, feats: list[str], seed: int = 42, max_rows: int = 5000
) -> dict:
    """Per family present in both datasets: median of each feature in each
    dataset and the two-sample Kolmogorov-Smirnov statistic (0 = identical
    distributions, 1 = no overlap at all).

    For BENIGN this doubles as a units check: if the medians differed by a
    factor like 1000 for every time feature, the datasets would use
    different units (a bug), not just different traffic.
    """
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(seed)
    out = {}
    families = sorted(set(src[LABEL_COL]) & set(tgt[LABEL_COL]))
    for fam in families:
        a = src[src[LABEL_COL] == fam]
        b = tgt[tgt[LABEL_COL] == fam]
        if len(a) > max_rows:
            a = a.iloc[rng.choice(len(a), max_rows, replace=False)]
        if len(b) > max_rows:
            b = b.iloc[rng.choice(len(b), max_rows, replace=False)]
        rows = {}
        for f in feats:
            x, y = a[f].to_numpy(np.float64), b[f].to_numpy(np.float64)
            rows[f] = {
                "median_2017": float(np.median(x)),
                "median_2018": float(np.median(y)),
                "ks": float(ks_2samp(x, y).statistic),
            }
        out[fam] = {"n_2017": int(len(a)), "n_2018": int(len(b)), "features": rows}
    return out


def shift_table(results: dict, top: int = 5) -> pd.DataFrame:
    """The `top` most-shifted features (highest KS) for every family."""
    rows = []
    for fam, d in results.get("shift_2017_vs_2018", {}).items():
        feats = sorted(d["features"].items(), key=lambda kv: kv[1]["ks"], reverse=True)[:top]
        for f, v in feats:
            rows.append({"family": fam, "feature": f, "ks": v["ks"],
                         "median_2017": v["median_2017"], "median_2018": v["median_2018"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. The Phase-2 experiment
# ---------------------------------------------------------------------------
def _matrix(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return df[feats].to_numpy(dtype=np.float32)


def prepare_cic_target(raw: pd.DataFrame, cfg: Phase2Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean CSE-CIC-IDS2018 like the source, cap rows, split adapt/test."""
    clean, report = clean_dataframe(normalise_columns(raw), cfg.source)
    sample = stratified_sample(clean, 1.0, 0, cfg.target_max_per_class, cfg.seed)
    adapt, test = train_test_split(
        sample, train_size=cfg.adapt_frac, stratify=sample[LABEL_COL], random_state=cfg.seed
    )
    info = {
        # Which day files were used (the report must state the subset).
        "files_used": sorted(clean["source_file"].unique().tolist())
        if "source_file" in clean.columns else [],
        "cleaning": report.to_dict(),
        "class_counts_test": test[LABEL_COL].value_counts().to_dict(),
        "class_counts_adapt": adapt[LABEL_COL].value_counts().to_dict(),
    }
    return adapt.reset_index(drop=True), test.reset_index(drop=True), info


def prepare_unsw(raw: pd.DataFrame, cfg: Phase2Config) -> tuple[pd.DataFrame, dict]:
    """Map UNSW-NB15 to the shared space, clean it, cap rows per class."""
    shared = unsw_to_shared(raw)
    # labels are final already: "fine" level keeps them unchanged
    fine_cfg = replace(cfg.source, label_level="fine")
    clean, report = clean_dataframe(shared, fine_cfg)
    sample = stratified_sample(clean, 1.0, 0, cfg.target_max_per_class, cfg.seed)
    return sample.reset_index(drop=True), {
        "cleaning": report.to_dict(),
        "class_counts": sample[LABEL_COL].value_counts().to_dict(),
    }


def run_phase2(
    cfg: Phase2Config,
    source_raw: pd.DataFrame,
    cic_target_raw: pd.DataFrame | None = None,
    unsw_raw: pd.DataFrame | None = None,
    write: bool = True,
) -> dict:
    """Train on the source dataset, test within and across datasets.

    Experiments (each run for every model in cfg.models):
      all_features          all CICFlowMeter features; tested on 2017 + 2018
      all_features_norm     same + per-dataset normalisation
      shared_features       the 14 SHARED_FEATURES; tested on 2017, 2018, UNSW
      shared_features_norm  same + per-dataset normalisation
      multi_dataset         shared features, trained on 2017 + part of 2018
      multi_dataset_norm    same + per-dataset normalisation
    The 2018 test part is identical for every experiment, so all rows of
    the result table are directly comparable. UNSW is never trained on.
    """
    from aegisflow.benchmark import _jsonable, prepare_data  # avoid import cycle

    set_global_seed(cfg.seed)
    t0 = time.time()
    src = prepare_data(source_raw, cfg.source)
    s_train, s_test = src["train"], src["test"]

    t_adapt = t_test = u = None
    info: dict = {"source": {"cleaning": src["cleaning"], "sampling": src["sampling"]}}
    if cic_target_raw is not None:
        t_adapt, t_test, info["cic2018"] = prepare_cic_target(cic_target_raw, cfg)
    if unsw_raw is not None:
        u, info["unsw"] = prepare_unsw(unsw_raw, cfg)

    # Feature lists.
    all_feats = FeaturePreprocessor().fit(s_train).feature_names_
    if t_test is not None:
        missing = [f for f in all_feats if f not in t_test.columns]
        if missing:
            log.warning("Features missing in CSE-CIC-IDS2018, not used: %s", missing)
        all_feats = [f for f in all_feats if f in t_test.columns]
    shared = [f for f in SHARED_FEATURES if f in s_train.columns]
    if len(shared) < len(SHARED_FEATURES):
        log.warning("Shared features missing in source: %s", sorted(set(SHARED_FEATURES) - set(shared)))
    info["features"] = {"all": all_feats, "shared": shared}

    results: dict = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "data": info,
        "experiments": {},
    }
    if t_test is not None:
        # Why does transfer fail? Compare each feature between datasets,
        # family by family (and BENIGN, which also checks units match).
        results["shift_2017_vs_2018"] = feature_shift(
            s_train, pd.concat([t_adapt, t_test]), shared, seed=cfg.seed)

    for exp in cfg.experiments:
        multi = exp.startswith("multi")
        if multi and t_adapt is None:
            log.info("Skipping %s (no CSE-CIC-IDS2018 data)", exp)
            continue
        if exp.startswith("all"):
            feats = all_feats
        elif exp.startswith("shared_no_tcpwin"):
            # Ablation: initial TCP window sizes mostly reflect the operating
            # system of the machines in one lab (a "fingerprint"), not attack
            # behaviour. Does the model rely on that shortcut?
            feats = [f for f in shared if f not in TCP_WINDOW_FEATURES]
        else:
            feats = shared
        norm = exp.endswith("_norm")

        # Feature matrices for every dataset, each normalised by itself.
        def prep(dfs: list[pd.DataFrame]) -> list[np.ndarray]:
            mats = [_matrix(d, feats) for d in dfs]
            if norm:
                n = DatasetNormaliser().fit(np.vstack(mats) if len(mats) > 1 else mats[0])
                mats = [n.transform(m) for m in mats]
            return mats

        # Source: statistics from TRAIN only, applied to train and test.
        Xs_tr = _matrix(s_train, feats)
        Xs_te = _matrix(s_test, feats)
        if norm:
            n_src = DatasetNormaliser().fit(Xs_tr)
            Xs_tr, Xs_te = n_src.transform(Xs_tr), n_src.transform(Xs_te)
        eval_sets = {"cicids2017_test": (Xs_te, s_test[LABEL_COL].to_numpy())}

        X_train, y_train_lbl = Xs_tr, s_train[LABEL_COL].to_numpy()
        if t_test is not None:
            # 2018 statistics: all of its (unlabelled) rows, adapt + test.
            Xt_ad, Xt_te = prep([t_adapt, t_test])
            eval_sets["cicids2018_test"] = (Xt_te, t_test[LABEL_COL].to_numpy())
            if multi:
                X_train = np.vstack([X_train, Xt_ad])
                y_train_lbl = np.concatenate([y_train_lbl, t_adapt[LABEL_COL].to_numpy()])
        if u is not None and set(feats) <= set(shared):
            (Xu,) = prep([u])
            eval_sets["unsw_nb15"] = (Xu, u[LABEL_COL].to_numpy())

        enc = LabelEncoder().fit(y_train_lbl)
        y_train = enc.transform(y_train_lbl)
        class_names = enc.classes_.tolist()
        weights = (balanced_sample_weight(y_train)
                   if cfg.source.imbalance in {"class_weight", "both"} else None)

        exp_res = {"n_features": len(feats), "normalised": norm,
                   "train_rows": int(len(y_train)), "classes": class_names, "models": {}}
        for name in cfg.models:
            log.info("[phase2] %s / %s: training on %d rows, %d features",
                     exp, name, len(y_train), len(feats))
            model = build_model(name, seed=cfg.seed, n_jobs=cfg.source.n_jobs)
            ts = time.time()
            fit_model(model, X_train, y_train, weights, seed=cfg.seed)
            m_res = {"train_seconds": time.time() - ts}
            for set_name, (X, lbl) in eval_sets.items():
                m_res[set_name] = evaluate_transfer(model, X, lbl, class_names)
            m_res["feature_importance"] = feature_importance(model, feats)
            exp_res["models"][name] = m_res
            log.info("[phase2] %s / %s: %s", exp, name, {
                k: round(v.get("binary_f1", float("nan")), 4)
                for k, v in m_res.items() if isinstance(v, dict)})
        results["experiments"][exp] = exp_res

    results["total_seconds"] = time.time() - t0
    if write:
        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        (cfg.results_dir / "phase2.json").write_text(json.dumps(results, indent=2, default=_jsonable))
        (cfg.results_dir / "phase2_summary.md").write_text(summary_markdown(results))
    return results


# ---------------------------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------------------------
def summary_table(results: dict) -> pd.DataFrame:
    """One row per (experiment, model). NaN = not applicable / no data."""
    rows = []
    for exp, e in results["experiments"].items():
        for name, m in e["models"].items():
            def g(set_name: str, key: str) -> float:
                return m.get(set_name, {}).get(key, float("nan"))
            rows.append({
                "experiment": exp,
                "model": name,
                "n_feat": e["n_features"],
                "2017_macro_f1": g("cicids2017_test", "macro_f1_known"),
                "2017_bin_f1": g("cicids2017_test", "binary_f1"),
                "2018_macro_f1": g("cicids2018_test", "macro_f1_known"),
                "2018_bin_f1": g("cicids2018_test", "binary_f1"),
                "2018_detect": g("cicids2018_test", "binary_detection_rate"),
                "2018_fpr": g("cicids2018_test", "binary_fpr"),
                "unsw_bin_f1": g("unsw_nb15", "binary_f1"),
                "unsw_detect": g("unsw_nb15", "binary_detection_rate"),
                "unsw_fpr": g("unsw_nb15", "binary_fpr"),
            })
    return pd.DataFrame(rows)


def per_family_table(results: dict, experiment: str, model: str, set_name: str) -> pd.DataFrame:
    """Share of each target family flagged as an attack (for one model)."""
    fam = results["experiments"][experiment]["models"][model][set_name]["per_family"]
    return pd.DataFrame(fam).T.sort_values("n", ascending=False)


def summary_markdown(results: dict) -> str:
    t = summary_table(results)
    parts = [
        "# Phase 2 - cross-dataset generalisation\n",
        "Trained on CIC-IDS2017 (corrected). `2017_*` = within-dataset test split; "
        "`2018_*` = CSE-CIC-IDS2018 (corrected) held-out part; `unsw_*` = UNSW-NB15 "
        "(never trained on; shared features only). `macro_f1` counts only families the "
        "model was trained on; `bin_*` = attack vs benign.\n",
        t.round(4).to_markdown(index=False),
        "",
    ]
    st = shift_table(results)
    if len(st):
        parts += [
            "## Distribution shift 2017 -> 2018 (most-shifted shared features per family)\n",
            "KS = Kolmogorov-Smirnov statistic: 0 identical, 1 no overlap.\n",
            st.round(4).to_markdown(index=False),
            "",
        ]
    return "\n".join(parts)
