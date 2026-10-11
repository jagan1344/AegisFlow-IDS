"""Phase 2 in ONE command: train on CIC-IDS2017, test on CSE-CIC-IDS2018 and UNSW-NB15.

Run from the repository root (VS Code terminal):

    python scripts/phase2_all.py --cic2018-zip "D:/downloads/<2018>.zip" --unsw-zip "D:/downloads/<unsw>.zip" --laptop

Each zip may also be a folder that contains the CSV files. Once a dataset is
unpacked into data/raw/ you can leave its option out on later runs.

Options:
    --laptop    same source sampling as Phase 1 laptop mode (keep comparable)
    --quick     LightGBM only (about 3x faster)
Results: results/phase2.json and results/phase2_summary.md
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aegisflow.config import Phase2Config  # noqa: E402
from aegisflow.crossdataset import (  # noqa: E402
    load_unsw_nb15,
    per_family_table,
    run_phase2,
    summary_table,
)
from aegisflow.preprocessing import load_csv_folder  # noqa: E402

_spec = importlib.util.spec_from_file_location("download", ROOT / "data" / "download.py")
download = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download)


def ensure(key: str, zip_path: str | None) -> bool:
    """data/raw/<key>/ has CSVs (extracting `zip_path` if given)?"""
    folder = download.RAW_DIR / key
    if zip_path:
        try:
            download.extract(key, Path(zip_path))
        except (RuntimeError, OSError) as exc:
            print(f"[!!] {key}: {exc}")
    if any(folder.rglob("*.csv")):
        print(f"[ok] {key} present")
        return True
    print(f"[--] {key} not found. Get it from {download.DATASETS[key]['page']}")
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cic2018-zip", default=None)
    p.add_argument("--unsw-zip", default=None)
    p.add_argument("--laptop", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    cfg = Phase2Config()
    if args.laptop:
        cfg.source.benign_load_frac = 0.3
        cfg.source.max_per_class = 100_000
    if args.quick:
        cfg.models = ("lightgbm",)

    print("=== Step 1/3: datasets ===")
    if not ensure(cfg.source_key, None):
        sys.exit("The corrected CIC-IDS2017 from Phase 1 is needed (data/raw/cicids2017_improved).")
    have_2018 = ensure(cfg.cic_target_key, args.cic2018_zip)
    have_unsw = ensure(cfg.unsw_key, args.unsw_zip)
    if not (have_2018 or have_unsw):
        sys.exit("Need at least one target dataset (CSE-CIC-IDS2018 or UNSW-NB15).")

    print("\n=== Step 2/3: loading (2018 is sampled while reading, this can take a while) ===")
    s = cfg.source
    source_raw = load_csv_folder(download.RAW_DIR / cfg.source_key, s.max_rows_per_file,
                                 s.benign_load_frac, s.seed, s.attack_load_frac)
    cic_raw = (load_csv_folder(download.RAW_DIR / cfg.cic_target_key, None,
                               cfg.target_benign_load_frac, cfg.seed, cfg.target_attack_load_frac)
               if have_2018 else None)
    unsw_raw = load_unsw_nb15(download.RAW_DIR / cfg.unsw_key) if have_unsw else None

    print("\n=== Step 3/3: experiments ===")
    res = run_phase2(cfg, source_raw, cic_raw, unsw_raw)

    print("\n--- Summary (each row: one experiment x model) ---")
    print(summary_table(res).round(4).to_string(index=False))
    for set_name in ("cicids2018_test", "unsw_nb15"):
        exp = "shared_features"
        if exp in res["experiments"] and set_name in res["experiments"][exp]["models"][cfg.models[-1]]:
            print(f"\n--- {set_name}: share of each family flagged as attack "
                  f"({exp}, {cfg.models[-1]}) ---")
            print(per_family_table(res, exp, cfg.models[-1], set_name).round(4).to_string())
    print(f"\nWrote {cfg.results_dir / 'phase2.json'} and {cfg.results_dir / 'phase2_summary.md'}")


if __name__ == "__main__":
    main()
