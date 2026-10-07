"""Phase 1 in ONE command: get data -> train 5 models on both versions -> compare.

Run from the repository root (VS Code terminal):

    python scripts/phase1_all.py --improved-zip "C:/Users/you/Downloads/<corrected>.zip"

Options:
    --original-zip PATH   use a MachineLearningCSV.zip you downloaded yourself
                          (otherwise the script downloads it)
    --improved-zip PATH   the corrected CIC-IDS2017 zip (manual download)
    --laptop              lower RAM use (keeps 30% of BENIGN rows while reading)
    --quick               only LogReg + LightGBM (a fast first run, ~5 min)

Results: results/phase1_<dataset>.json and results/phase1_comparison.md
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aegisflow.benchmark import (  # noqa: E402
    compare_versions,
    per_class_f1_table,
    run_from_folder,
    summary_table,
)
from aegisflow.config import Phase1Config  # noqa: E402

# data/download.py is a script, not a package module, so load it by path.
_spec = importlib.util.spec_from_file_location("download", ROOT / "data" / "download.py")
download = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download)


def has_csvs(key: str) -> bool:
    return any((download.RAW_DIR / key).rglob("*.csv"))


def ensure_original(zip_path: str | None) -> bool:
    """Make sure data/raw/cicids2017_original/ has CSVs. Returns success."""
    key = "cicids2017_original"
    if has_csvs(key):
        print(f"[ok] {key} already present")
        return True
    if zip_path:
        download.extract(key, Path(zip_path))
        return has_csvs(key)
    url = download.DATASETS[key]["url"]
    print(f"[..] downloading {key} (~230 MB) from\n     {url}")
    try:
        download.cmd_fetch(argparse.Namespace(dataset=key, url=None))
    except Exception as exc:  # network errors, 403, moved file...
        print(f"[!!] automatic download failed: {exc}\n"
              f"     Download MachineLearningCSV.zip by hand from\n"
              f"     {download.DATASETS[key]['page']}\n"
              f"     then re-run with --original-zip <path to zip>")
        return False
    return has_csvs(key)


def ensure_improved(zip_path: str | None) -> bool:
    key = "cicids2017_improved"
    if has_csvs(key):
        print(f"[ok] {key} already present")
        return True
    if zip_path:
        download.extract(key, Path(zip_path))
        if has_csvs(key):
            return True
        print("[!!] no CSV files found inside that zip - send the zip's file list to your assistant")
        return False
    print(f"[--] {key} not found. Download it by hand from\n"
          f"     {download.DATASETS[key]['page']}\n"
          f"     and re-run with --improved-zip <path to zip>. Running original only for now.")
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original-zip", default=None)
    p.add_argument("--improved-zip", default=None)
    p.add_argument("--laptop", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    cfg = Phase1Config()
    if args.laptop:
        cfg.benign_load_frac = 0.3
        cfg.max_per_class = 100_000
    if args.quick:
        cfg.models = ("logreg", "lightgbm")
    print(f"Settings: benign_load_frac={cfg.benign_load_frac}, sample_frac={cfg.sample_frac}, "
          f"max_per_class={cfg.max_per_class}, models={cfg.models}")

    print("\n=== Step 1/3: datasets ===")
    have_orig = ensure_original(args.original_zip)
    have_impr = ensure_improved(args.improved_zip)
    if not have_orig and not have_impr:
        sys.exit("No dataset available - see the messages above.")

    print("\n=== Step 2/3: training and evaluation ===")
    results = {}
    for key, ok in (("cicids2017_original", have_orig), ("cicids2017_improved", have_impr)):
        if not ok:
            continue
        res = run_from_folder(cfg.data_dir / key, key, cfg)
        results[key] = res
        print(f"\n--- {key}: summary (test split) ---")
        print(summary_table(res).round(4).to_string(index=False))
        print(f"\n--- {key}: per-class F1 ---")
        print(per_class_f1_table(res).round(4).to_string())

    print("\n=== Step 3/3: comparison ===")
    if len(results) == 2:
        table = compare_versions(results["cicids2017_original"], results["cicids2017_improved"])
        print(table.round(4).to_string())
        out = cfg.results_dir / "phase1_comparison.md"
        out.write_text("# Phase 1 - original vs corrected CIC-IDS2017\n\n"
                       + table.round(4).to_markdown() + "\n")
        print(f"\nWrote {out}")
    else:
        print("Only one dataset was run; add the other and re-run to get the comparison.")
    print(f"All results are in {cfg.results_dir}")


if __name__ == "__main__":
    main()
