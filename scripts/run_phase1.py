"""Run the Phase-1 baseline benchmark on one dataset version.

Examples (from the repository root):
    python scripts/run_phase1.py --dataset cicids2017_original
    python scripts/run_phase1.py --dataset cicids2017_improved
    python scripts/run_phase1.py --dataset cicids2017_original --imbalance smote
    python scripts/run_phase1.py --synthetic        # 10-second smoke test, NOT real results
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aegisflow.benchmark import per_class_f1_table, run_from_folder, run_phase1, summary_table  # noqa: E402
from aegisflow.config import PROJECT_ROOT, Phase1Config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cicids2017_original",
                   help="folder name under data/raw/ (also used as the results name)")
    p.add_argument("--data-dir", default=None, help="override CSV folder")
    p.add_argument("--synthetic", action="store_true", help="smoke test on fake data")
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--min-per-class", type=int, default=None)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--max-rows-per-file", type=int, default=None)
    p.add_argument("--imbalance", choices=["class_weight", "smote", "both", "none"], default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--keep-dst-port", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = Phase1Config()
    for arg, attr in [("sample_frac", "sample_frac"), ("min_per_class", "min_per_class"),
                      ("max_per_class", "max_per_class"), ("max_rows_per_file", "max_rows_per_file"),
                      ("imbalance", "imbalance")]:
        if getattr(args, arg) is not None:
            setattr(cfg, attr, getattr(args, arg))
    if args.models:
        cfg.models = tuple(args.models)
    cfg.keep_dst_port = args.keep_dst_port

    if args.synthetic:
        from aegisflow.synthetic import make_synthetic_cicids

        cfg.results_dir = PROJECT_ROOT / "results" / "synthetic_smoke"
        cfg.artifacts_dir = PROJECT_ROOT / "artifacts" / "synthetic_smoke"
        cfg.min_per_class, cfg.sample_frac = 10, 1.0
        results = run_phase1(make_synthetic_cicids(4000), "synthetic_smoke", cfg)
    else:
        folder = Path(args.data_dir) if args.data_dir else cfg.data_dir / args.dataset
        results = run_from_folder(folder, args.dataset, cfg)

    print("\n=== Summary (test split) ===")
    print(summary_table(results).round(4).to_string(index=False))
    print("\n=== Per-class F1 ===")
    print(per_class_f1_table(results).round(4).to_string())


if __name__ == "__main__":
    main()
