"""Compare Phase-1 results of original vs corrected CIC-IDS2017.

    python scripts/compare_phase1.py \
        --original results/phase1_cicids2017_original.json \
        --improved results/phase1_cicids2017_improved.json

Prints markdown tables and writes results/phase1_comparison.md.
Every number comes from the two JSON files produced by run_phase1.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aegisflow.benchmark import compare_versions, per_class_f1_table, summary_table  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--improved", required=True)
    p.add_argument("--out", default="results/phase1_comparison.md")
    args = p.parse_args()

    orig = json.loads(Path(args.original).read_text())
    impr = json.loads(Path(args.improved).read_text())

    parts = ["# Phase 1 – Original vs corrected CIC-IDS2017\n"]
    for res in (orig, impr):
        s = res["sampling"]
        parts.append(f"## {res['dataset']}\n")
        parts.append(f"Rows after cleaning: {s['rows_after_cleaning']:,} | "
                     f"after sampling: {s['rows_after_sampling']:,} | "
                     f"split: {s['split_sizes']}\n")
        parts.append(summary_table(res).round(4).to_markdown(index=False) + "\n")
        parts.append("Per-class F1:\n\n" + per_class_f1_table(res).round(4).to_markdown() + "\n")
    parts.append("## Difference (improved − original)\n")
    parts.append(compare_versions(orig, impr).round(4).to_markdown() + "\n")

    text = "\n".join(parts)
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
