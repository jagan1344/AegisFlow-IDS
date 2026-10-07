"""Download / verify helper for the free datasets used by AegisFlow.

No raw data is stored in git. Every dataset lands in data/raw/<key>/.

Usage (from the repository root):
    python data/download.py list
    python data/download.py fetch cicids2017_original            # default URL
    python data/download.py fetch cicids2017_improved --url <zip>  # URL from the site
    python data/download.py verify cicids2017_original

Some providers put the files behind a short registration/terms page. For
those the script prints the page to open; download the zip yourself, then:
    python data/download.py extract cicids2017_improved /path/to/file.zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"

DATASETS: dict[str, dict] = {
    "cicids2017_original": {
        "name": "CIC-IDS2017 (original, CICFlowMeter-V3 CSVs)",
        "page": "https://www.unb.ca/cic/datasets/ids-2017.html",
        # The UNB page links to this file server; the path has been stable for
        # years but confirm it on the page above if the download fails.
        "url": "http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip",
        "license": "Free for research; cite Sharafaldin et al., ICISSP 2018.",
        "phase": "1",
    },
    "cicids2017_improved": {
        "name": "CIC-IDS2017 improved/re-labelled (Engelen, Rimmer, Joosen; KU Leuven DistriNet)",
        "page": "https://intrusion-detection.distrinet-research.be/CNS2022/",
        "url": None,  # copy the CSV zip link from the page
        "license": "Free for research; cite Engelen et al., WTMC 2021 and Lanvin et al., CRiSIS 2022.",
        "phase": "1",
    },
    "cicids2018_improved": {
        "name": "CSE-CIC-IDS2018 improved (same DistriNet project, same feature extractor)",
        "page": "https://intrusion-detection.distrinet-research.be/CNS2022/",
        "url": None,
        "license": "Free for research; cite Lanvin et al. / Engelen et al.",
        "phase": "2",
    },
    "nf_unsw_nb15_v2": {
        "name": "NF-UNSW-NB15-v2 (NetFlow, 43 features, Univ. of Queensland)",
        "page": "https://staff.itee.uq.edu.au/marius/NIDS_datasets/",
        "url": None,
        "license": "Free for academic research with citation (Sarhan, Layeghy, Portmann).",
        "phase": "2",
    },
    "nf_cse_cic_ids2018_v2": {
        "name": "NF-CSE-CIC-IDS2018-v2 (NetFlow, 43 features, Univ. of Queensland)",
        "page": "https://staff.itee.uq.edu.au/marius/NIDS_datasets/",
        "url": None,
        "license": "Free for academic research with citation (Sarhan, Layeghy, Portmann).",
        "phase": "2",
    },
}


def _target(key: str) -> Path:
    if key not in DATASETS:
        sys.exit(f"Unknown dataset {key!r}. Run: python data/download.py list")
    return RAW_DIR / key


def cmd_list(_: argparse.Namespace) -> None:
    for key, d in DATASETS.items():
        print(f"{key}  [phase {d['phase']}]\n  {d['name']}\n  page:    {d['page']}")
        print(f"  url:     {d['url'] or '(manual: copy link from page)'}\n  license: {d['license']}\n")


def extract(key: str, zip_path: Path) -> None:
    """Unzip into data/raw/<key>/ (CSV files only)."""
    out = _target(key)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        for m in members:
            # Flatten paths and refuse anything trying to escape the folder.
            name = Path(m).name
            with zf.open(m) as src, open(out / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"Extracted {len(members)} CSV files to {out}")


def cmd_fetch(args: argparse.Namespace) -> None:
    d = DATASETS[args.dataset] if args.dataset in DATASETS else None
    out = _target(args.dataset)
    url = args.url or (d or {}).get("url")
    if not url:
        print(f"{args.dataset} needs a manual step. Open:\n  {d['page']}\n"
              "download the CSV zip, then run:\n"
              f"  python data/download.py extract {args.dataset} /path/to/file.zip\n"
              "or re-run fetch with --url <direct zip link>.")
        return
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "download.zip"
    print(f"Downloading {url}\n  -> {zip_path}")
    urllib.request.urlretrieve(url, zip_path)  # noqa: S310 (user-chosen URL)
    extract(args.dataset, zip_path)
    zip_path.unlink()


def cmd_extract(args: argparse.Namespace) -> None:
    extract(args.dataset, Path(args.zip))


def cmd_verify(args: argparse.Namespace) -> None:
    """Check CSVs exist and contain a Label column; print row counts."""
    import pandas as pd

    out = _target(args.dataset)
    files = sorted(out.rglob("*.csv"))
    if not files:
        sys.exit(f"No CSVs in {out}")
    total = 0
    for f in files:
        head = pd.read_csv(f, nrows=0, encoding="latin-1")
        has_label = any(c.strip().lower() == "label" for c in head.columns)
        with open(f, "rb") as fh:
            rows = sum(1 for _ in fh) - 1
        total += rows
        print(f"{f.name:60s} rows={rows:>9,d} cols={len(head.columns):3d} label={'yes' if has_label else 'NO'}")
    print(f"TOTAL rows: {total:,d}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    f = sub.add_parser("fetch")
    f.add_argument("dataset")
    f.add_argument("--url", default=None)
    f.set_defaults(func=cmd_fetch)
    e = sub.add_parser("extract")
    e.add_argument("dataset")
    e.add_argument("zip")
    e.set_defaults(func=cmd_extract)
    v = sub.add_parser("verify")
    v.add_argument("dataset")
    v.set_defaults(func=cmd_verify)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
