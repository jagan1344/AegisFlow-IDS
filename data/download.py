"""Download / verify helper for the free datasets used by AegisFlow.

No raw data is stored in git. Every dataset lands in data/raw/<key>/.

Usage (from the repository root):
    python data/download.py list
    python data/download.py fetch cicids2017_original            # default URL
    python data/download.py fetch cicids2017_improved --url <zip>  # URL from the site
    python data/download.py verify cicids2017_original
    python data/download.py fetch cicids2018_original               # 6 days from AWS
    python data/download.py remote-zip cicids2018_improved <zip-url>  # list a remote zip
    python data/download.py remote-zip cicids2018_improved <zip-url> --match 14-02-2018 16-02-2018

Some providers put the files behind a short registration/terms page. For
those the script prints the page to open; download the zip yourself, then:
    python data/download.py extract cicids2017_improved /path/to/file.zip
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"

# Public, no-account AWS folder of the original CSE-CIC-IDS2018 CSVs.
_AWS_2018 = ("https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
             "Processed%20Traffic%20Data%20for%20ML%20Algorithms")

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
    "cicids2018_original": {
        "name": "CSE-CIC-IDS2018 original (CIC/AWS, per-day CSVs, NOT label-corrected)",
        "page": "https://www.unb.ca/cic/datasets/ids-2018.html",
        "url": None,
        "day_url": _AWS_2018 + "/{day}_TrafficForML_CICFlowMeter.csv",
        # One day per attack family (Tuesday 20-02 is not in the AWS folder).
        "default_days": ["Wednesday-14-02-2018", "Friday-16-02-2018", "Wednesday-21-02-2018",
                         "Thursday-22-02-2018", "Wednesday-28-02-2018", "Friday-02-03-2018"],
        "license": "Free for research; cite Sharafaldin et al., ICISSP 2018.",
        "phase": "2",
    },
    "unsw_nb15": {
        "name": "UNSW-NB15 partition CSVs (UNSW_NB15_training-set.csv + UNSW_NB15_testing-set.csv)",
        "page": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "url": None,  # download link on the page (folder 'Training and Testing Sets')
        "license": "Free for research; cite Moustafa & Slay, MilCIS 2015.",
        "phase": "2",
    },
}


def _use_system_certificates() -> None:
    """Trust the operating system's certificate store (Windows/macOS/Linux).

    Antivirus "HTTPS scanning" and college/office networks re-sign web
    traffic with their own certificate. Browsers trust it (it is in the
    Windows store) but Python's built-in list does not, which shows up as
    CERTIFICATE_VERIFY_FAILED / "self-signed certificate in certificate
    chain". `truststore` makes Python use the same store as the browser.
    Verification stays ON; we only change which trusted list is used.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass  # optional dependency; download may still work without it


def _target(key: str) -> Path:
    if key not in DATASETS:
        sys.exit(f"Unknown dataset {key!r}. Run: python data/download.py list")
    return RAW_DIR / key


def cmd_list(_: argparse.Namespace) -> None:
    for key, d in DATASETS.items():
        print(f"{key}  [phase {d['phase']}]\n  {d['name']}\n  page:    {d['page']}")
        print(f"  url:     {d['url'] or '(manual: copy link from page)'}\n  license: {d['license']}\n")


def extract(key: str, zip_path: Path) -> None:
    """Unzip into data/raw/<key>/ (CSV files only).

    `zip_path` may also be a FOLDER that already contains the CSV files
    (e.g. you unzipped it yourself); they are copied in.
    """
    out = _target(key)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = Path(zip_path)
    if zip_path.is_dir():
        csvs = sorted(zip_path.rglob("*.csv"))
        for f in csvs:
            shutil.copy2(f, out / f.name)
        print(f"Copied {len(csvs)} CSV files to {out}")
        return
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"{zip_path} is not a zip file (incomplete download, or a web page saved as .zip)")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        for m in members:
            # Flatten paths and refuse anything trying to escape the folder.
            name = Path(m).name
            with zf.open(m) as src, open(out / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"Extracted {len(members)} CSV files to {out}")


def _download_file(url: str, dest: Path) -> None:
    """Stream a URL to disk with a progress line (resumes nothing, but a
    finished file of the right size is not downloaded again)."""
    _use_system_certificates()
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (fixed public URLs)
        total = int(resp.headers.get("Content-Length") or 0)
        if dest.exists() and total and dest.stat().st_size == total:
            print(f"  {dest.name}: already downloaded")
            return
        tmp = dest.with_suffix(dest.suffix + ".part")
        done, step = 0, 50_000_000
        with open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if done // step != (done - len(chunk)) // step:
                    print(f"  {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
    tmp.replace(dest)
    print(f"  {dest.name}: done ({done / 1e6:,.0f} MB)")


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP using Range requests.

    Python's zipfile only needs seek() + read(); a zip keeps its table of
    contents at the END, so with this class we can list a remote zip and
    extract single members while downloading only those bytes.
    """

    def __init__(self, url: str) -> None:
        _use_system_certificates()
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:  # noqa: S310 (user-chosen URL)
            if r.status != 206:
                raise RuntimeError("this server does not support partial downloads "
                                   "(HTTP Range); download the whole file instead")
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:  # noqa: S310
            data = r.read()
        n = len(data)
        b[:n] = data
        self.pos += n
        return n


def list_folder(url: str) -> list[str]:
    """Print the files of a web folder (an 'Index of /...' page) with sizes."""
    import re
    import urllib.parse

    _use_system_certificates()
    with urllib.request.urlopen(url) as r:  # noqa: S310 (user-chosen URL)
        html = r.read().decode("utf-8", "replace")
    links = []
    for href in re.findall(r'href="([^"?#]+)"', html):
        full = urllib.parse.urljoin(url, href)
        if full.startswith(url) and full != url:
            links.append(full)
    links = sorted(set(links))
    print(f"Files and folders in {url}:")
    for link in links:
        size = ""
        if not link.endswith("/"):
            try:
                req = urllib.request.Request(link, method="HEAD")
                with urllib.request.urlopen(req) as h:  # noqa: S310
                    n = int(h.headers.get("Content-Length") or 0)
                size = f"  ({n / 1e9:.2f} GB)" if n > 1e9 else f"  ({n / 1e6:,.0f} MB)"
            except OSError:
                pass
        print(f"  {link}{size}")
    if not links:
        print("  (no links found - open the address in a browser instead)")
    return links


def remote_zip_extract(key: str, url: str, patterns: list[str]) -> list[str]:
    """Extract CSV members whose name contains any pattern from a REMOTE zip.

    With no patterns, only lists the CSV members (name and size).
    """
    if url.split("?")[0].endswith("/"):
        list_folder(url)
        return []
    if url.lower().split("?")[0].endswith((".html", ".htm")):
        raise RuntimeError(
            "that link is a web page, not a file. On the page, right-click the word "
            "'here' (or the 2018 file) and choose 'Copy link address', then use that link.")
    raw = io.BufferedReader(HttpRangeFile(url), buffer_size=8 << 20)
    try:
        zf = zipfile.ZipFile(raw)
    except zipfile.BadZipFile:
        raise RuntimeError(
            "the link does not point to a .zip file (maybe a web page or a .7z/.tar.gz "
            "archive). Send the link to your assistant.") from None
    with zf:
        members = [i for i in zf.infolist() if i.filename.lower().endswith(".csv")]
        if not patterns:
            for i in members:
                print(f"  {i.filename}  ({i.file_size / 1e6:,.0f} MB unzipped, "
                      f"{i.compress_size / 1e6:,.0f} MB to download)")
            return []
        chosen = [i for i in members if any(p.lower() in i.filename.lower() for p in patterns)]
        if not chosen:
            raise RuntimeError(f"no CSV in the zip matches {patterns}; run without --match to list them")
        out = _target(key)
        out.mkdir(parents=True, exist_ok=True)
        for i in chosen:
            print(f"  extracting {i.filename} ({i.compress_size / 1e6:,.0f} MB to download)...")
            with zf.open(i) as src, open(out / Path(i.filename).name, "wb") as dst:
                shutil.copyfileobj(src, dst, 8 << 20)
        return [i.filename for i in chosen]


def cmd_remote_zip(args: argparse.Namespace) -> None:
    try:
        names = remote_zip_extract(args.dataset, args.url, args.match or [])
    except RuntimeError as exc:
        sys.exit(f"[!!] {exc}")
    if names:
        print(f"Extracted {len(names)} CSV files to {_target(args.dataset)}")


def cmd_fetch(args: argparse.Namespace) -> None:
    d = DATASETS[args.dataset] if args.dataset in DATASETS else None
    out = _target(args.dataset)
    if d and d.get("day_url"):
        days = args.days or d["default_days"]
        out.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {len(days)} day files of {args.dataset} into {out}")
        for day in days:
            _download_file(d["day_url"].format(day=day), out / f"{day}.csv")
        return
    urls = args.url if isinstance(args.url, list) else ([args.url] if args.url else [])
    # Direct CSV links (e.g. day files in a web folder): stream each one.
    if urls and all(u.lower().split("?")[0].endswith(".csv") for u in urls):
        out.mkdir(parents=True, exist_ok=True)
        for u in urls:
            name = urllib.request.url2pathname(u.split("?")[0].rsplit("/", 1)[-1])
            _download_file(u, out / Path(name).name)
        return
    url = urls[0] if urls else (d or {}).get("url")
    if not url:
        print(f"{args.dataset} needs a manual step. Open:\n  {d['page']}\n"
              "download the CSV zip, then run:\n"
              f"  python data/download.py extract {args.dataset} /path/to/file.zip\n"
              "or re-run fetch with --url <direct zip link>.")
        return
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "download.zip"
    print(f"Downloading {url}\n  -> {zip_path}")
    _use_system_certificates()
    urllib.request.urlretrieve(url, zip_path)  # noqa: S310 (user-chosen URL)
    if not zipfile.is_zipfile(zip_path):
        head = zip_path.read_bytes()[:300].decode("utf-8", "replace").lower()
        zip_path.unlink()
        if "<html" in head or "<!doctype" in head:
            raise RuntimeError("the server returned a web page instead of the zip "
                               "(the direct link now goes through a registration form)")
        raise RuntimeError("the downloaded file is not a zip (incomplete download?)")
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
    f.add_argument("--url", nargs="+", default=None,
                   help="a zip link, or one or more direct .csv links")
    f.add_argument("--days", nargs="+", default=None,
                   help="per-day datasets only, e.g. Wednesday-14-02-2018")
    f.set_defaults(func=cmd_fetch)
    r = sub.add_parser("remote-zip", help="list / extract chosen CSVs from a zip on a web server")
    r.add_argument("dataset")
    r.add_argument("url")
    r.add_argument("--match", nargs="+", default=None,
                   help="extract CSVs whose name contains any of these, e.g. 14-02-2018 16-02-2018")
    r.set_defaults(func=cmd_remote_zip)
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
