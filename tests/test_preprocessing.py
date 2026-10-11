"""Unit tests for aegisflow.preprocessing (run: pytest -q from the repo root)."""
import numpy as np
import pandas as pd
import pytest

from aegisflow.config import Phase1Config
from aegisflow.preprocessing import (
    LABEL_COL,
    FeaturePreprocessor,
    apply_smote,
    canonical_name,
    clean_dataframe,
    map_label,
    normalise_columns,
    split_data,
    stratified_sample,
)
from aegisflow.synthetic import make_synthetic_cicids


@pytest.mark.parametrize(
    "raw,expected",
    [
        (" Flow Bytes/s", "flow_bytes_s"),
        (" Total Fwd Packets", "total_fwd_packets"),
        ("Total Fwd Packet", "total_fwd_packets"),   # improved-dataset spelling
        (" Destination Port", "dst_port"),
        ("Flow ID", "flow_id"),
        (" CWE Flag Count", "cwr_flag_count"),
    ],
)
def test_canonical_name(raw, expected):
    assert canonical_name(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BENIGN", "BENIGN"),
        ("DDoS", "DDoS"),
        ("DoS Hulk", "DoS"),
        ("DoS slowloris", "DoS"),
        ("PortScan", "PortScan"),
        ("FTP-Patator", "BruteForce"),
        ("SSH-Patator", "BruteForce"),
        ("Web Attack \x96 Brute Force", "WebAttack"),  # must NOT become BruteForce
        ("Web Attack � XSS", "WebAttack"),
        ("Bot", "Bot"),
        ("Infiltration - Portscan", "Infiltration"),
        ("Heartbleed", "Heartbleed"),
        ("DoS Hulk - Attempted", "BENIGN"),
    ],
)
def test_map_label_family(raw, expected):
    assert map_label(raw, "family", attempted_as_benign=True) == expected


def test_map_label_attempted_merged_into_attack_never_separate():
    assert map_label("DoS Hulk - Attempted", "family", attempted_as_benign=False) == "DoS"
    assert map_label("Web Attack - XSS - Attempted", "family", attempted_as_benign=False) == "WebAttack"
    assert map_label("DoS Hulk - Attempted", "fine", attempted_as_benign=False) == "DoS Hulk"


def test_normalise_columns_drops_duplicate_column():
    df = normalise_columns(make_synthetic_cicids(100))
    assert "fwd_header_length_1" not in df.columns
    assert "total_fwd_packets" in df.columns and "label" in df.columns


def _clean(n=2000, **cfg_kwargs):
    cfg = Phase1Config(min_class_samples=5, **cfg_kwargs)
    raw = normalise_columns(make_synthetic_cicids(n, improved=True))
    return clean_dataframe(raw, cfg)


def test_clean_removes_leakage_inf_nan_duplicates():
    df, rep = _clean()
    for col in ("flow_id", "src_ip", "timestamp", "dst_port"):
        assert col not in df.columns
    feats = [c for c in df.columns if c not in (LABEL_COL, "source_file")]
    assert np.isfinite(df[feats].to_numpy()).all()
    assert rep.inf_values_found == 5
    assert rep.rows_with_nan_or_inf_dropped >= 5
    assert rep.duplicate_rows_dropped >= 1
    assert not df.duplicated(subset=feats + [LABEL_COL]).any()
    assert rep.rows_out == len(df)


def test_keep_dst_port_option():
    df, _ = _clean(keep_dst_port=True)
    assert "dst_port" in df.columns


def test_conflicting_rows_dropped():
    cfg = Phase1Config(min_class_samples=1)
    df = pd.DataFrame({"a": [1.0, 1.0, 2.0, 3.0], "b": [0.0, 0.0, 1.0, 1.0],
                       "label": ["BENIGN", "DDoS", "BENIGN", "DDoS"]})
    out, rep = clean_dataframe(df, cfg)
    assert rep.conflicting_rows_dropped == 2
    assert len(out) == 2


def test_rare_class_dropped():
    cfg = Phase1Config(min_class_samples=3)
    df = pd.DataFrame({"a": np.arange(10, dtype=float),
                       "label": ["BENIGN"] * 8 + ["Heartbleed"] * 2})
    out, rep = clean_dataframe(df, cfg)
    assert rep.rare_classes_dropped == {"Heartbleed": 2}
    assert set(out[LABEL_COL]) == {"BENIGN"}


def test_stratified_sample_keeps_rare_classes():
    df = pd.DataFrame({"x": np.arange(1100.0), "label": ["BENIGN"] * 1000 + ["Bot"] * 100})
    s = stratified_sample(df, frac=0.1, min_per_class=50, max_per_class=None, seed=0)
    counts = s[LABEL_COL].value_counts()
    assert counts["BENIGN"] == 100       # 10% of 1000
    assert counts["Bot"] == 50           # floor applies
    s2 = stratified_sample(df, frac=0.1, min_per_class=500, max_per_class=None, seed=0)
    assert s2[LABEL_COL].value_counts()["Bot"] == 100  # whole class kept


def test_split_is_stratified_and_disjoint():
    df, _ = _clean(3000)
    df = df.reset_index(drop=True)
    df["row"] = np.arange(len(df))
    tr, va, te = split_data(df, 0.15, 0.15, seed=0)
    assert len(tr) + len(va) + len(te) == len(df)
    assert set(tr.row).isdisjoint(te.row) and set(tr.row).isdisjoint(va.row)
    assert set(te[LABEL_COL]) == set(df[LABEL_COL])


def test_feature_preprocessor_fits_on_train_only():
    tr = pd.DataFrame({"a": [1.0, 2.0], "const": [0.0, 0.0], "label": ["BENIGN", "Bot"]})
    te = pd.DataFrame({"a": [3.0], "const": [5.0], "label": ["Bot"]})
    prep = FeaturePreprocessor().fit(tr)
    assert prep.constant_columns_ == ["const"]
    X = prep.transform(te)
    assert X.shape == (1, 1) and X.dtype == np.float32


def test_smote_only_grows_minority():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3)).astype(np.float32)
    y = np.array([0] * 100 + [1] * 20)
    Xr, yr = apply_smote(X, y, target=60, seed=0)
    assert (yr == 0).sum() == 100 and (yr == 1).sum() == 60


def test_benign_load_frac_keeps_all_attacks(tmp_path):
    from aegisflow.preprocessing import load_csv_folder

    raw = make_synthetic_cicids(2000, seed=5)
    raw.to_csv(tmp_path / "day.csv", index=False)
    full = load_csv_folder(tmp_path)
    part = load_csv_folder(tmp_path, benign_load_frac=0.2, seed=0)
    n_att_full = (full[LABEL_COL] != "BENIGN").sum()
    assert (part[LABEL_COL] != "BENIGN").sum() == n_att_full
    ratio = (part[LABEL_COL] == "BENIGN").sum() / (full[LABEL_COL] == "BENIGN").sum()
    assert 0.1 < ratio < 0.3


def _load_download_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data" / "download.py"
    spec = importlib.util.spec_from_file_location("download", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extract_accepts_folder_and_rejects_non_zip(tmp_path, monkeypatch):
    dl = _load_download_module()
    monkeypatch.setattr(dl, "RAW_DIR", tmp_path / "raw")
    src = tmp_path / "unzipped" / "MachineLearningCVE"
    src.mkdir(parents=True)
    (src / "Monday.csv").write_text("a, Label\n1,BENIGN\n")
    dl.extract("cicids2017_original", tmp_path / "unzipped")
    assert (tmp_path / "raw" / "cicids2017_original" / "Monday.csv").exists()

    fake = tmp_path / "page.zip"
    fake.write_text("<html>please register</html>")
    with pytest.raises(RuntimeError):
        dl.extract("cicids2017_original", fake)


def _range_server(directory):
    """Tiny local HTTP server that supports Range requests (like real servers)."""
    import http.server
    import threading

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.translate_path(self.path)
            data = open(path, "rb").read()
            rng = self.headers.get("Range")
            if rng:
                start, end = rng.replace("bytes=", "").split("-")
                start, end = int(start), min(int(end), len(data) - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                body = data[start:end + 1]
            else:
                self.send_response(200)
                body = data
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_remote_zip_extracts_only_chosen_days(tmp_path, monkeypatch):
    import zipfile

    dl = _load_download_module()
    monkeypatch.setattr(dl, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(dl, "_use_system_certificates", lambda: None)
    srv_dir = tmp_path / "srv"
    srv_dir.mkdir()
    with zipfile.ZipFile(srv_dir / "all.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for day in ("Wednesday-14-02-2018", "Tuesday-20-02-2018", "Friday-02-03-2018"):
            zf.writestr(f"CSECICIDS2018/{day}.csv", "a,Label\n" + "1,BENIGN\n" * 1000)
    srv = _range_server(srv_dir)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/all.zip"
        names = dl.remote_zip_extract("cicids2018_improved", url, ["14-02-2018", "02-03-2018"])
    finally:
        srv.shutdown()
    out = tmp_path / "raw" / "cicids2018_improved"
    assert len(names) == 2
    assert sorted(p.name for p in out.iterdir()) == ["Friday-02-03-2018.csv", "Wednesday-14-02-2018.csv"]
    assert (out / "Friday-02-03-2018.csv").read_text().count("BENIGN") == 1000


def test_fetch_per_day_uses_default_days(tmp_path, monkeypatch):
    import argparse

    dl = _load_download_module()
    monkeypatch.setattr(dl, "RAW_DIR", tmp_path / "raw")
    got = []
    monkeypatch.setattr(dl, "_download_file", lambda url, dest: got.append((url, dest.name)))
    dl.cmd_fetch(argparse.Namespace(dataset="cicids2018_original", url=None, days=None))
    assert len(got) == 6
    assert got[0][0].endswith("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv")
    assert got[0][1] == "Wednesday-14-02-2018.csv"


def test_aws_2018_abbreviated_columns_map_to_cic_names():
    from aegisflow.preprocessing import canonical_name

    for raw, canon in [("Tot Fwd Pkts", "total_fwd_packets"), ("TotLen Bwd Pkts", "total_length_bwd_packets"),
                       ("Flow Byts/s", "flow_bytes_s"), ("Init Fwd Win Byts", "fwd_init_win_bytes"),
                       ("Pkt Size Avg", "avg_packet_size"), ("Fwd IAT Tot", "fwd_iat_total")]:
        assert canonical_name(raw) == canon


def test_remote_zip_rejects_web_page_link():
    dl = _load_download_module()
    with pytest.raises(RuntimeError, match="web page"):
        dl.remote_zip_extract("cicids2018_improved",
                              "https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html", [])


def test_remote_zip_lists_a_web_folder(tmp_path, monkeypatch, capsys):
    dl = _load_download_module()
    monkeypatch.setattr(dl, "_use_system_certificates", lambda: None)
    for v in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        monkeypatch.delenv(v, raising=False)
    srv_dir = tmp_path / "srv" / "Datasets"
    srv_dir.mkdir(parents=True)
    (srv_dir / "CSE-CIC-IDS2018.zip").write_bytes(b"x" * 2000)
    (srv_dir / "index.html").write_text(
        '<a href="../">Parent</a> <a href="CSE-CIC-IDS2018.zip">CSE-CIC-IDS2018.zip</a> <a href="?C=N;O=D">Name</a>')
    srv = _range_server(tmp_path / "srv")
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/Datasets/"
        monkeypatch.setattr(dl.urllib.request, "urlopen", _index_aware_urlopen(dl.urllib.request.urlopen, base))
        links = dl.remote_zip_extract("cicids2018_improved", base, [])
    finally:
        srv.shutdown()
    assert links == []
    out = capsys.readouterr().out
    assert "CSE-CIC-IDS2018.zip" in out and "Parent" not in out


def _index_aware_urlopen(real_urlopen, base):
    """The test server has no directory index: map 'folder/' to 'folder/index.html'."""
    import urllib.request

    def opener(req, *a, **kw):
        url = req.full_url if isinstance(req, urllib.request.Request) else req
        if url == base:
            url = base + "index.html"
            req = url if not isinstance(req, urllib.request.Request) else urllib.request.Request(
                url, headers=dict(req.header_items()), method=req.get_method())
        return real_urlopen(req, *a, **kw)
    return opener


def test_fetch_direct_csv_links(tmp_path, monkeypatch):
    import argparse

    dl = _load_download_module()
    monkeypatch.setattr(dl, "RAW_DIR", tmp_path / "raw")
    got = []
    monkeypatch.setattr(dl, "_download_file", lambda url, dest: got.append(dest.name))
    dl.cmd_fetch(argparse.Namespace(dataset="cicids2018_improved", days=None, url=[
        "https://x/Datasets/CSE-CIC-IDS2018/Wednesday-14-02-2018.csv",
        "https://x/Datasets/CSE-CIC-IDS2018/Friday%2002-03-2018.csv"]))
    assert got == ["Wednesday-14-02-2018.csv", "Friday 02-03-2018.csv"]
