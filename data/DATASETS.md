# Datasets used by AegisFlow (all free for academic use)

No raw data is committed. Files go to `data/raw/<key>/` (git-ignored).
Run `python data/download.py list` for the same information.

| Key | Dataset | Where to get it | Cost / terms | Used in |
|---|---|---|---|---|
| `cicids2017_original` | CIC-IDS2017, `MachineLearningCSV.zip` (78 CICFlowMeter-V3 features, ~2.8 M flows, Mon–Fri 3–7 July 2017) | Official page: https://www.unb.ca/cic/datasets/ids-2017.html. The page links to the CIC file server, and `download.py` uses the long-standing direct path `http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip`. | Free. Cite Sharafaldin, Lashkari, Ghorbani, ICISSP 2018. | 1, 2, 3, 4, 7 |
| `cicids2017_improved` | CIC-IDS2017 regenerated with a fixed CICFlowMeter and corrected labels, including new "– Attempted" labels for attacks that carried no payload | KU Leuven imec-DistriNet (Engelen, Rimmer, Joosen). Project page: https://intrusion-detection.distrinet-research.be/CNS2022/ (the older WTMC 2021 release was at `downloads.distrinet-research.be/WTMC2021`). | Free. Cite Engelen et al., *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study*, IEEE S&P Workshops (WTMC) 2021, and Lanvin et al., CRiSIS 2022. | 1 (comparison), 2–4 |
| alt. corrected version | LYCOS-IDS2017 (Univ. Le Mans) | http://lycos-ids.univ-lemans.fr/ | Free | optional alternative |
| `cicids2018_original` | CSE-CIC-IDS2018 original per-day CSVs (abbreviated column names, NOT label-corrected) | Public AWS folder, no account: `python data/download.py fetch cicids2018_original` downloads 6 days (14-02, 16-02, 21-02, 22-02, 28-02, 02-03), 108–383 MB each, about 1.9 GB in total (sizes checked 2026-10-11) | Free; cite Sharafaldin et al. 2018 | 2 (fallback) |
| `unsw_nb15` | UNSW-NB15 partition files `UNSW_NB15_training-set.csv` (175,341 rows) + `UNSW_NB15_testing-set.csv` (82,332 rows), Argus/Bro features | https://research.unsw.edu.au/projects/unsw-nb15-dataset (folder *Training and Testing Sets*) | Free for research; cite Moustafa & Slay, MilCIS 2015 | 2 |
| `cicids2018_improved` | CSE-CIC-IDS2018, corrected by the same DistriNet project (same feature extractor as `cicids2017_improved`, so features match 1:1) | Same DistriNet page | Free | 2 |

## Honest notes on verification

* The links were checked on 2026-10-07 through web search and the dataset
  papers. The sandbox this code was written in blocks those domains, so the
  files could **not** be downloaded or opened from there. Check each link
  in a browser or Colab before relying on it, and fix `DATASETS` in
  `download.py` if a provider has moved its files.
* Some providers show a short form or terms page before downloading. Use
  `python data/download.py extract <key> <zip>` once you have the zip.
* **Cross-dataset feature spaces (Phase 2).** CIC-IDS2017 and CSE-CIC-IDS2018
  (both DistriNet-corrected) come from the same fixed CICFlowMeter and share
  all features. UNSW-NB15 uses Argus/Bro features, so `aegisflow/crossdataset.py`
  maps the 14 features both tools measure (durations, packet/byte counts,
  mean sizes, rates, inter-arrival times, TCP windows) into a shared space with
  unit conversion. The mapping is approximate, which is a stated limitation.
* The corrected datasets are described in Liu, Engelen, Lynar, Essam, Joosen,
  *Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and
  CSE-CIC-IDS-2018*, IEEE CNS 2022.

## Sample size used in Colab (filled in from your run's JSON)

Phase 1 uses per-class stratified sampling (`config.Phase1Config`):
`n_class = clip(0.3 × count, 5 000, 200 000)`. Classes smaller than 5 000
rows are kept whole. The exact counts before and after sampling are written
to `results/phase1_<dataset>.json → sampling`. Copy them into the report
from there; do not retype them.

## Getting only some days of a big zip

The corrected CSE-CIC-IDS2018 is distributed as one large archive. If the
server supports partial downloads, only the chosen day files are downloaded:

```bash
python data/download.py remote-zip cicids2018_improved "<zip link>"                 # list files
python data/download.py remote-zip cicids2018_improved "<zip link>" --match 14-02-2018 16-02-2018 21-02-2018 22-02-2018 28-02-2018 02-03-2018
```
