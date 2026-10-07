# Datasets used by AegisFlow (all free for academic use)

No raw data is committed. Files go to `data/raw/<key>/` (git-ignored).
Run `python data/download.py list` for the same information.

| Key | Dataset | Where to get it | Cost / terms | Used in |
|---|---|---|---|---|
| `cicids2017_original` | CIC-IDS2017, `MachineLearningCSV.zip` (78 CICFlowMeter-V3 features, ~2.8 M flows, Mon–Fri 3–7 July 2017) | Official page: https://www.unb.ca/cic/datasets/ids-2017.html. The page links to the CIC file server, and `download.py` uses the long-standing direct path `http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip`. | Free. Cite Sharafaldin, Lashkari, Ghorbani, ICISSP 2018. | 1, 2, 3, 4, 7 |
| `cicids2017_improved` | CIC-IDS2017 regenerated with a fixed CICFlowMeter and corrected labels, including new "– Attempted" labels for attacks that carried no payload | KU Leuven imec-DistriNet (Engelen, Rimmer, Joosen). Project page: https://intrusion-detection.distrinet-research.be/CNS2022/ (the older WTMC 2021 release was at `downloads.distrinet-research.be/WTMC2021`). | Free. Cite Engelen et al., *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study*, IEEE S&P Workshops (WTMC) 2021, and Lanvin et al., CRiSIS 2022. | 1 (comparison), 2–4 |
| alt. corrected version | LYCOS-IDS2017 (Univ. Le Mans) | http://lycos-ids.univ-lemans.fr/ | Free | optional alternative |
| `cicids2018_improved` | CSE-CIC-IDS2018, corrected by the same DistriNet project (same feature extractor as `cicids2017_improved`, so features match 1:1) | Same DistriNet page | Free | 2 |
| `nf_unsw_nb15_v2` | NF-UNSW-NB15-v2 (43 NetFlow features) | Univ. of Queensland: https://staff.itee.uq.edu.au/marius/NIDS_datasets/ (also listed on researchdata.edu.au). A v3 with 53 features also exists. | Free for academic research with citation (Sarhan, Layeghy, Portmann, *Towards a Standard Feature Set for NIDS Datasets*, MONET 2022) | 2 |
| `nf_cse_cic_ids2018_v2` | NF-CSE-CIC-IDS2018-v2 (43 NetFlow features) | same UQ page | same | 2 |

## Honest notes on verification

* The links were checked on 2026-10-07 through web search and the dataset
  papers. The sandbox this code was written in blocks those domains, so the
  files could **not** be downloaded or opened from there. Check each link
  in a browser or Colab before relying on it, and fix `DATASETS` in
  `download.py` if a provider has moved its files.
* Some providers show a short form or terms page before downloading. Use
  `python data/download.py extract <key> <zip>` once you have the zip.
* **Cross-dataset caveat for Phase 2.** UQ publishes no official NetFlow
  (NF-*) version of CIC-IDS2017. The plan is therefore: CIC-IDS2017 →
  CSE-CIC-IDS2018 using the shared CICFlowMeter features (DistriNet versions
  of both), and CIC-IDS2017 → UNSW-NB15 using a hand-mapped set of features
  common to both. That is decided and documented in Phase 2.

## Sample size used in Colab (filled in from your run's JSON)

Phase 1 uses per-class stratified sampling (`config.Phase1Config`):
`n_class = clip(0.3 × count, 5 000, 200 000)`. Classes smaller than 5 000
rows are kept whole. The exact counts before and after sampling are written
to `results/phase1_<dataset>.json → sampling`. Copy them into the report
from there; do not retype them.
