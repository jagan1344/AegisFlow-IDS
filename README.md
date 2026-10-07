# AegisFlow IDS

A confidence-aware, explainable, zero-day-aware network intrusion detection
system with an LLM-powered SOC Copilot. It is free and software-only: the
models train on Colab's free tier and the app runs on an ordinary laptop.

> Status: **Phase 1 of 9 is done** (data pipeline and honest baselines).
> Phases 2–9 (cross-dataset tests, open-set detection, confidence engine,
> explainability and LLM copilot, alert correlation, drift, app,
> documentation) come later.

## Phase 1: what it does

```
raw CSVs ─► normalise column names ─► map labels to families ─► drop leakage columns
        ─► NaN/inf rows ─► duplicates ─► conflicting labels ─► rare classes
        ─► stratified per-class sample ─► stratified train/val/test (70/15/15)
        ─► FeaturePreprocessor (fit on TRAIN only: drop constant columns)
        ─► imbalance handling (balanced sample weights | SMOTE on TRAIN only)
        ─► LogReg · RandomForest · XGBoost · LightGBM · MLP
        ─► metrics: per-class P/R/F1, macro-F1, confusion matrix, PR-AUC,
                    FPR, detection rate, latency (ms/flow), model size (MB)
```

* **Leakage removed from inputs:** Flow ID, Src/Dst IP, Src Port, Timestamp,
  row id, and Dst Port by default (`--keep-dst-port` turns on an ablation
  that keeps it).
* **Label families:** BENIGN, DoS, DDoS, PortScan, BruteForce, WebAttack,
  Bot, Infiltration, Heartbleed. In the corrected dataset, "– Attempted"
  flows count as BENIGN by default (`attempted_as_benign`).
* **Everything is recorded:** each run writes `results/phase1_<dataset>.json`
  with the config, what every cleaning step removed, sample sizes and all
  metrics.

## Layout

```
AegisFlow-IDS/
  aegisflow/   config.py  preprocessing.py  metrics.py  benchmark.py
               synthetic.py (test data only)  models/baselines.py
  data/        download.py  DATASETS.md          (raw data is git-ignored)
  notebooks/   01_phase1_baselines.ipynb          (Colab)
  scripts/     run_phase1.py  compare_phase1.py
  tests/       pytest suite
```

## Run it

```bash
git clone https://github.com/jagan1344/AegisFlow-IDS.git && cd AegisFlow-IDS
pip install -r requirements.txt
pytest -q                                        # unit tests, about 10 s
python scripts/run_phase1.py --synthetic         # smoke test on FAKE data (not results)

python data/download.py list                     # dataset sources and licences
python data/download.py fetch cicids2017_original
python data/download.py extract cicids2017_improved ~/Downloads/<zip>   # after manual download
python scripts/run_phase1.py --dataset cicids2017_original
python scripts/run_phase1.py --dataset cicids2017_improved
python scripts/compare_phase1.py --original results/phase1_cicids2017_original.json \
                                 --improved results/phase1_cicids2017_improved.json
```

On a laptop with little RAM, add `--max-rows-per-file 200000` or
`--sample-frac 0.1`.

## Results

No real-data results are reported yet. Run the notebook and paste the
tables from `results/phase1_comparison.md`. The synthetic smoke test only
checks that the code runs and says nothing about IDS performance.
