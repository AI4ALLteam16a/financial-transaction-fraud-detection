# Fraud Detection — Streamlit App

Compares the three models from `logistic_regression.ipynb`,
`random_forest_fraud_detection.ipynb`, and `svm_fraud_detection.ipynb` side
by side: pick a transaction (or upload a CSV of them) and see each model's
fraud probability and flag.

## Setup

1. Make sure the raw dataset is in `../data/` (see `../data/README.md` —
   download from Kaggle, the files are gitignored/too large to commit).
2. From this folder, install dependencies (already present in this repo's
   environment, but for a fresh one):
   ```
   pip install -r requirements.txt
   ```
3. Train and export the models (one-time, ~a few minutes):
   ```
   python train_models.py
   ```
   This writes `models/*.joblib`, `models/metrics.json`,
   `models/mcc_risk_map.json`, and the sample data under `data_samples/`.
   Re-run it any time `src/features.py` or `src/data_loader.py` change.
4. Launch the app:
   ```
   streamlit run app.py
   ```

## Why this isn't a 1:1 rerun of the three notebooks

Two things made a direct "load each notebook's exact pipeline" approach
impractical, so `train_models.py` reproduces each model's feature set and
hyperparameters but adapts how the data is prepared:

- **None of the three notebooks saved their fitted model or preprocessing
  objects** (no `joblib`/`pickle` calls) — everything was trained in-memory
  and lost on kernel restart. `train_models.py` retrains all three from
  scratch and actually persists them.
- **The notebooks trained at three different, inconsistent scales**:
  logistic regression used the full ~8.9M-row merged dataset with SMOTE,
  random forest used a 200k-row stratified subsample, and SVM used a
  53k-row 3:1-undersampled set built from the full 13.3M-row table. Training
  logistic regression + SMOTE on 8.9M rows from a plain script would be slow
  and memory-heavy. Instead, all three models here train from **one shared
  200k-row stratified sample** (`src/data_loader.py`, same recipe the RF
  notebook used), with each model then applying its own feature
  engineering/encoding/undersampling on top. This keeps training fast
  (~1–2 min total) and the three models comparable, at the cost of not
  exactly matching each notebook's originally reported metrics — expect
  broadly similar but not identical numbers, especially for logistic
  regression (much less SMOTE-resampled data) and SVM (far fewer rows to
  undersample from, so a smaller effective training set).

One encoding change was also made for robustness rather than fidelity:
the logistic regression notebook used `pandas.get_dummies`, which breaks
(or silently misaligns columns) on a category it didn't see during
training. The app instead uses
`OneHotEncoder(handle_unknown="ignore")`, so an unfamiliar card brand,
merchant city, etc. in a browsed/uploaded row degrades gracefully instead
of crashing.

## Files

```
streamlit_app/
  app.py                  # the Streamlit UI
  train_models.py         # fits & exports the three models (run this first)
  requirements.txt
  src/
    data_loader.py         # raw data loading + shared stratified sample
    features.py             # feature derivation + the 3 model pipelines
  models/                   # generated: *.joblib, metrics.json, mcc_risk_map.json (gitignored)
  data_samples/             # generated: sample_transactions.csv, upload_template.csv (gitignored)
```

## Decision thresholds

Each model's probability → fraud/legitimate cutoff defaults to whichever
threshold maximized F1 on that model's own test split (see the "Model
comparison" tab), not a flat 0.5 — fraud detection is heavily imbalanced,
so 0.5 is rarely the right cutoff. The threshold used is shown under each
prediction.
