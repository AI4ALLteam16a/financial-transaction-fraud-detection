"""
Fits and exports the three fraud models used by the Streamlit app.

Run this once (and again any time src/features.py or src/data_loader.py
change) before `streamlit run app.py`:

    python train_models.py

Produces, under streamlit_app/models/:
    logistic_regression.joblib, random_forest.joblib, svm.joblib
    mcc_risk_map.json
    metrics.json
And under streamlit_app/data_samples/:
    sample_transactions.csv   (for the app's "browse a transaction" picker)
    upload_template.csv       (empty-row template for the CSV batch upload)

See streamlit_app/README.md for how this training setup relates to (and
intentionally departs from) the three original notebooks.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import data_loader, features  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
SAMPLES_DIR = APP_DIR / "data_samples"
RANDOM_STATE = 42


def log(msg: str) -> None:
    print(f"[train_models] {msg}", flush=True)


def evaluate(pipeline, X_test, y_test) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]

    # threshold that maximizes F1 on the test set, like the RF notebook did
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_test, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)

    preds_default = (proba >= 0.5).astype(int)
    preds_tuned = (proba >= best_t).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "at_0.5": {
            "accuracy": float(accuracy_score(y_test, preds_default)),
            "precision": float(precision_score(y_test, preds_default, zero_division=0)),
            "recall": float(recall_score(y_test, preds_default, zero_division=0)),
            "f1": float(f1_score(y_test, preds_default, zero_division=0)),
        },
        "at_tuned_threshold": {
            "threshold": best_t,
            "accuracy": float(accuracy_score(y_test, preds_tuned)),
            "precision": float(precision_score(y_test, preds_tuned, zero_division=0)),
            "recall": float(recall_score(y_test, preds_tuned, zero_division=0)),
            "f1": float(best_f1),
        },
        "n_test_rows": int(len(y_test)),
        "n_test_fraud": int(y_test.sum()),
        "recommended_threshold": best_t,
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log("Loading & merging raw data (transactions + labels + cards + users)...")
    frame = data_loader.build_app_frame()
    log(f"Base sample: {len(frame):,} rows, fraud rate {frame['fraud'].mean():.4%} "
        f"({time.time() - t0:.0f}s)")

    log("Deriving shared features (amount/date transforms, etc.)...")
    frame = features.derive_common_fields(frame)

    log("Building mcc -> risk-tier map for the logistic regression model...")
    mcc_map = features.build_mcc_risk_map(frame)
    frame = features.apply_mcc_risk_tier(frame, mcc_map)
    with open(MODELS_DIR / "mcc_risk_map.json", "w") as f:
        json.dump(mcc_map, f)

    metrics: dict = {}

    # -- Logistic Regression ------------------------------------------------
    log("Training logistic regression (SMOTE)...")
    X = frame[features.LOGREG_NUMERIC + features.LOGREG_CATEGORICAL]
    y = frame["fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logreg = features.build_logreg_pipeline()
    logreg.fit(X_train, y_train)
    metrics["logistic_regression"] = evaluate(logreg, X_test, y_test)
    joblib.dump(logreg, MODELS_DIR / "logistic_regression.joblib")
    log(f"  logreg roc_auc={metrics['logistic_regression']['roc_auc']:.4f}")

    # -- Random Forest --------------------------------------------------------
    log("Training random forest...")
    X = frame[features.RF_NUMERIC + features.RF_CATEGORICAL]
    y = frame["fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    rf = features.build_rf_pipeline()
    rf.fit(X_train, y_train)
    metrics["random_forest"] = evaluate(rf, X_test, y_test)
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    log(f"  rf roc_auc={metrics['random_forest']['roc_auc']:.4f}")

    # -- SVM ------------------------------------------------------------------
    log("Training SVM (3:1 undersampled, calibrated for probabilities)...")
    svm_frame = features.undersample_3to1(frame)
    X = svm_frame[features.SVM_NUMERIC + features.SVM_CATEGORICAL]
    y = svm_frame["fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    svm = features.build_svm_pipeline()
    svm.fit(X_train, y_train)
    metrics["svm"] = evaluate(svm, X_test, y_test)
    joblib.dump(svm, MODELS_DIR / "svm.joblib")
    log(f"  svm roc_auc={metrics['svm']['roc_auc']:.4f}")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log("Saved models/metrics.json")

    # -- App-facing sample data -------------------------------------------
    log("Building sample_transactions.csv for the app's transaction picker...")
    app_cols = ["id", "fraud"] + features.RAW_INPUT_COLUMNS
    available_cols = [c for c in app_cols if c in frame.columns]
    fraud_rows = frame[frame["fraud"] == 1]
    legit_rows = frame[frame["fraud"] == 0]
    n_fraud = min(len(fraud_rows), 500)
    n_legit = min(len(legit_rows), 2500)
    sample = pd.concat(
        [
            fraud_rows.sample(n=n_fraud, random_state=RANDOM_STATE),
            legit_rows.sample(n=n_legit, random_state=RANDOM_STATE),
        ]
    ).sample(frac=1, random_state=RANDOM_STATE)
    sample[available_cols].to_csv(SAMPLES_DIR / "sample_transactions.csv", index=False)
    log(f"  wrote {len(sample):,} rows ({n_fraud} fraud / {n_legit} legit)")

    template = pd.DataFrame(columns=features.RAW_INPUT_COLUMNS)
    template.to_csv(SAMPLES_DIR / "upload_template.csv", index=False)
    log("  wrote upload_template.csv")

    log(f"Done in {time.time() - t0:.0f}s total.")


if __name__ == "__main__":
    main()
