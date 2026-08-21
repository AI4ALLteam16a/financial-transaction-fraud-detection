"""
Loads the raw Kaggle CSV/JSON files from ../data and builds the working
samples used to train the three models and to power the Streamlit app.

The three notebooks (logistic_regression, random_forest_fraud_detection,
svm_fraud_detection) each do their own independent merge/sample of the raw
data. To keep the app fast and the three models comparable, we build ONE
shared stratified base sample here (same recipe the random-forest notebook
used: 200k rows, stratified on the fraud label, random_state=42) and reuse
it as the training base for all three models. See streamlit_app/README.md
for why, and what this trades off against the original notebooks' full-scale
runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

BASE_SAMPLE_SIZE = 200_000
RANDOM_STATE = 42


def _require_data_files() -> None:
    required = [
        "transactions_data.csv",
        "cards_data.csv",
        "users_data.csv",
        "train_fraud_labels.json",
    ]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing data file(s) in 'data/': "
            + ", ".join(missing)
            + ". Download the dataset per data/README.md and place the files there."
        )


def load_labels() -> pd.DataFrame:
    """id (str), fraud_label ('Yes'/'No'), fraud (0/1)."""
    with open(DATA_DIR / "train_fraud_labels.json") as f:
        raw = json.load(f)["target"]
    labels = pd.DataFrame({"id": list(raw.keys()), "fraud_label": list(raw.values())})
    labels["fraud"] = (labels["fraud_label"] == "Yes").astype(int)
    return labels


def load_base_sample(
    sample_size: int = BASE_SAMPLE_SIZE, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Transactions inner-joined with fraud labels, then a stratified subsample.

    Mirrors random_forest_fraud_detection.ipynb's sampling step so all three
    models train on the same base rows.
    """
    _require_data_files()
    transactions = pd.read_csv(DATA_DIR / "transactions_data.csv")
    transactions["id"] = transactions["id"].astype(str)

    labels = load_labels()
    merged = transactions.merge(labels[["id", "fraud"]], on="id", how="inner")

    sample, _ = train_test_split(
        merged,
        train_size=sample_size,
        stratify=merged["fraud"],
        random_state=random_state,
    )
    return sample.reset_index(drop=True)


def attach_cards_and_users(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join card and user attributes needed by the logistic-regression
    model and by the app's unified transaction schema."""
    _require_data_files()
    cards = pd.read_csv(DATA_DIR / "cards_data.csv")
    users = pd.read_csv(DATA_DIR / "users_data.csv")

    out = df.merge(
        cards, left_on="card_id", right_on="id", how="left", suffixes=("", "_card")
    )
    out = out.merge(
        users, left_on="client_id", right_on="id", how="left", suffixes=("", "_user")
    )
    return out


def build_app_frame(sample_size: int = BASE_SAMPLE_SIZE) -> pd.DataFrame:
    """The single fully-joined frame (transactions + cards + users + label)
    that both the training script and the app's sample browser draw from."""
    base = load_base_sample(sample_size=sample_size)
    return attach_cards_and_users(base)
