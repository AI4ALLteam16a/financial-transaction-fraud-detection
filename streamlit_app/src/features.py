"""
Feature engineering + pipeline definitions for the three fraud models.

Each model keeps the raw feature set / encoding choices its notebook used
(see streamlit_app/README.md for the mapping), with two deliberate
adaptations so the same code can run at both training time and on
arbitrary new rows in the Streamlit app:

  * pandas.get_dummies (logistic regression notebook) is replaced with
    sklearn's OneHotEncoder(handle_unknown="ignore") so an unseen category
    at inference time doesn't crash or silently misalign columns.
  * Each model is a single sklearn (or imblearn) Pipeline bundling its
    preprocessing + classifier, so it can be joblib-dumped and loaded as
    one object.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

# Raw columns every model (and the app's CSV upload / sample browser) expects
# to find on an input row, before any derived features are computed.
RAW_INPUT_COLUMNS = [
    "date",
    "amount",
    "use_chip",
    "merchant_city",
    "merchant_state",
    "mcc",
    "errors",
    "card_brand",
    "card_type",
    "has_chip",
    "credit_limit",
    "num_cards_issued",
    "credit_score",
    "total_debt",
    "num_credit_cards",
    "yearly_income",
]


def _clean_money(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(
        series.astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce"
    )


def derive_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the date/amount-derived columns every model draws from.
    Safe to call on a single-row DataFrame (app inference) or a full sample.
    """
    df = df.copy()

    df["amount_clean"] = _clean_money(df["amount"])
    df["amount_log"] = np.sign(df["amount_clean"]) * np.log1p(
        np.abs(df["amount_clean"])
    )

    dt = pd.to_datetime(df["date"])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["dayofweek"] = df["day_of_week"]
    df["month"] = dt.dt.month
    df["is_night"] = df["hour"].between(0, 5).astype(int)

    df["errors"] = df["errors"].where(df["errors"].notna() & (df["errors"] != ""), None)
    df["had_error"] = df["errors"].notna().astype(int)
    df["errors"] = df["errors"].fillna("No Error")

    for col, default in [
        ("merchant_city", "Unknown"),
        ("merchant_state", "Unknown"),
        ("use_chip", "Unknown"),
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(default)

    for col in ["credit_limit", "total_debt", "yearly_income"]:
        if col in df.columns:
            df[col] = _clean_money(df[col])

    return df


def build_mcc_risk_map(df: pd.DataFrame, q: int = 3) -> dict:
    """mcc -> {'low_risk','medium_risk','high_risk'}, from fraud rate by mcc.

    NOTE: computed across the whole sample (train+test), mirroring the
    logistic-regression notebook, which explicitly flagged this as a mild
    leakage shortcut rather than a strictly train-only statistic.
    """
    rates = df.groupby("mcc")["fraud"].mean()
    # Many mcc codes tie at a 0% fraud rate in a stratified sample this size,
    # which would give qcut duplicate bin edges. Rank first (ties broken
    # arbitrarily but consistently) so qcut always yields exactly `q` bins.
    ranks = rates.rank(method="first")
    tier_names = ["low_risk", "medium_risk", "high_risk"][:q]
    tier_codes = pd.qcut(ranks, q=q, labels=False)
    tiers = tier_codes.map(lambda i: tier_names[int(i)])
    return {int(k): str(v) for k, v in tiers.to_dict().items()}


def apply_mcc_risk_tier(df: pd.DataFrame, mapping: dict, default: str = "medium_risk") -> pd.DataFrame:
    df = df.copy()
    df["mcc_risk_tier"] = df["mcc"].map(mapping).fillna(default).astype(str)
    return df


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

LOGREG_NUMERIC = [
    "amount_log", "hour", "day_of_week", "month", "had_error",
    "credit_limit", "num_cards_issued", "credit_score", "total_debt",
    "num_credit_cards", "yearly_income",
]
LOGREG_CATEGORICAL = ["use_chip", "card_brand", "card_type", "has_chip", "mcc_risk_tier"]


def build_logreg_pipeline() -> ImbPipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), LOGREG_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), LOGREG_CATEGORICAL),
        ]
    )
    return ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

RF_NUMERIC = ["amount_clean", "mcc", "hour", "day_of_week", "month", "is_night"]
RF_CATEGORICAL = ["use_chip", "merchant_city", "merchant_state", "errors"]


def build_rf_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", RF_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), RF_CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------

SVM_NUMERIC = ["amount_clean", "hour", "dayofweek", "mcc"]
SVM_CATEGORICAL = ["use_chip"]


def build_svm_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), SVM_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), SVM_CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                CalibratedClassifierCV(
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                    ensemble=False,
                ),
            ),
        ]
    )


def undersample_3to1(df: pd.DataFrame, target_col: str = "fraud", random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Mirrors the SVM notebook: keep all fraud rows, randomly sample
    legitimate rows down to 3x the fraud count."""
    fraud = df[df[target_col] == 1]
    legit = df[df[target_col] == 0]
    n_legit = min(len(legit), 3 * len(fraud))
    legit_sampled = legit.sample(n=n_legit, random_state=random_state)
    out = pd.concat([fraud, legit_sampled]).sample(frac=1, random_state=random_state)
    return out.reset_index(drop=True)


MODEL_KEYS = ["logistic_regression", "random_forest", "svm"]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm": "SVM",
}
