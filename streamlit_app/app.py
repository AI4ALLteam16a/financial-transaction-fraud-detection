"""
Fraud Detection Model Comparison — Streamlit app.

Loads the three trained pipelines (models/*.joblib) and lets you:
  * pick a transaction from a sample of the dataset, or
  * upload a CSV of transactions,
and see all three models' fraud predictions side by side, plus a metrics
comparison panel.

Run `python train_models.py` once first to generate models/*.joblib.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import features  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
SAMPLES_DIR = APP_DIR / "data_samples"

st.set_page_config(page_title="Fraud Detection — Model Comparison", layout="wide")


# ---------------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    missing = [
        p.name
        for p in [
            MODELS_DIR / "logistic_regression.joblib",
            MODELS_DIR / "random_forest.joblib",
            MODELS_DIR / "svm.joblib",
            MODELS_DIR / "mcc_risk_map.json",
            MODELS_DIR / "metrics.json",
        ]
        if not p.exists()
    ]
    if missing:
        return None

    pipelines = {
        "logistic_regression": joblib.load(MODELS_DIR / "logistic_regression.joblib"),
        "random_forest": joblib.load(MODELS_DIR / "random_forest.joblib"),
        "svm": joblib.load(MODELS_DIR / "svm.joblib"),
    }
    with open(MODELS_DIR / "mcc_risk_map.json") as f:
        mcc_map = {int(k): v for k, v in json.load(f).items()}
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return pipelines, mcc_map, metrics


@st.cache_data
def load_sample_transactions() -> pd.DataFrame:
    path = SAMPLES_DIR / "sample_transactions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_upload_template() -> pd.DataFrame:
    path = SAMPLES_DIR / "upload_template.csv"
    if not path.exists():
        return pd.DataFrame(columns=features.RAW_INPUT_COLUMNS)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_all(raw_df: pd.DataFrame, pipelines: dict, mcc_map: dict) -> pd.DataFrame:
    """raw_df: rows with (a subset of) features.RAW_INPUT_COLUMNS.
    Returns a DataFrame of fraud probabilities, one column per model, plus
    a per-model error column when a row is missing fields that model needs.
    """
    missing = [c for c in features.RAW_INPUT_COLUMNS if c not in raw_df.columns]
    work = raw_df.copy()
    for c in missing:
        work[c] = pd.NA

    derived = features.derive_common_fields(work)
    derived = features.apply_mcc_risk_tier(derived, mcc_map)

    results = pd.DataFrame(index=derived.index)

    model_cols = {
        "logistic_regression": features.LOGREG_NUMERIC + features.LOGREG_CATEGORICAL,
        "random_forest": features.RF_NUMERIC + features.RF_CATEGORICAL,
        "svm": features.SVM_NUMERIC + features.SVM_CATEGORICAL,
    }
    for key, cols in model_cols.items():
        row_has_data = derived[cols].notna().all(axis=1)
        proba = pd.Series(float("nan"), index=derived.index)
        if row_has_data.any():
            X = derived.loc[row_has_data, cols]
            proba.loc[row_has_data] = pipelines[key].predict_proba(X)[:, 1]
        results[key] = proba

    return results


def threshold_for(key: str, metrics: dict) -> float:
    return metrics.get(key, {}).get("recommended_threshold", 0.5)


# Fixed categorical order/colors, one hue per model — same order everywhere
# so a model's color always means the same thing across charts.
MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",  # blue
    "Random Forest": "#eb6834",        # orange
    "SVM": "#1baf7a",                  # aqua
}


def horizontal_bar_chart(metrics_df: pd.DataFrame, value_col: str, value_format: str = ".2f"):
    """A horizontal bar chart (model names read left-to-right, never rotated),
    one bar per model, value labeled at the bar's tip."""
    model_order = list(metrics_df.index)
    chart_df = metrics_df[[value_col]].reset_index().rename(columns={value_col: "value"})

    y_axis = alt.Y(
        "Model:N",
        sort=model_order,
        title=None,
        axis=alt.Axis(domain=False, ticks=False, labelColor="#52514e", labelFontSize=13),
    )
    x_scale = alt.Scale(domain=[0, max(1.0, float(chart_df["value"].max()) * 1.15)])
    x_axis = alt.X(
        "value:Q",
        title=None,
        scale=x_scale,
        axis=alt.Axis(gridColor="#e1e0d9", domainColor="#c3c2b7", labelColor="#898781"),
    )
    color = alt.Color(
        "Model:N",
        sort=model_order,
        scale=alt.Scale(domain=list(MODEL_COLORS.keys()), range=list(MODEL_COLORS.values())),
        legend=None,
    )

    bars = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=4, size=20).encode(
        y=y_axis, x=x_axis, color=color,
        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("value:Q", format=value_format)],
    )
    labels = alt.Chart(chart_df).mark_text(align="left", dx=6, color="#0b0b0b").encode(
        y=y_axis, x=x_axis, text=alt.Text("value:Q", format=value_format),
    )
    st.altair_chart((bars + labels).properties(height=34 * len(model_order) + 10), width="stretch")


def render_prediction_panel(raw_row: pd.DataFrame, pipelines, mcc_map, metrics):
    proba = predict_all(raw_row, pipelines, mcc_map).iloc[0]
    cols = st.columns(3)
    for col, key in zip(cols, features.MODEL_KEYS):
        with col:
            label = features.MODEL_LABELS[key]
            p = proba[key]
            if pd.isna(p):
                st.metric(label, "N/A", help="Row is missing fields this model needs.")
                continue
            t = threshold_for(key, metrics)
            is_fraud = p >= t
            st.metric(
                label,
                f"{p:.1%} fraud risk",
                delta="⚠️ Flagged as fraud" if is_fraud else "Looks legitimate",
                delta_color="inverse" if is_fraud else "normal",
            )
            st.caption(f"decision threshold: {t:.2f}")
            st.progress(min(max(p, 0.0), 1.0))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("💳 Fraud Detection — Model Comparison")
st.caption(
    "Compare Logistic Regression, Random Forest, and SVM predictions on financial "
    "transactions, side by side."
)

artifacts = load_artifacts()
if artifacts is None:
    st.error(
        "Trained models not found. Run `python train_models.py` from the "
        "`streamlit_app/` folder first (this fits and exports the three "
        "models to `models/*.joblib`)."
    )
    st.stop()

pipelines, mcc_map, metrics = artifacts

tab_browse, tab_upload, tab_metrics = st.tabs(
    ["🔍 Browse a transaction", "📤 Batch CSV upload", "📊 Model comparison"]
)

# --- Tab 1: browse ----------------------------------------------------------
with tab_browse:
    sample_df = load_sample_transactions()
    if sample_df.empty:
        st.warning("No sample transactions found. Run `python train_models.py` first.")
    else:
        st.write(
            f"Sampled {len(sample_df):,} transactions from the dataset "
            f"({int(sample_df['fraud'].sum())} labeled fraud)."
        )
        filter_choice = st.radio(
            "Filter", ["All", "Known fraud only", "Known legitimate only"], horizontal=True
        )
        view = sample_df
        if filter_choice == "Known fraud only":
            view = sample_df[sample_df["fraud"] == 1]
        elif filter_choice == "Known legitimate only":
            view = sample_df[sample_df["fraud"] == 0]

        options = view["id"].astype(str).tolist()
        if not options:
            st.info("No transactions match that filter.")
        else:
            chosen_id = st.selectbox("Transaction ID", options)
            row = view[view["id"].astype(str) == chosen_id]
            st.dataframe(row, width='stretch', hide_index=True)

            actual = row["fraud"].iloc[0]
            st.caption(
                f"Ground-truth label for this transaction: "
                f"{'🚩 Fraud' if actual == 1 else '✅ Legitimate'}"
            )

            st.subheader("Predictions")
            render_prediction_panel(row, pipelines, mcc_map, metrics)

# --- Tab 2: batch upload -----------------------------------------------------
with tab_upload:
    st.write(
        "Upload a CSV with (a subset of) these columns: "
        f"`{', '.join(features.RAW_INPUT_COLUMNS)}`. "
        "A model's prediction shows as N/A for any row missing fields it needs."
    )
    template = load_upload_template()
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False),
        file_name="upload_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that CSV: {e}")
        else:
            with st.spinner(f"Scoring {len(batch_df):,} rows..."):
                preds = predict_all(batch_df, pipelines, mcc_map)
            out = pd.concat([batch_df.reset_index(drop=True), preds.add_suffix("_fraud_probability")], axis=1)
            for key in features.MODEL_KEYS:
                t = threshold_for(key, metrics)
                out[f"{key}_flagged"] = out[f"{key}_fraud_probability"] >= t

            st.success(f"Scored {len(out):,} rows.")
            st.dataframe(out, width='stretch')
            st.download_button(
                "Download results CSV",
                out.to_csv(index=False),
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

# --- Tab 3: metrics ----------------------------------------------------------
with tab_metrics:
    st.write(
        "Metrics from each model's held-out test split (20% of its training sample). "
        "See `streamlit_app/README.md` for how this training run compares to the "
        "original notebooks."
    )
    rows = []
    for key in features.MODEL_KEYS:
        m = metrics.get(key, {})
        tuned = m.get("at_tuned_threshold", {})
        default = m.get("at_0.5", {})
        rows.append(
            {
                "Model": features.MODEL_LABELS[key],
                "ROC-AUC": m.get("roc_auc"),
                "Precision (tuned)": tuned.get("precision"),
                "Recall (tuned)": tuned.get("recall"),
                "F1 (tuned)": tuned.get("f1"),
                "Threshold used": tuned.get("threshold"),
                "Precision (0.5)": default.get("precision"),
                "Recall (0.5)": default.get("recall"),
                "F1 (0.5)": default.get("f1"),
                "Test rows": m.get("n_test_rows"),
                "Test fraud rows": m.get("n_test_fraud"),
            }
        )
    metrics_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        metrics_df.style.format(
            {c: "{:.3f}" for c in metrics_df.columns if metrics_df[c].dtype != "int64"}
        ),
        width='stretch',
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**ROC-AUC**")
        horizontal_bar_chart(metrics_df, "ROC-AUC")
    with chart_col2:
        st.markdown("**F1 (tuned threshold)**")
        horizontal_bar_chart(metrics_df, "F1 (tuned)")
