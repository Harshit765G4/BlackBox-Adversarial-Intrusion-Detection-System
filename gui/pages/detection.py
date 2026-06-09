"""
gui/pages/detection.py

Traffic Detection page.
FIX: Uses centralized preprocessor → no more column-mismatch ValueError.
Shows per-model probabilities and ensemble confidence.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from gui.components.predictor import predict_ensemble


def show_detection():
    st.title("🔍 Traffic Detection")
    st.markdown(
        "Upload a network traffic CSV and run ensemble detection "
        "across Random Forest, LightGBM, and the FNN."
    )

    uploaded = st.file_uploader(
        "Upload traffic CSV (any column order, Label optional)",
        type=["csv"]
    )

    if uploaded is None:
        st.info("No file uploaded yet.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    st.write(f"**{len(df):,} rows** loaded.")

    with st.expander("Preview first 5 rows"):
        st.dataframe(df.head())

    if st.button("Run Ensemble Detection", type="primary"):
        with st.spinner("Running models..."):
            try:
                result = predict_ensemble(df)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

        preds = result["predictions"]
        probs = result["probabilities"]

        attack_count = int(preds.sum())
        benign_count = int(len(preds) - attack_count)

        # ── Summary metrics ────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric("⚠️ Attacks",  f"{attack_count:,}")
        col2.metric("✅ Benign",   f"{benign_count:,}")
        col3.metric("Avg Threat Score", f"{probs.mean():.3f}")

        st.divider()

        # ── Per-model agreement ────────────────────────────────────────────
        st.subheader("Per-Model Agreement")
        model_preds = pd.DataFrame({
            "Random Forest": (result["rf_probs"]   >= 0.5).astype(int),
            "LightGBM":      (result["lgbm_probs"] >= 0.5).astype(int),
            "FNN":           (result["fnn_probs"]  >= 0.5).astype(int),
            "Ensemble":      preds,
        })
        agree_pct = (model_preds.sum(axis=0) / len(df) * 100).rename("Attack %")
        st.bar_chart(agree_pct)

        # ── Probability distribution ───────────────────────────────────────
        st.subheader("Ensemble Threat Probability Distribution")
        fig = px.histogram(
            x=probs, nbins=50, labels={"x": "Threat Probability"},
            color_discrete_sequence=["#e74c3c"],
        )
        fig.add_vline(x=0.5, line_dash="dash", annotation_text="Threshold")
        st.plotly_chart(fig, use_container_width=True)

        # ── Detailed table (top-10 highest-risk) ──────────────────────────
        st.subheader("Top 10 Highest-Risk Samples")
        top10_idx = np.argsort(probs)[-10:][::-1]
        top10 = pd.DataFrame({
            "Row":           top10_idx,
            "Prediction":    ["Attack" if p else "Benign" for p in preds[top10_idx]],
            "Ensemble Prob": probs[top10_idx].round(4),
            "RF Prob":       result["rf_probs"][top10_idx].round(4),
            "LGBM Prob":     result["lgbm_probs"][top10_idx].round(4),
            "FNN Prob":      result["fnn_probs"][top10_idx].round(4),
        })
        st.dataframe(top10, use_container_width=True)
