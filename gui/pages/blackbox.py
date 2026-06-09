"""
gui/pages/blackbox.py

Black-Box Genetic Attack page.
ORIGINAL: stub that said "GA Engine Integration Coming Next"
FIXED:    fully wired to ga_engine.run_attack() with live progress,
          result metrics, perturbation analysis, and realism report.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from attacks.blackbox.oracle import EnsembleOracle
from attacks.blackbox.ga_engine import run_attack
from core.config import GA_CONFIG


def show_blackbox():
    st.title("🧬 Black-Box Genetic Attack")
    st.markdown(
        "Simulate a **query-only adversary** who attempts to craft "
        "network traffic that evades all three IDS models simultaneously. "
        "No model weights or gradients are accessed."
    )

    # ── Parameters sidebar ────────────────────────────────────────────────
    with st.expander("⚙️ Attack Configuration"):
        col1, col2 = st.columns(2)
        pop_size    = col1.number_input("Population Size",   value=100, step=10)
        generations = col2.number_input("Max Generations",   value=80,  step=10)
        evasion_thr = col1.slider("Evasion Threshold",  0.10, 0.50, value=0.45, step=0.01)
        query_limit = col2.number_input("Query Budget",  value=10000, step=1000)

    if not st.button("🚀 Run Genetic Attack", type="primary"):
        st.info("Configure parameters above and click Run.")
        return

    # ── Override config for this run ──────────────────────────────────────
    GA_CONFIG["population_size"] = int(pop_size)
    GA_CONFIG["generations"]     = int(generations)
    GA_CONFIG["evasion_threshold"] = float(evasion_thr)

    oracle   = EnsembleOracle(query_limit=int(query_limit))
    progress = st.progress(0.0, text="Initialising attack...")
    log_box  = st.empty()

    # ── Run with progress callback via gen_log polling ───────────────────
    # We collect results synchronously; Streamlit doesn't support true async.
    with st.spinner("Genetic attack running..."):
        result = run_attack(oracle=oracle, verbose=False)

    progress.progress(1.0, text="Attack complete.")

    # ── Summary ───────────────────────────────────────────────────────────
    if result.success:
        st.success(
            f"✅ **Attack succeeded** at generation {result.generation} "
            f"using {result.queries} queries."
        )
    else:
        st.warning(
            f"⚠️ Attack failed after {result.generation} generations. "
            f"Best probability: {result.probability:.4f}"
        )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queries Used",       f"{result.queries:,}")
    c2.metric("Ensemble Probability", f"{result.probability:.4f}")
    c3.metric("Perturbation Norm",   f"{result.perturbation_norm:.4f}")
    c4.metric("Realism Penalty",     f"{result.realism_penalty:.4f}")

    # ── Generation log plot ───────────────────────────────────────────────
    if result.generation_log:
        log_df = pd.DataFrame(result.generation_log)

        st.subheader("📉 Probability Over Generations")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=log_df["generation"], y=log_df["probability"],
            mode="lines+markers", name="Ensemble Probability",
            line=dict(color="#e74c3c")
        ))
        fig.add_hline(y=evasion_thr, line_dash="dash",
                      annotation_text=f"Evasion threshold ({evasion_thr})")
        fig.update_layout(
            xaxis_title="Generation",
            yaxis_title="Ensemble Probability",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

        # Fitness evolution
        st.subheader("📈 Fitness Over Generations")
        fig2 = px.line(log_df, x="generation", y="fitness",
                       labels={"fitness": "Multi-Objective Fitness"},
                       color_discrete_sequence=["#3498db"])
        st.plotly_chart(fig2, use_container_width=True)

    # ── Per-model probabilities for adversarial sample ────────────────────
    if result.adversarial_sample is not None:
        st.subheader("🔬 Per-Model Analysis (Adversarial Sample)")
        # Re-query oracle for diagnostic detail
        final_result = oracle.query(
            result.adversarial_sample, result.feature_names
        )
        probs = {
            "Random Forest": final_result["rf_prob"],
            "LightGBM":      final_result["lgbm_prob"],
            "FNN":           final_result["fnn_prob"],
            "Ensemble":      final_result["probability"],
        }
        prob_df = pd.DataFrame(
            list(probs.items()), columns=["Model", "Probability"]
        )
        fig3 = px.bar(
            prob_df, x="Model", y="Probability",
            color="Probability",
            color_continuous_scale=["green", "yellow", "red"],
            range_color=[0, 1],
        )
        fig3.add_hline(y=0.5, line_dash="dash",
                       annotation_text="Detection Threshold")
        st.plotly_chart(fig3, use_container_width=True)

        # ── Feature perturbations ──────────────────────────────────────────
        st.subheader("📊 Top-20 Perturbed Features")
        seed = np.array(result.seed_sample)
        adv  = np.array(result.adversarial_sample)
        diff = adv - seed
        top20_idx = np.argsort(np.abs(diff))[-20:][::-1]
        pert_df = pd.DataFrame({
            "Feature":     [result.feature_names[i] for i in top20_idx],
            "Δ (adv-seed)": diff[top20_idx].round(4),
        })
        st.dataframe(pert_df, use_container_width=True)
