import streamlit as st
import plotly.express as px
import pandas as pd

def show_dashboard():

    st.title("🛡️ EnCryptoGuard")

    st.subheader(
        "Adversarial Intrusion Detection System"
    )

    st.divider()

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Random Forest",
        "99.85%"
    )

    col2.metric(
        "LightGBM",
        "99.91%"
    )

    col3.metric(
        "FNN",
        "98.19%"
    )

    st.divider()

    df = pd.DataFrame({
        "Model":[
            "Random Forest",
            "LightGBM",
            "FNN"
        ],
        "Accuracy":[
            99.85,
            99.91,
            98.19
        ]
    })

    fig = px.bar(
        df,
        x="Model",
        y="Accuracy",
        title="Model Accuracy Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.success(
        "System Ready"
    )