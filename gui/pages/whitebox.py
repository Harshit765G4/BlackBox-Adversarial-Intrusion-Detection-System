import streamlit as st

def show_whitebox():

    st.title(
        "White-Box Attacks"
    )

    st.subheader(
        "FGSM Attack"
    )

    if st.button(
        "Run FGSM"
    ):
        st.success(
            "FGSM Execution Connected"
        )

    st.subheader(
        "PGD Attack"
    )

    if st.button(
        "Run PGD"
    ):
        st.success(
            "PGD Execution Connected"
        )