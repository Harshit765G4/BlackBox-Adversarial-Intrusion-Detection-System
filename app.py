import streamlit as st

from gui.pages.dashboard import show_dashboard
from gui.pages.detection import show_detection
from gui.pages.whitebox import show_whitebox
from gui.pages.blackbox import show_blackbox
from gui.pages.analytics import show_analytics

st.set_page_config(
    page_title="EnCryptoGuard",
    page_icon="🛡️",
    layout="wide"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Traffic Detection",
        "White-Box Attacks",
        "Black-Box Attacks",
        "Analytics"
    ]
)

if page == "Dashboard":
    show_dashboard()

elif page == "Traffic Detection":
    show_detection()

elif page == "White-Box Attacks":
    show_whitebox()

elif page == "Black-Box Attacks":
    show_blackbox()

elif page == "Analytics":
    show_analytics()