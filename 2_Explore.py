# pages/2_📊_Explore.py
import streamlit as st, pandas as pd
st.title("📊 Explore")
st.caption("Bring a recent batch (with predictions) to visualize patterns.")
f = st.file_uploader("Upload predictions CSV", type=["csv"])
if f:
    df = pd.read_csv(f)
    st.bar_chart(df.groupby("route")["pred_delay_min"].mean().sort_values().tail(10))
