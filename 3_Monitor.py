# pages/3_🛠️_Monitor.py
import streamlit as st, pandas as pd, numpy as np
st.title("🛠️ Monitor")
st.caption("Drop in a file with `actual_delay_min` to compare vs predictions.")
f = st.file_uploader("Upload scored CSV with actuals", type=["csv"])
if f:
    df = pd.read_csv(f)
    if {"pred_delay_min","actual_delay_min"}.issubset(df.columns):
        err = df["pred_delay_min"] - df["actual_delay_min"]
        st.metric("MAE (min)", f"{err.abs().mean():.2f}")
        st.metric("RMSE (min)", f"{np.sqrt((err**2).mean()):.2f}")
        st.line_chart(err)
    else:
        st.warning("Missing required columns.")
