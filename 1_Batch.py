# pages/1_📄_Batch.py
import streamlit as st, pandas as pd, joblib
from feature_builder import build_features_minimal

st.title("📄 Batch Predictions")
st.caption("Upload CSV with columns at least: departure_point, arrival_point, departure_time, aircraft_model")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")
model = load_model()

file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    df_in = pd.read_csv(file)
    if "departure_time" in df_in.columns:
        df_in["departure_time"] = pd.to_datetime(df_in["departure_time"], errors="coerce")
    X = build_features_minimal(df_in)
    preds = model.predict(X)
    df_out = df_in.copy()
    df_out["pred_delay_min"] = preds
    st.dataframe(df_out.head(50), use_container_width=True)
    st.download_button("Download predictions", df_out.to_csv(index=False), "predictions.csv", "text/csv")
