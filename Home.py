# Home.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from feature_builder import build_features_minimal
from hype_pipeline import HypeStacker

st.set_page_config(page_title="Flight Delay Dashboard", page_icon="✈️", layout="wide")

@st.cache_resource
def load_model():
    model: HypeStacker = joblib.load("model.joblib")
    # ensure everything is in memory (small top-level object, big artifacts on disk)
    model.load_artifacts("artifacts")
    return model

model = load_model()

st.title("✈️ Flight Delay – Decision Assist")
st.caption("Enter flight details. The app builds features (route, time parts, ETA, etc.) and uses your trained stack.")

left, right = st.columns([1, 2])
with left:
    dep = st.text_input("Departure IATA", "TUN").upper().strip()
    arr = st.text_input("Arrival IATA", "CDG").upper().strip()
    dep_time = st.date_input("Departure time", value=datetime(2017, 1, 16, 8, 5))
    aircraft_model = st.text_input("Aircraft model", "Airbus A320").strip()
    go = st.button("Predict")

with right:
    if go:
        raw = pd.DataFrame([{
            "departure_point": dep,
            "arrival_point": arr,
            "departure_time": dep_time,
            "aircraft_model": aircraft_model
        }])
        X = build_features_minimal(raw)
        y = model.predict(X)[0]
        st.metric("Predicted delay (minutes)", f"{float(y):.1f}")
