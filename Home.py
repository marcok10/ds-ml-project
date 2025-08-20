# Home.py — Flight Delay Predictions
# To run: streamlit run Home.py
import streamlit as st
import pandas as pd
import joblib
from datetime import date, time, datetime
from feature_builder import build_features_minimal

st.set_page_config(page_title="Flight Delay Predictions", page_icon="✈️", layout="wide")

# -------- cache helpers --------
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

@st.cache_data
def load_airports():
    # Airports served by YOUR network (created by make_airports_served.py)
    try:
        df = pd.read_csv("artifacts/airports_served.csv")
        opts = sorted({c for c in df["iata"].astype(str) if c and c != "nan"})
        return opts
    except Exception:
        return []

model = load_model()
airport_options = load_airports()

# -------- UI --------
st.title("✈️ Flight Delay Predictions")
st.caption("Choose route and departure; we’ll engineer features and predict delay.")

left, right = st.columns([1, 2])

with left:
    # Airports (dropdowns restricted to your network; fall back to free text if missing)
    if airport_options:
        dep = st.selectbox("Departure Airport (IATA)", airport_options, index=0, key="dep_iata")
        # try to preselect a different default arrival
        default_arr_idx = 1 if len(airport_options) > 1 else 0
        arr = st.selectbox("Arrival Airport (IATA)", airport_options, index=default_arr_idx, key="arr_iata")
    else:
        st.info("`artifacts/airports_served.csv` not found or empty — using free text inputs.")
        dep = st.text_input("Departure Airport (IATA)", "TUN").upper().strip()
        arr = st.text_input("Arrival Airport (IATA)", "CDG").upper().strip()

    # Date
    dep_date = st.date_input("Departure Date", value=date(2017, 1, 16), key="dep_date")

    # Time (hour & minute pickers — no giant minute dropdown)
    c1, c2 = st.columns(2)
    hour   = c1.selectbox("Hour",   list(range(0, 24)),     index=8, format_func=lambda h: f"{h:02d}")
    minute = c2.selectbox("Minute", list(range(0, 60, 5)),  index=1, format_func=lambda m: f"{m:02d}")
    # ↑ change step to 1 if you want 1-min precision: list(range(60))

    go = st.button("Predict")

with right:
    if go:
        if dep == arr:
            st.error("Departure and arrival airports must be different.")
        else:
            dep_dt = datetime.combine(dep_date, time(hour, minute))
            raw = pd.DataFrame([{
                "departure_point": dep,
                "arrival_point":   arr,
                "departure_time":  dep_dt,
                # 'aircraft_model' intentionally omitted; feature_builder fills missing cats via modes.
            }])
            X = build_features_minimal(raw)
            y = float(model.predict(X)[0])
            st.metric("Predicted delay (minutes)", f"{y:.1f}")
