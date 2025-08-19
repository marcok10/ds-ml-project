# To run: streamlit run Home.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, date, time as dtime
from feature_builder import build_features_minimal

st.set_page_config(page_title="Flight Delay Predictions", page_icon="✈️", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

@st.cache_data
def load_airports_from_artifacts(path: str = "artifacts/route_expected_duration.csv"):
    """Read the route table artifact and derive a sorted unique list of IATA codes."""
    try:
        df = pd.read_csv(path)
        # route format assumed as "AAA-BBB"
        iatas = set()
        for r in df["route"].astype(str):
            parts = r.split("-")
            for p in parts:
                p = p.strip().upper()
                if 2 < len(p) <= 4:  # basic sanity
                    iatas.add(p)
        return sorted(iatas)
    except Exception:
        # Fallback minimal options if artifact missing
        return ["TUN", "CDG", "JFK", "LHR", "DXB", "AMS", "FRA", "IST", "MAD", "SFO", "LAX"]

model = load_model()
airports = load_airports_from_artifacts()

# pick sensible defaults if available
def_idx_dep = airports.index("TUN") if "TUN" in airports else 0
def_idx_arr = airports.index("CDG") if "CDG" in airports else min(1, len(airports)-1)

st.title("✈️ Flight Delay Predictions")
st.caption("Enter flight details. The app builds features (route, time parts, ETA, etc.) and uses your trained stack.")

left, right = st.columns([1, 2])

with left:
    dep = st.selectbox("Departure Airport (IATA)", airports, index=def_idx_dep)
    arr = st.selectbox("Arrival Airport (IATA)", airports, index=def_idx_arr)

    dep_date: date = st.date_input("Departure Date", value=date(2017, 1, 16))
    dep_time: dtime = st.time_input("Departure Time", value=dtime(8, 5))

    go = st.button("Predict")

with right:
    if go:
        # combine date + time into a single timestamp
        dep_dt = datetime.combine(dep_date, dep_time)

        # Only pass the columns your feature_builder needs. It will fill the rest from artifacts.
        raw = pd.DataFrame([{
            "departure_point": dep,
            "arrival_point": arr,
            "departure_time": dep_dt
        }])

        X = build_features_minimal(raw)
        y = model.predict(X)[0]
        st.metric("Predicted delay (minutes)", f"{float(y):.1f}")
