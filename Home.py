# Home.py — Flight Delay Predictions (airports from route_expected_duration.csv)
# Run: streamlit run Home.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import joblib
from datetime import date, time, datetime
from pathlib import Path

from feature_builder import build_features_minimal

st.set_page_config(page_title="Flight Delay Predictions", page_icon="✈️", layout="wide")

# ---------- caches ----------
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

@st.cache_data
def load_route_endpoints() -> tuple[list[str], list[str]]:
    """
    Read artifacts/route_expected_duration.csv and extract unique departure (left)
    and arrival (right) IATA codes.

    Supports routes formatted like:
      - "AAA-BBB"
      - "AAA → BBB"
      - separate columns (e.g., departure_point / arrival_point, origin / destination, etc.)
    """
    p = Path("artifacts/route_expected_duration.csv")
    if not p.exists():
        st.error("Missing artifacts/route_expected_duration.csv — cannot determine airports.")
        st.stop()

    df = pd.read_csv(p)

    # Try to find dep/arr directly
    dep_candidates = [c for c in df.columns if c.lower() in
                      {"departure_point", "departure", "origin", "from", "from_iata", "dep"}]
    arr_candidates = [c for c in df.columns if c.lower() in
                      {"arrival_point", "arrival", "destination", "to", "to_iata", "arr"}]

    dep_list: list[str] = []
    arr_list: list[str] = []

    if dep_candidates and arr_candidates:
        dep_col = dep_candidates[0]
        arr_col = arr_candidates[0]
        dep_list = (
            df[dep_col].astype(str).str.upper().str.strip()
            .loc[lambda s: s.str.len() == 3]
            .dropna().unique().tolist()
        )
        arr_list = (
            df[arr_col].astype(str).str.upper().str.strip()
            .loc[lambda s: s.str.len() == 3]
            .dropna().unique().tolist()
        )
    elif "route" in df.columns:
        # Parse "AAA-BBB" or "AAA → BBB"
        routes = df["route"].astype(str)
        for r in routes:
            r_clean = r.strip()
            if "→" in r_clean:
                a, b = r_clean.split("→", 1)
            elif "-" in r_clean:
                a, b = r_clean.split("-", 1)
            elif "—" in r_clean:  # em dash fallback
                a, b = r_clean.split("—", 1)
            elif "->" in r_clean:
                a, b = r_clean.split("->", 1)
            else:
                continue
            a = a.strip().upper()
            b = b.strip().upper()
            if len(a) == 3:
                dep_list.append(a)
            if len(b) == 3:
                arr_list.append(b)
    else:
        st.error(
            "route_expected_duration.csv must have either a 'route' column (e.g. 'AAA-BBB' or 'AAA → BBB') "
            "or explicit departure/arrival columns."
        )
        st.stop()

    dep_codes = sorted(set(dep_list))
    arr_codes = sorted(set(arr_list))

    if not dep_codes or not arr_codes:
        st.error("No valid IATA codes found in artifacts/route_expected_duration.csv.")
        st.stop()

    return dep_codes, arr_codes

# ---------- data ----------
model = load_model()
dep_codes, arr_codes = load_route_endpoints()

# ---------- UI ----------
st.title("✈️ Flight Delay Predictions")
st.caption("Pick a route and departure. The model predicts delay based on your trained stack.")

with st.form("predict_form", clear_on_submit=False):
    c1, c2 = st.columns(2)
    dep = c1.selectbox("Departure Airport (IATA)", dep_codes, index=0, key="dep_code")
    # pick first arrival that's different to avoid same-same default
    default_arr_idx = 0
    if dep in arr_codes and len(arr_codes) > 1:
        default_arr_idx = 1 if arr_codes[0] == dep else 0
    arr = c2.selectbox("Arrival Airport (IATA)", arr_codes, index=default_arr_idx, key="arr_code")

    c3, c4 = st.columns(2)
    dep_date = c3.date_input("Departure Date", value=date(2017, 1, 16))
    # Native time picker; step=60 shows minutes in 1-min increments but uses the OS/browser time control (not a giant list)
    dep_time = c4.time_input("Departure Time", value=time(8, 5), step=60)

    submitted = st.form_submit_button("Predict")

# ---------- prediction ----------
if submitted:
    if dep == arr:
        st.error("Departure and arrival airports must be different.")
    else:
        dep_dt = datetime.combine(dep_date, dep_time)
        raw = pd.DataFrame([{
            "departure_point": dep,
            "arrival_point":   arr,
            "departure_time":  dep_dt,
        }])
        X = build_features_minimal(raw)
        y = float(model.predict(X)[0])
        st.metric("Predicted delay (minutes)", f"{y:.1f}")
