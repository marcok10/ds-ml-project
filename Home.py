# Home.py — To run: streamlit run Home.py
from __future__ import annotations

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
def load_iata_options():
    """
    Prefer a CSV at artifacts/airports_iata.csv with at least column 'iata'
    (optionally: name, city, country). If missing, return a small fallback list.
    """
    try:
        df = pd.read_csv("artifacts/airports_iata.csv")
        # Normalize column names
        cols = {c.lower(): c for c in df.columns}
        code_col = cols.get("iata") or cols.get("code") or cols.get("iata_code")
        if not code_col:
            raise ValueError("CSV needs an 'iata' (or 'code') column.")
        # Build display labels if we have extra info
        name_col = cols.get("name")
        city_col = cols.get("city")
        country_col = cols.get("country")

        rows = []
        for _, r in df.iterrows():
            code = str(r[code_col]).strip().upper()
            if not code or code == "NAN":
                continue
            parts = [code]
            if name_col and pd.notna(r[name_col]):
                parts.append(str(r[name_col]).strip())
            if city_col and pd.notna(r[city_col]):
                parts.append(str(r[city_col]).strip())
            if country_col and pd.notna(r[country_col]):
                parts.append(str(r[country_col]).strip())
            label = " — ".join(parts)
            rows.append({"code": code, "label": label})
        # Unique, sorted by label
        seen, out = set(), []
        for row in sorted(rows, key=lambda x: x["label"]):
            if row["code"] not in seen:
                seen.add(row["code"])
                out.append(row)
        return out
    except Exception:
        # Minimal fallback so UI is usable without the CSV
        fallback = [
            {"code": "TUN", "label": "TUN — Tunis Carthage"},
            {"code": "CDG", "label": "CDG — Paris Charles de Gaulle"},
            {"code": "JFK", "label": "JFK — New York JFK"},
            {"code": "LHR", "label": "LHR — London Heathrow"},
            {"code": "SFO", "label": "SFO — San Francisco"},
            {"code": "DXB", "label": "DXB — Dubai"},
            {"code": "HND", "label": "HND — Tokyo Haneda"},
            {"code": "SIN", "label": "SIN — Singapore Changi"},
        ]
        return fallback

model = load_model()
iata_options = load_iata_options()

st.title("✈️ Flight Delay Predictions")
st.caption("Enter flight details. The app builds features (route, time parts, ETA, etc.) and uses your trained stack.")

# Helper to render a selectbox with pretty labels but return code
def iata_selectbox(label: str, key: str, default_code: str | None = None):
    options = iata_options
    idx = 0
    if default_code:
        for i, opt in enumerate(options):
            if opt["code"] == default_code:
                idx = i
                break
    sel = st.selectbox(
        label,
        options,
        index=idx if options else None,
        format_func=lambda o: o["label"],
        key=key,
        disabled=not bool(options),
        placeholder="Start typing to search…",
    )
    return sel["code"] if sel else None

with st.form("predict_form"):
    dep_code = iata_selectbox("Departure Airport (IATA)", key="dep", default_code="TUN")
    arr_code = iata_selectbox("Arrival Airport (IATA)", key="arr", default_code="CDG")

    col1, col2 = st.columns([1, 1])
    with col1:
        dep_date: date = st.date_input("Departure Date", value=datetime(2017, 1, 16).date())
    with col2:
        dep_time: dtime = st.time_input("Departure Time", value=dtime(8, 5), step=60)

    submitted = st.form_submit_button("Predict")

# Predict
if submitted:
    if not dep_code or not arr_code:
        st.error("Please choose both a departure and an arrival airport.")
    else:
        # Combine date + time for the builder
        dep_dt = datetime.combine(dep_date, dep_time)

        raw = pd.DataFrame([{
            "departure_point": dep_code,
            "arrival_point": arr_code,
            "departure_time": dep_dt,
            # no aircraft_model provided; feature_builder will fill from mode
        }])

        X = build_features_minimal(raw)
        y = model.predict(X)[0]
        st.metric("Predicted delay (minutes)", f"{float(y):.1f}")
