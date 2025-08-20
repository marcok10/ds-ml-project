# Home.py — drop-in UI refresh
# To run locally: streamlit run Home.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, date, time
from pathlib import Path

from feature_builder import build_features_minimal

st.set_page_config(page_title="Flight Delay Predictions", page_icon="✈️", layout="wide")

# ---------- helpers ----------
def load_airports() -> pd.DataFrame:
    """
    Return a dataframe with at least columns: ['iata','name'].
    Tries data/airports.csv first; otherwise uses a built-in fallback list.
    """
    csv_path = Path("data/airports.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # normalize columns
        cols = {c.lower(): c for c in df.columns}
        # we accept iata / IATA and name / airport / airport_name
        iata_col = next((cols[k] for k in cols if k in ("iata", "code", "iata_code")), None)
        name_col = next((cols[k] for k in cols if k in ("name", "airport", "airport_name", "airport fullname")), None)
        if iata_col is None or name_col is None:
            raise ValueError("data/airports.csv must have columns for IATA code and airport name.")
        out = df[[iata_col, name_col]].rename(columns={iata_col: "iata", name_col: "name"})
        out["iata"] = out["iata"].astype(str).str.upper().str.strip()
        out["name"] = out["name"].astype(str).str.strip()
        out = out[out["iata"].str.len() == 3].drop_duplicates(subset=["iata"]).sort_values("iata")
        return out.reset_index(drop=True)

    # Fallback (keeps the app usable without the CSV)
    fallback = [
        ("TUN", "Tunis Carthage"),
        ("CDG", "Paris Charles de Gaulle"),
        ("ORY", "Paris Orly"),
        ("LHR", "London Heathrow"),
        ("LGW", "London Gatwick"),
        ("JFK", "New York JFK"),
        ("EWR", "Newark Liberty"),
        ("LGA", "New York LaGuardia"),
        ("SFO", "San Francisco"),
        ("LAX", "Los Angeles"),
        ("HND", "Tokyo Haneda"),
        ("NRT", "Tokyo Narita"),
        ("DXB", "Dubai"),
        ("SIN", "Singapore Changi"),
        ("AMS", "Amsterdam Schiphol"),
        ("FRA", "Frankfurt"),
        ("MAD", "Madrid Barajas"),
        ("BCN", "Barcelona"),
        ("FCO", "Rome Fiumicino"),
        ("IST", "Istanbul"),
        ("DOH", "Doha Hamad"),
        ("CAI", "Cairo"),
        ("CMN", "Casablanca"),
        ("ZRH", "Zurich"),
        ("MUC", "Munich"),
    ]
    return pd.DataFrame(fallback, columns=["iata", "name"])

def options_from_airports(df: pd.DataFrame):
    # show as "IATA — Name" but keep a mapping back to code
    labels = [f"{row.iata} — {row.name}" for row in df.itertuples(index=False)]
    code_by_label = {label: code for label, code in zip(labels, df["iata"])}
    return labels, code_by_label

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

# ---------- UI ----------
st.title("✈️ Flight Delay Predictions")
st.caption("Enter flight details. The app builds features (route, time parts, ETA, etc.) and uses your trained stack.")

air_df = load_airports()
labels, code_map = options_from_airports(air_df)

with st.form("predict_form", clear_on_submit=False):
    # Two columns for a cleaner layout
    c1, c2 = st.columns(2)

    with c1:
        dep_label = st.selectbox("Departure Airport (IATA)", labels, index=labels.index("TUN — Tunis Carthage") if "TUN — Tunis Carthage" in labels else 0)
    with c2:
        arr_label = st.selectbox("Arrival Airport (IATA)", labels, index=labels.index("CDG — Paris Charles de Gaulle") if "CDG — Paris Charles de Gaulle" in labels else min(1, len(labels)-1))

    c3, c4 = st.columns(2)
    with c3:
        dep_date: date = st.date_input("Departure Date", value=date(2017, 1, 16))
    with c4:
        # Native time picker; shows OS/Browser time UI (HH:MM)
        dep_time: time = st.time_input("Departure Time", value=time(8, 5), step=60)

    submitted = st.form_submit_button("Predict")

# ---------- Prediction ----------
if submitted:
    dep_code = code_map[dep_label]
    arr_code = code_map[arr_label]
    dep_dt = datetime.combine(dep_date, dep_time)

    raw = pd.DataFrame([{
        "departure_point": dep_code,
        "arrival_point":   arr_code,
        "departure_time":  dep_dt,
        # no aircraft_model field; your feature_builder fills it from category_modes.json
    }])

    model = load_model()
    X = build_features_minimal(raw)
    y = model.predict(X)[0]

    st.metric("Predicted delay (minutes)", f"{float(y):.1f}")
