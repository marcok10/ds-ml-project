# Home.py — Flight Delay Predictions with SHAP + History
# Run: streamlit run Home.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, date, time
from pathlib import Path

from feature_builder import build_features_minimal

st.set_page_config(page_title="Flight Delay Predictions", page_icon="✈️", layout="wide")

# ----------------- CACHING HELPERS -----------------

@st.cache_resource
def load_model():
    # Loads your HypeStacker (with preprocessors + base models + meta)
    return joblib.load("model.joblib")

@st.cache_data
def load_routes_table() -> pd.DataFrame:
    # Used to populate airport dropdowns from your actual network
    path = Path("artifacts/route_expected_duration.csv")
    df = pd.read_csv(path)
    # Expect columns: route, expected_duration_min  (route like "AAA-BBB")
    # Derive departure and arrival columns
    dep, arr = [], []
    for r in df["route"].astype(str):
        if "-" in r:
            a, b = r.split("-", 1)
            dep.append(a.strip())
            arr.append(b.strip())
        else:
            dep.append(None); arr.append(None)
    df["dep"] = dep
    df["arr"] = arr
    return df.dropna(subset=["dep", "arr"])

@st.cache_data
def load_training_history() -> pd.DataFrame:
    # Historical training data with 'target' (actual delay in minutes)
    path = Path("data/data_all_col_upto_duration_ratio.csv")
    df = pd.read_csv(path)
    # Ensure route exists (some versions may already have it)
    if "route" not in df.columns and {"departure_point", "arrival_point"}.issubset(df.columns):
        df["route"] = df["departure_point"].astype(str) + "-" + df["arrival_point"].astype(str)
    return df

def _airport_options_from_routes(routes_df: pd.DataFrame):
    # Left (dep) codes: unique departures seen in your schedule
    dep_codes = sorted(routes_df["dep"].dropna().astype(str).str.upper().unique().tolist())
    # Right (arr) codes: unique arrivals seen in your schedule
    arr_codes = sorted(routes_df["arr"].dropna().astype(str).str.upper().unique().tolist())
    return dep_codes, arr_codes

def _native_datetime(dep_date: date, dep_time: time) -> datetime:
    return datetime.combine(dep_date, dep_time)

# ----------------- UI -----------------

st.title("✈️ Flight Delay Predictions")
st.caption("Choose a route and departure time. The model will build features and predict the delay, with an explanation and historical context.")

routes_df = load_routes_table()
dep_codes, arr_codes = _airport_options_from_routes(routes_df)

with st.form("predict_form", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        dep_code = st.selectbox("Departure Airport (IATA)", dep_codes, index=(dep_codes.index("TUN") if "TUN" in dep_codes else 0))
    with c2:
        arr_code = st.selectbox("Arrival Airport (IATA)", arr_codes, index=(arr_codes.index("CDG") if "CDG" in arr_codes else (1 if len(arr_codes) > 1 else 0)))

    c3, c4 = st.columns(2)
    with c3:
        dep_date = st.date_input("Departure Date", value=date(2017, 1, 16))
    with c4:
        # True time picker (HH:MM)
        dep_time_val = st.time_input("Departure Time", value=time(8, 5), step=60)

    submitted = st.form_submit_button("Predict")

# ----------------- PREDICTION -----------------

if submitted:
    dep_dt = _native_datetime(dep_date, dep_time_val)

    raw = pd.DataFrame([{
        "departure_point": dep_code,
        "arrival_point":   arr_code,
        "departure_time":  dep_dt,
        # 'aircraft_model' intentionally omitted; feature_builder will fill from modes artifact
    }])

    model = load_model()
    X = build_features_minimal(raw)
    y_pred = float(model.predict(X)[0])
    st.metric("Predicted delay (minutes)", f"{y_pred:.1f}")

    st.divider()

    # ----------------- VISUALS -----------------

    viz_left, viz_right = st.columns([1,1])

    # ---- (A) SHAP: per-feature contributions for the XGB base model ----
    with viz_left:
        st.subheader("Why this prediction? (SHAP for XGBoost base)")

        # Try importing SHAP only when needed (reduces overhead on first load)
        try:
            import shap  # requires shap + numba in requirements
            # Build the encoded row the same way the base XGB saw it
            X_enc = model._transform_encoded(X)  # scipy CSR
            # Dense for TreeExplainer single-row stability
            X_dense = X_enc.toarray().astype(np.float32)

            # Feature names (scaled numerics + one-hot cats)
            num_names = list(model.cols.num_cols)
            cat_names = []
            if model.encoder_ is not None:
                cat_names = list(model.encoder_.get_feature_names_out(model.cols.cat_cols))
            feat_names = num_names + cat_names

            # Use the base XGB model; prefer booster if available
            booster = getattr(model, "xgb_booster_", None)
            if booster is not None:
                explainer = shap.TreeExplainer(booster)
                shap_vals = explainer.shap_values(X_dense)
                base_value = explainer.expected_value
            else:
                # Fall back to wrapper
                explainer = shap.TreeExplainer(model.xgb_)
                shap_vals = explainer.shap_values(X_dense)
                base_value = explainer.expected_value

            # shap_values shape: (1, n_features)
            vals = np.array(shap_vals).reshape(-1)
            # Build a small dataframe and plot top 10 absolute contributors
            shap_df = (pd.DataFrame({"feature": feat_names, "shap_value": vals})
                         .assign(abs_val=lambda d: d["shap_value"].abs())
                         .sort_values("abs_val", ascending=False)
                         .head(10))

            # Positive → increases delay; Negative → decreases delay
            st.write("Top factors (positive ↑ increases predicted delay; negative ↓ decreases)")
            st.bar_chart(shap_df.set_index("feature")["shap_value"])
            with st.expander("Details"):
                st.dataframe(shap_df[["feature", "shap_value"]], use_container_width=True)

        except Exception as e:
            st.warning("SHAP could not be computed. Make sure `shap` and `numba` are installed and compatible.")
            st.code(str(e))

    # ---- (B) Historical distribution for this route ----
    with viz_right:
        st.subheader("How does this compare to history?")
        hist_df = load_training_history()

        # Ensure consistent route key like "AAA-BBB"
        current_route = f"{dep_code}-{arr_code}"
        route_hist = hist_df[hist_df["route"].astype(str) == current_route]

        if route_hist.empty or "target" not in route_hist.columns:
            st.info("No historical delays found for this route in the training data.")
        else:
            st.caption(f"Route: {current_route} · {len(route_hist)} past flights")

            # Build a histogram of 'target' (minutes) and a vertical line for y_pred
            # Use Altair (ships well with Streamlit)
            import altair as alt
            base = alt.Chart(route_hist).transform_filter(
                alt.datum.target != None
            )

            hist = base.mark_bar(opacity=0.7).encode(
                alt.X("target:Q", bin=alt.Bin(maxbins=40), title="Historical delay (min)"),
                alt.Y("count()", title="Flights"),
                tooltip=[alt.Tooltip("count()", title="Flights")]
            )

            rule = alt.Chart(pd.DataFrame({"pred": [y_pred]})).mark_rule().encode(
                x="pred:Q",
                tooltip=[alt.Tooltip("pred:Q", title="Predicted delay (min)", format=".1f")]
            )

            # Mean / median markers (optional)
            stats = {
                "mean": float(route_hist["target"].mean()),
                "median": float(route_hist["target"].median())
            }
            stats_df = pd.DataFrame([
                {"stat": "mean", "val": stats["mean"]},
                {"stat": "median", "val": stats["median"]}
            ])
            stats_rule = alt.Chart(stats_df).mark_rule(strokeDash=[4,4]).encode(
                x="val:Q",
                color=alt.Color("stat:N", legend=alt.Legend(title="Stats"))
            )

            st.altair_chart((hist + rule + stats_rule).properties(height=260), use_container_width=True)

            st.caption(f"Mean: {stats['mean']:.1f} min · Median: {stats['median']:.1f} min")