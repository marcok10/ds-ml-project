# Home.py — Flight Delay Predictions (airports from route_expected_duration.csv)
# Run: streamlit run Home.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
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

@st.cache_data
def load_training_history() -> pd.DataFrame:
    """Load historical training data with `target` (actual delay)."""
    p = Path("data/data_all_col_upto_duration_ratio.csv")
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "route" not in df.columns and {"departure_point", "arrival_point"}.issubset(df.columns):
        df["route"] = df["departure_point"].astype(str) + "-" + df["arrival_point"].astype(str)
    return df

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

        st.divider()

        # ===================== VISUALS =====================

        # (A) SHAP explanation for the XGBoost base model
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Why this prediction? (SHAP)")

            try:
                import shap  # needs shap + numba in requirements
                # Build encoded feature row exactly like training
                X_enc = model._transform_encoded(X)
                X_dense = X_enc.toarray().astype(np.float32)

                # Feature names = numeric columns + encoder one-hot names
                num_names = list(model.cols.num_cols)
                cat_names = []
                if getattr(model, "encoder_", None) is not None:
                    try:
                        cat_names = list(model.encoder_.get_feature_names_out(model.cols.cat_cols))
                    except Exception:
                        # Fallback if older sklearn
                        cat_names = list(model.encoder_.get_feature_names(model.cols.cat_cols))
                feat_names = num_names + cat_names

                # Prefer booster for stability
                booster = getattr(model, "xgb_booster_", None)
                if booster is not None:
                    explainer = shap.TreeExplainer(booster)
                    shap_vals = explainer.shap_values(X_dense)
                else:
                    explainer = shap.TreeExplainer(model.xgb_)
                    shap_vals = explainer.shap_values(X_dense)

                vals = np.array(shap_vals).reshape(-1)
                shap_df = (
                    pd.DataFrame({"feature": feat_names, "shap_value": vals})
                    .assign(abs_val=lambda d: d["shap_value"].abs())
                    .sort_values("abs_val", ascending=False)
                    .head(10)
                )

                st.write("Top drivers (positive ↑ increases delay; negative ↓ decreases)")
                st.bar_chart(shap_df.set_index("feature")["shap_value"])
                with st.expander("See SHAP details"):
                    st.dataframe(shap_df[["feature", "shap_value"]], use_container_width=True)

            except Exception as e:
                st.warning("SHAP explanation unavailable. Check that `shap` and `numba` are installed and compatible.")
                st.code(str(e))

        # (B) Historical distribution for this route
        with right:
            st.subheader("How does it compare to history?")
            hist_df = load_training_history()
            current_route = f"{dep}-{arr}"
            route_hist = hist_df[hist_df.get("route", "").astype(str) == current_route] if not hist_df.empty else pd.DataFrame()

            if route_hist.empty or "target" not in route_hist.columns:
                st.info("No historical delays found for this route in the training data.")
            else:
                st.caption(f"Route: {current_route} · {len(route_hist)} past flights")

                import altair as alt
                base = alt.Chart(route_hist)

                hist = base.mark_bar(opacity=0.75).encode(
                    alt.X("target:Q", bin=alt.Bin(maxbins=40), title="Historical delay (min)"),
                    alt.Y("count()", title="Flights"),
                    tooltip=[alt.Tooltip("count()", title="Flights")]
                )

                pred_line = alt.Chart(pd.DataFrame({"pred": [y]})).mark_rule().encode(
                    x="pred:Q",
                    tooltip=[alt.Tooltip("pred:Q", title="Predicted delay (min)", format=".1f")]
                )

                stats_df = pd.DataFrame({
                    "stat": ["mean", "median"],
                    "val":  [route_hist["target"].mean(), route_hist["target"].median()]
                })
                stats_rule = alt.Chart(stats_df).mark_rule(strokeDash=[4,4]).encode(
                    x="val:Q",
                    color=alt.Color("stat:N", legend=alt.Legend(title="Stats"))
                )

                st.altair_chart((hist + pred_line + stats_rule).properties(height=260), use_container_width=True)
                st.caption(f"Mean: {route_hist['target'].mean():.1f} · Median: {route_hist['target'].median():.1f} (minutes)")
