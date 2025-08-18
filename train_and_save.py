# train_and_save.py
import os
import joblib
import pandas as pd
from hype_pipeline import HypeStacker, ColumnsConfig

# --- paths ---
DATA = "data/data_all_col_upto_duration_ratio.csv"   # <-- your training CSV
MODEL_OUT = "model.joblib"                           # saved pipeline
ARTIFACTS_DIR = "artifacts"

# --- columns from your project ---
num_cols = [
    "duration","dep_temp","dep_precip","dep_wind","arr_temp",
    "arr_precip","arr_wind","holiday_length","num_passenger_year",
    "distance_km","expected_duration","delay_relative_to_expected",
    "duration_ratio","dep_lat","dep_long","arr_lat","arr_long"
]

cat_cols = [
    "departure_point","arrival_point","flight_status","aircraft_code","dep_hour",
    "dep_day","dep_month","dep_dayofweek","dep_quarter","dep_season",
    "dep_is_weekend","dep_time_of_day","arr_hour","arr_day","arr_month",
    "arr_dayofweek","arr_quarter","arr_season","arr_is_weekend",
    "arr_time_of_day","route","is_holiday","Country","City","aircraft_model"
]

# if these exist in your data, we drop them for CatBoost (it can’t take datetimes)
datetime_cols = ["departure_time","arrival_time","departure_date","arrival_date"]

def main():
    # 1) load data
    df = pd.read_csv(DATA)

    # 2) y and X
    if "target" not in df.columns:
        raise ValueError("Column 'target' not found in the training CSV.")
    y = df["target"]
    X = df.drop(columns=["target"])

    # 3) configure & train stack
    cols = ColumnsConfig(num_cols=num_cols, cat_cols=cat_cols, datetime_cols=datetime_cols)
    model = HypeStacker(cols=cols, random_state=42, test_size=0.2, artifacts_dir=ARTIFACTS_DIR)
    model.fit(X, y)

    # 4) save top-level pipeline (artifacts already saved by .fit())
    joblib.dump(model, MODEL_OUT)
    print(f"\nSaved: {MODEL_OUT}")
    print(f"Artifacts in: {os.path.abspath(ARTIFACTS_DIR)}")
    print("Done ✅")

if __name__ == "__main__":
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    main()