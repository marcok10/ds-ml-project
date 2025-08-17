# train_pipeline.py
import joblib
import pandas as pd

from hype_pipeline import HypeStacker, ColumnsConfig

# 1) Load the same fully-engineered training data you used for the _hype models
#    (i.e., includes all feature columns like route, dep_hour, expected_duration, etc.)
df = pd.read_csv("data/data_all_col_upto_duration_ratio.csv")

y = df["target"]
X = df.drop(columns=["target"])

# 2) These MUST match your dataset columns exactly
num_col = [
    'duration','dep_temp','dep_precip','dep_wind','arr_temp',
    'arr_precip','arr_wind','holiday_length','num_passenger_year',
    'distance_km','expected_duration','delay_relative_to_expected',
    'duration_ratio','dep_lat','dep_long','arr_lat','arr_long'
]

cat_col = [
    'departure_point','arrival_point','flight_status','aircraft_code','dep_hour',
    'dep_day','dep_month','dep_dayofweek','dep_quarter','dep_season',
    'dep_is_weekend','dep_time_of_day','arr_hour','arr_day','arr_month',
    'arr_dayofweek','arr_quarter','arr_season','arr_is_weekend',
    'arr_time_of_day','route','is_holiday','Country','City','aircraft_model'
]

# If these exist in your dataframe, CatBoost will drop them internally via _drop_dt_for_catboost()
datetime_cols = ['departure_time','arrival_time','departure_date','arrival_date']

cols = ColumnsConfig(num_cols=num_col, cat_cols=cat_col, datetime_cols=datetime_cols)

# 3) Train the end-to-end stacker
model = HypeStacker(cols=cols, random_state=42, test_size=0.2)
model.fit(X, y)

# 4) Save a single artifact for Streamlit / API
joblib.dump(model, "model.joblib")
print("✅ Saved model.joblib")
