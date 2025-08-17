# make_artifacts.py
import json, pandas as pd, numpy as np
df = pd.read_csv("data/data_all_col_upto_duration_ratio.csv")

num_cols = ['duration','dep_temp','dep_precip','dep_wind','arr_temp',
            'arr_precip','arr_wind','holiday_length','num_passenger_year',
            'distance_km','expected_duration','delay_relative_to_expected',
            'duration_ratio','dep_lat','dep_long','arr_lat','arr_long']

cat_cols = ['departure_point','arrival_point','flight_status','aircraft_code','dep_hour',
            'dep_day','dep_month','dep_dayofweek','dep_quarter','dep_season',
            'dep_is_weekend','dep_time_of_day','arr_hour','arr_day','arr_month',
            'arr_dayofweek','arr_quarter','arr_season','arr_is_weekend',
            'arr_time_of_day','route','is_holiday','Country','City','aircraft_model']

df['route'] = df['route'].astype(str)

route_table = (df.groupby('route')['expected_duration']
                 .median().rename('expected_duration_min').reset_index())

med = df[num_cols].median(numeric_only=True).to_dict()
mod = df[cat_cols].mode(dropna=True).iloc[0].to_dict()

import os, pathlib
pathlib.Path("artifacts").mkdir(exist_ok=True)
pd.DataFrame(route_table).to_csv("artifacts/route_expected_duration.csv", index=False)
with open("artifacts/feature_medians.json","w") as f: json.dump({k: float(v) for k,v in med.items()}, f)
with open("artifacts/category_modes.json","w") as f: json.dump(mod, f)
print("Artifacts written to ./artifacts")