# feature_builder.py
import json, pandas as pd, numpy as np
from datetime import timedelta
from pathlib import Path

NUM_COLS = ['duration','dep_temp','dep_precip','dep_wind','arr_temp',
            'arr_precip','arr_wind','holiday_length','num_passenger_year',
            'distance_km','expected_duration','delay_relative_to_expected',
            'duration_ratio','dep_lat','dep_long','arr_lat','arr_long']

CAT_COLS = ['departure_point','arrival_point','flight_status','aircraft_code','dep_hour',
            'dep_day','dep_month','dep_dayofweek','dep_quarter','dep_season',
            'dep_is_weekend','dep_time_of_day','arr_hour','arr_day','arr_month',
            'arr_dayofweek','arr_quarter','arr_season','arr_is_weekend',
            'arr_time_of_day','route','is_holiday','Country','City','aircraft_model']

DATETIME_COLS = ['departure_time','arrival_time','departure_date','arrival_date']

def _season(m):
    return {12:"winter",1:"winter",2:"winter",
            3:"spring",4:"spring",5:"spring",
            6:"summer",7:"summer",8:"summer",
            9:"autumn",10:"autumn",11:"autumn"}[m]

def _time_of_day(h):
    if 5<=h<12: return "morning"
    if 12<=h<17: return "afternoon"
    if 17<=h<22: return "evening"
    return "night"

def load_artifacts(path="artifacts"):
    path = Path(path)
    with open(path/"feature_medians.json") as f: med = json.load(f)
    with open(path/"category_modes.json") as f: mod = json.load(f)
    route_tbl = pd.read_csv(path/"route_expected_duration.csv")
    return med, mod, route_tbl

def build_features_minimal(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    raw_df must have: departure_point, arrival_point, departure_time, aircraft_model
    Optional: Country, City, flight_status, aircraft_code, is_holiday, weather fields...
    """
    df = raw_df.copy()
    df['route'] = df['departure_point'].astype(str) + '-' + df['arrival_point'].astype(str)
    df['departure_time'] = pd.to_datetime(df['departure_time'], errors="coerce")

    # time parts (dep)
    df['dep_hour'] = df['departure_time'].dt.hour
    df['dep_day'] = df['departure_time'].dt.day
    df['dep_month'] = df['departure_time'].dt.month
    df['dep_dayofweek'] = df['departure_time'].dt.dayofweek
    df['dep_quarter'] = df['departure_time'].dt.quarter
    df['dep_season'] = df['dep_month'].apply(_season)
    df['dep_is_weekend'] = df['dep_dayofweek'].isin([5,6]).astype(int)
    df['dep_time_of_day'] = df['dep_hour'].apply(_time_of_day)
    df['departure_date'] = df['departure_time'].dt.date

    # load artifacts
    med, mod, route_tbl = load_artifacts()
    route_map = dict(zip(route_tbl['route'], route_tbl['expected_duration_min']))

    # expected_duration from route median (mins)
    df['expected_duration'] = df['route'].map(route_map).fillna(med['expected_duration'])

    # simple ETA and arrival time features (based on expected duration)
    df['arrival_time'] = df['departure_time'] + pd.to_timedelta(df['expected_duration'], unit='m')
    df['arrival_date'] = df['arrival_time'].dt.date
    df['arr_hour'] = df['arrival_time'].dt.hour
    df['arr_day'] = df['arrival_time'].dt.day
    df['arr_month'] = df['arrival_time'].dt.month
    df['arr_dayofweek'] = df['arrival_time'].dt.dayofweek
    df['arr_quarter'] = df['arrival_time'].dt.quarter
    df['arr_season'] = df['arr_month'].apply(_season)
    df['arr_is_weekend'] = df['arr_dayofweek'].isin([5,6]).astype(int)
    df['arr_time_of_day'] = df['arr_hour'].apply(_time_of_day)

    # placeholders for unknowns (use medians/modes so the pipeline runs)
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = med.get(c, 0.0)
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = mod.get(c, str(mod.get('route','UNK')))

    # ensure some special fields exist
    df['flight_status'] = df.get('flight_status', 'SCHEDULED')
    df['is_holiday'] = df.get('is_holiday', 0)
    df['duration'] = df.get('duration', df['expected_duration'])
    df['delay_relative_to_expected'] = df.get('delay_relative_to_expected', 0.0)
    df['duration_ratio'] = df.get('duration_ratio', 1.0)

    # minimal geo/weather placeholders (better: compute distance_km, dep/arr weather)
    for c in ['dep_temp','dep_precip','dep_wind','arr_temp','arr_precip','arr_wind',
              'num_passenger_year','distance_km','dep_lat','dep_long','arr_lat','arr_long',
              'holiday_length']:
        if c not in df.columns:
            df[c] = med.get(c, 0.0)

    # reorder to model’s expectation
    final_cols = NUM_COLS + CAT_COLS + DATETIME_COLS
    for c in DATETIME_COLS:
        if c not in df.columns:
            df[c] = pd.NaT
    return df[final_cols]
