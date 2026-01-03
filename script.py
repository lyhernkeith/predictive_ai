import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv("taiwan_river_data.csv")

# Rename columns if needed (adjust names to match your CSV)
df = df.rename(columns={
    "站號": "station_id",
    "日期": "date",
    "水溫": "temperature",
    "濁度": "turbidity",
    "SS": "ss"
})

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Keep only required columns
df = df[[
    "station_id",
    "date",
    "temperature",
    "pH",
    "turbidity",
    "ss"
]]

# Drop rows with missing critical values
df = df.dropna()
