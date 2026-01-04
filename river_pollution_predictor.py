import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask
from dash import Dash, dash_table, html

# =========================
# App setup
# =========================
server = Flask(__name__)
app = Dash(__name__, server=server)

# =========================
# Load model (STRICT)
# =========================
model, scaler, features, p5, p95 = joblib.load("model.pkl")

# =========================
# Load & clean data
# =========================
df = pd.read_csv("taiwan_river_data.csv")

df = df.rename(columns={
    "監測站代碼": "station_id",
    "監測站名": "station_name",
    "採樣日期": "date",
    "採樣時間": "time",
    "水溫_溫度": "temperature",
    "pH值_統計": "pH",
    "懸浮固體_mg-L": "turbidity"
})

for col in ["temperature", "pH", "turbidity"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["timestamp"] = pd.to_datetime(
    df["date"].astype(str) + df["time"].astype(str),
    format="%Y%m%d%H:%M",
    errors="coerce"
)

df = df.dropna(subset=["timestamp"])

# =========================
# Monthly aggregation
# =========================
df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

monthly = df.groupby(
    ["station_id", "station_name", "month"],
    as_index=False
).agg({
    "temperature": "mean",
    "pH": "mean",
    "turbidity": "mean"
})

monthly["pollution_score"] = (
    0.6 * monthly["turbidity"] +
    0.4 * (monthly["pH"] - 7).abs()
)

monthly = monthly.sort_values(["station_id", "month"])

# =========================
# Feature engineering (IDENTICAL)
# =========================
for col in ["temperature", "pH", "turbidity"]:
    for lag in range(1, 7):
        monthly[f"{col}_lag{lag}"] = monthly.groupby("station_id")[col].shift(lag)

    monthly[f"{col}_ma3"] = (
        monthly.groupby("station_id")[col].rolling(3).mean().reset_index(0, drop=True)
    )

    monthly[f"{col}_ma6"] = (
        monthly.groupby("station_id")[col].rolling(6).mean().reset_index(0, drop=True)
    )

monthly["month_num"] = monthly["month"].dt.month
monthly["month_sin"] = np.sin(2 * np.pi * monthly["month_num"] / 12)
monthly["month_cos"] = np.cos(2 * np.pi * monthly["month_num"] / 12)

monthly = monthly.dropna()

# =========================
# Predict latest month
# =========================
latest = (
    monthly
    .sort_values(["station_id", "month"])
    .groupby("station_id")
    .tail(1)
)

X_latest = scaler.transform(latest[features])
latest["predicted_delta"] = model.predict(X_latest)
latest["predicted_delta"] = latest["predicted_delta"].clip(p5, p95)

# =========================
# Table
# =========================
table = latest.loc[:, ["station_name", "month", "predicted_delta"]].copy()
table["month"] = table["month"].dt.strftime("%Y-%m")

app.layout = html.Div([
    html.H2("Next-Month River Pollution Change Prediction"),
    dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in table.columns],
        data=table.to_dict("records"),
        page_size=15
    )
])

# =========================
# Run
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)
