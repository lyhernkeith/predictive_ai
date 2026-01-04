import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask
from dash import Dash, dash_table, html

print("Starting app...")

server = Flask(__name__)
app = Dash(__name__, server=server)


model, features = joblib.load("model.pkl")

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


df["timestamp"] = pd.to_datetime(
    df["date"].astype(str) + df["time"].astype(str),
    format="%Y%m%d%H:%M",
    errors="coerce"
)

df = df.dropna(subset=["timestamp"])
df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()


numeric_cols = ["temperature", "pH", "turbidity"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols)


monthly_numeric = (
    df
    .groupby(["station_id", "month"], as_index=False)[numeric_cols]
    .mean()
)

monthly_names = (
    df
    .groupby("station_id", as_index=False)["station_name"]
    .first()
)

monthly = monthly_numeric.merge(monthly_names, on="station_id", how="left")

monthly["pollution_score"] = (
    0.6 * monthly["turbidity"] +
    0.4 * (monthly["pH"] - 7).abs()
)


monthly = monthly.sort_values(["station_id", "month"])
monthly["next_score"] = monthly.groupby("station_id")["pollution_score"].shift(-1)

for col in numeric_cols:
    for lag in range(1, 7):
        monthly[f"{col}_lag{lag}"] = monthly.groupby("station_id")[col].shift(lag)

    monthly[f"{col}_ma3"] = (
        monthly.groupby("station_id")[col]
        .rolling(3)
        .mean()
        .reset_index(0, drop=True)
    )

    monthly[f"{col}_ma6"] = (
        monthly.groupby("station_id")[col]
        .rolling(6)
        .mean()
        .reset_index(0, drop=True)
    )


monthly["month_num"] = monthly["month"].dt.month
monthly["month_sin"] = np.sin(2 * np.pi * monthly["month_num"] / 12)
monthly["month_cos"] = np.cos(2 * np.pi * monthly["month_num"] / 12)

monthly = monthly.dropna()


latest = (
    monthly
    .sort_values(["station_id", "month"])
    .groupby("station_id")
    .tail(1)
)

latest["predicted_delta"] = model.predict(latest[features])

table = latest[["station_name", "month", "predicted_delta"]]


app.layout = html.Div([
    html.H2("Next Month River Pollution Predictions"),
    dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in table.columns],
        data=table.to_dict("records"),
    )
])

print("App ready.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)
