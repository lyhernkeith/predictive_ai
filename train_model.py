import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

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
for col in ["temperature", "pH", "turbidity"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[["station_id", "timestamp", "temperature", "pH", "turbidity"]].dropna()
df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

monthly = df.groupby(["station_id", "month"], as_index=False).mean()

monthly["pollution_score"] = (
    0.6 * monthly["turbidity"] +
    0.4 * (monthly["pH"] - 7).abs()
)

scaler = StandardScaler()
monthly[["temperature", "pH", "turbidity"]] = scaler.fit_transform(
    monthly[["temperature", "pH", "turbidity"]]
)

monthly = monthly.sort_values(["station_id", "month"])
monthly["next_score"] = monthly.groupby("station_id")["pollution_score"].shift(-1)

for col in ["temperature", "pH", "turbidity"]:
    for lag in range(1, 7):
        monthly[f"{col}_lag{lag}"] = monthly.groupby("station_id")[col].shift(lag)
    monthly[f"{col}_ma3"] = monthly.groupby("station_id")[col].rolling(3).mean().reset_index(0, drop=True)
    monthly[f"{col}_ma6"] = monthly.groupby("station_id")[col].rolling(6).mean().reset_index(0, drop=True)

monthly["month_num"] = monthly["month"].dt.month
monthly["month_sin"] = np.sin(2 * np.pi * monthly["month_num"] / 12)
monthly["month_cos"] = np.cos(2 * np.pi * monthly["month_num"] / 12)

monthly = monthly.dropna()

monthly["delta"] = monthly["next_score"] - monthly["pollution_score"]
p5, p95 = np.percentile(monthly["delta"], [5, 95])
monthly["target"] = np.clip(monthly["delta"], p5, p95)
monthly["target"] = 2 * (monthly["target"] - p5) / (p95 - p5) - 1

features = [
    c for c in monthly.columns
    if "lag" in c or "ma" in c or c in ["month_sin", "month_cos"]
]

X = monthly[features]
y = monthly["target"]

rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)

# joblib.dump((rf, features), "model.pkl")
joblib.dump((rf, features, p5, p95), "model.pkl")

print("Model saved as model.pkl")

