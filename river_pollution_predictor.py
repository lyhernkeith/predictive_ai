import os
import pandas as pd
import numpy as np
from flask import Flask
from dash import Dash, dash_table, html
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV


server = Flask(__name__)
app = Dash(__name__, server=server)

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

df["timestamp"] = pd.to_datetime(df["date"].astype(str) + df["time"].astype(str),
                                 format="%Y%m%d%H:%M", errors="coerce")
df = df.dropna(subset=["timestamp"])

for col in ["temperature", "pH", "turbidity"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[["station_id", "station_name", "timestamp", "temperature", "pH", "turbidity"]].dropna()
df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

monthly = df.groupby(["station_id", "month"], as_index=False).agg({
    "temperature": "mean",
    "pH": "mean",
    "turbidity": "mean"
})


monthly["pollution_score"] = 0.6 * monthly["turbidity"] + 0.4 * (monthly["pH"] - 7).abs()


scaler = StandardScaler()
monthly[["temperature", "pH", "turbidity"]] = scaler.fit_transform(monthly[["temperature", "pH", "turbidity"]])

monthly = monthly.sort_values(["station_id", "month"])
monthly["next_score"] = monthly.groupby("station_id")["pollution_score"].shift(-1)

for col in ["temperature", "pH", "turbidity"]:
    for lag in range(1, 7):
        monthly[f"{col}_lag{lag}"] = monthly.groupby("station_id")[col].shift(lag)
    monthly[f"{col}_ma3"] = monthly.groupby("station_id")[col].transform(lambda x: x.rolling(3).mean())
    monthly[f"{col}_ma6"] = monthly.groupby("station_id")[col].transform(lambda x: x.rolling(6).mean())


monthly['month_num'] = monthly['month'].dt.month
monthly['month_sin'] = np.sin(2 * np.pi * monthly['month_num']/12)
monthly['month_cos'] = np.cos(2 * np.pi * monthly['month_num']/12)

monthly = monthly.dropna()


monthly["delta"] = monthly["next_score"] - monthly["pollution_score"]
p5, p95 = np.percentile(monthly['delta'], [5, 95])
monthly['target'] = np.clip(monthly['delta'], p5, p95)
monthly['target'] = 2 * (monthly['target'] - p5) / (p95 - p5) - 1


features = []
for col in ["temperature", "pH", "turbidity"]:
    for lag in range(1, 7):
        features.append(f"{col}_lag{lag}")
    features.append(f"{col}_ma3")
    features.append(f"{col}_ma6")
features += ["month_sin", "month_cos"]

X = monthly[features]
y = monthly["target"]

split = int(len(monthly) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


param_grid = {
    "n_estimators": [500, 700, 900],
    "max_depth": [8, 10, 12, 15],
    "min_samples_leaf": [2, 3, 5],
    "max_features": ["sqrt", "log2", 0.5, 0.7],
    "min_samples_split": [2, 4, 6]
}
rf = RandomForestRegressor(random_state=42)
search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=20,
                            scoring="r2", cv=5, random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
best_model = search.best_estimator_


monthly_features = monthly.copy()
station_names = df[['station_id', 'station_name']].drop_duplicates()
monthly_features = monthly_features.merge(station_names, on='station_id', how='left')

latest = monthly_features.sort_values(["station_id", "month"]).groupby("station_id").tail(1)
latest["predicted_delta"] = best_model.predict(latest[features])

latest_table = latest[["station_name", "month", "predicted_delta"]]

def color_delta(val):
    if val > 0.01:
        color = 'tomato'
    elif val < -0.01:
        color = 'lightgreen'
    else:
        color = 'lightyellow'
    return f'background-color: {color}'

app.layout = html.Div([
    html.H2("Next Month River Pollution Predictions"),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in latest_table.columns],
        data=latest_table.to_dict('records'),
        style_data_conditional=[
            {
                'if': {'filter_query': '{predicted_delta} > 0.01', 'column_id': 'predicted_delta'},
                'backgroundColor': 'tomato', 'color': 'white'
            },
            {
                'if': {'filter_query': '{predicted_delta} < -0.01', 'column_id': 'predicted_delta'},
                'backgroundColor': 'lightgreen', 'color': 'black'
            },
            {
                'if': {'filter_query': '{predicted_delta} >= -0.01 && {predicted_delta} <= 0.01', 'column_id': 'predicted_delta'},
                'backgroundColor': 'lightyellow', 'color': 'black'
            },
        ],
        style_cell={'textAlign': 'center', 'padding': '5px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': 'lightgrey'}
    )
])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)
