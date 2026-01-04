# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# def main():
#     # ===============================
#     # 1. LOAD DATA
#     # ===============================
#     df = pd.read_csv("taiwan_river_data.csv")

#     # Rename columns from EPA dataset
#     df = df.rename(columns={
#         "監測站代碼": "station_id",
#         "採樣日期": "date",
#         "採樣時間": "time",
#         "水溫_溫度": "temperature",
#         "pH值_統計": "pH",
#         "懸浮固體_mg-L": "turbidity"
#     })

#     # ===============================
#     # 2. TIMESTAMP ALIGNMENT
#     # ===============================
#     df["timestamp"] = pd.to_datetime(
#         df["date"].astype(str) + df["time"].astype(str),
#         format="%Y%m%d%H:%M",
#         errors="coerce"
#     )
    
#     print(df[["station_id", "timestamp"]])


#     df = df.dropna(subset=["timestamp"])

#     # Keep only needed columns

#     # df = df[[
#     #     "station_id",
#     #     "timestamp",
#     #     "temperature",
#     #     "pH",
#     #     "turbidity"
#     # ]].dropna()

#     for col in ["temperature", "pH", "turbidity"]:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     df = df[[
#         "station_id",
#         "timestamp",
#         "temperature",
#         "pH",
#         "turbidity"
#     ]]

#     df = df.dropna()


#     # Align to month start
#     df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

#     # ===============================
#     # 3. MONTHLY AGGREGATION
#     # ===============================
#     monthly = (
#         df
#         .groupby(["station_id", "month"], as_index=False)
#         .agg({
#             "temperature": "mean",
#             "pH": "mean",
#             "turbidity": "mean"
#         })
#     )

#     # ===============================
#     # 4. POLLUTION SCORE (NO SS)
#     # ===============================
#     monthly["pollution_score"] = (
#         0.6 * monthly["turbidity"] +
#         0.4 * (monthly["pH"] - 7).abs()
#     )

#     # ===============================
#     # 5. LABEL CREATION
#     # ===============================
#     monthly = monthly.sort_values(["station_id", "month"])

#     monthly["next_score"] = (
#         monthly
#         .groupby("station_id")["pollution_score"]
#         .shift(-1)
#     )

#     # for col in ["turbidity", "pH", "temperature"]:
#     #     monthly[f"{col}_lag1"] = (
#     #         monthly
#     #         .groupby("station_id")[col]
#     #         .shift(1)
#     #     )

#     # monthly = monthly.dropna()
#     monthly["delta"] = monthly["next_score"] - monthly["pollution_score"]

#     print("Monthly rows:", len(monthly))
#     print("Stations with >=2 months:",
#         (monthly.groupby("station_id").size() >= 2).sum())


#     def make_label(delta, threshold=0.3):
#         if delta > threshold:
#             return 1     # more polluted
#         elif delta < -threshold:
#             return -1    # less polluted
#         return 0        # stable

#     monthly["target"] = monthly["delta"].apply(make_label)
#     monthly = monthly.dropna()

#     # ===============================
#     # 6. TRAIN / TEST SPLIT
#     # ===============================
#     features = ["temperature", "pH", "turbidity"]

#     X = monthly[features]
#     y = monthly["target"]

#     split = int(len(monthly) * 0.8)

#     X_train, X_test = X.iloc[:split], X.iloc[split:]
#     y_train, y_test = y.iloc[:split], y.iloc[split:]

#     # ===============================
#     # 7. MODEL
#     # ===============================
#     model = RandomForestClassifier(
#         n_estimators=300,
#         max_depth=7,
#         min_samples_leaf=5,
#         random_state=42
#     )

#     model.fit(X_train, y_train)

#     # ===============================
#     # 8. EVALUATION
#     # ===============================
#     pred = model.predict(X_test)

#     print("\nClassification Report:\n")
#     print(classification_report(y_test, pred))

#     print("\nFeature Importance:")
#     for f, i in sorted(
#         zip(features, model.feature_importances_),
#         key=lambda x: x[1],
#         reverse=True
#     ):
#         print(f"{f}: {i:.3f}")

#     # ===============================
#     # 9. LATEST PREDICTION
#     # ===============================
#     latest = (
#         monthly
#         .sort_values(["station_id", "month"])
#         .groupby("station_id")
#         .tail(1)
#     )

#     latest["prediction"] = model.predict(latest[features])

#     print("\nNext Month Prediction:")
#     print(latest[["station_id", "month", "prediction"]])

# if __name__ == "__main__":
#     main()


# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np




# def main():
#     # ===============================
#     # 1. LOAD DATA
#     # ===============================
#     df = pd.read_csv("taiwan_river_data.csv")

#     # Rename columns from EPA dataset
#     df = df.rename(columns={
#         "監測站代碼": "station_id",
#         "採樣日期": "date",
#         "採樣時間": "time",
#         "水溫_溫度": "temperature",
#         "pH值_統計": "pH",
#         "懸浮固體_mg-L": "turbidity"
#     })

#     # ===============================
#     # 2. TIMESTAMP ALIGNMENT
#     # ===============================
#     df["timestamp"] = pd.to_datetime(
#         df["date"].astype(str) + df["time"].astype(str),
#         format="%Y%m%d%H:%M",
#         errors="coerce"
#     )
    
#     df = df.dropna(subset=["timestamp"])

#     # Convert features to numeric
#     for col in ["temperature", "pH", "turbidity"]:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     df = df[["station_id", "timestamp", "temperature", "pH", "turbidity"]].dropna()

#     # Align to month start
#     df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

#     # ===============================
#     # 3. MONTHLY AGGREGATION
#     # ===============================
#     monthly = df.groupby(["station_id", "month"], as_index=False).agg({
#         "temperature": "mean",
#         "pH": "mean",
#         "turbidity": "mean"
#     })

#     # ===============================
#     # 4. POLLUTION SCORE
#     # ===============================
#     # Compute a pollution score (scaled)
#     monthly["pollution_score"] = 0.6 * monthly["turbidity"] + 0.4 * (monthly["pH"] - 7).abs()

#     # Standardize features
#     scaler = StandardScaler()
#     monthly[["temperature", "pH", "turbidity"]] = scaler.fit_transform(
#         monthly[["temperature", "pH", "turbidity"]]
#     )

#     # ===============================
#     # 5. CREATE LAGS AND DELTA
#     # ===============================
#     monthly = monthly.sort_values(["station_id", "month"])

#     # next month's pollution score
#     monthly["next_score"] = monthly.groupby("station_id")["pollution_score"].shift(-1)

#     # lag features
#     for col in ["turbidity", "pH", "temperature"]:
#         monthly[f"{col}_lag1"] = monthly.groupby("station_id")[col].shift(1)

#     monthly = monthly.dropna()
    
#     # delta = how much pollution changes next month
#     monthly["delta"] = monthly["next_score"] - monthly["pollution_score"]

#     # Normalize delta per station to [-1, 1]
#     def normalize(series):
#         min_val = series.min()
#         max_val = series.max()
#         if max_val - min_val == 0:
#             return series * 0  # if constant, return 0
#         return 2 * (series - min_val) / (max_val - min_val) - 1

#     monthly["target"] = monthly.groupby("station_id")["delta"].transform(normalize)

#     # ===============================
#     # 6. TRAIN / TEST SPLIT
#     # ===============================
#     features = ["temperature_lag1", "pH_lag1", "turbidity_lag1"]
#     X = monthly[features]
#     y = monthly["target"]

#     split = int(len(monthly) * 0.8)
#     X_train, X_test = X.iloc[:split], X.iloc[split:]
#     y_train, y_test = y.iloc[:split], y.iloc[split:]

#     # ===============================
#     # 7. MODEL
#     # ===============================
#     model = RandomForestRegressor(
#         n_estimators=300,
#         max_depth=7,
#         min_samples_leaf=5,
#         random_state=42
#     )
#     model.fit(X_train, y_train)

#     # ===============================
#     # 8. EVALUATION
#     # ===============================
#     y_pred = model.predict(X_test)

#     r2 = r2_score(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # manual RMSE

#     print("R2 Score:", r2)
#     print("RMSE:", rmse)


#     print("\nFeature Importance:")
#     for f, i in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
#         print(f"{f}: {i:.3f}")

#     # ===============================
#     # 9. LATEST PREDICTION
#     # ===============================
#     latest = monthly.sort_values(["station_id", "month"]).groupby("station_id").tail(1)
#     latest["predicted_delta"] = model.predict(latest[features])

#     print("\nNext Month Predicted Change (-1: better, +1: worse):")
#     print(latest[["station_id", "month", "predicted_delta"]])


# if __name__ == "__main__":
#     main()



# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score

# def main():
#     # ===============================
#     # 1. LOAD DATA
#     # ===============================
#     df = pd.read_csv("taiwan_river_data.csv")

#     # Rename columns from EPA dataset
#     df = df.rename(columns={
#         "監測站代碼": "station_id",
#         "採樣日期": "date",
#         "採樣時間": "time",
#         "水溫_溫度": "temperature",
#         "pH值_統計": "pH",
#         "懸浮固體_mg-L": "turbidity"
#     })

#     # ===============================
#     # 2. TIMESTAMP ALIGNMENT
#     # ===============================
#     df["timestamp"] = pd.to_datetime(
#         df["date"].astype(str) + df["time"].astype(str),
#         format="%Y%m%d%H:%M",
#         errors="coerce"
#     )
#     df = df.dropna(subset=["timestamp"])

#     # Convert features to numeric
#     for col in ["temperature", "pH", "turbidity"]:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     df = df[["station_id", "timestamp", "temperature", "pH", "turbidity"]].dropna()

#     # Align to month start
#     df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

#     # ===============================
#     # 3. MONTHLY AGGREGATION
#     # ===============================
#     monthly = df.groupby(["station_id", "month"], as_index=False).agg({
#         "temperature": "mean",
#         "pH": "mean",
#         "turbidity": "mean"
#     })

#     # ===============================
#     # 4. POLLUTION SCORE
#     # ===============================
#     monthly["pollution_score"] = 0.6 * monthly["turbidity"] + 0.4 * (monthly["pH"] - 7).abs()

#     # Standardize features
#     scaler = StandardScaler()
#     monthly[["temperature", "pH", "turbidity"]] = scaler.fit_transform(
#         monthly[["temperature", "pH", "turbidity"]]
#     )

#     # ===============================
#     # 5. CREATE LAGS AND ROLLING AVERAGES
#     # ===============================
#     monthly = monthly.sort_values(["station_id", "month"])

#     # next month's pollution score
#     monthly["next_score"] = monthly.groupby("station_id")["pollution_score"].shift(-1)

#     # multiple lags (1, 2, 3 months)
#     for col in ["temperature", "pH", "turbidity"]:
#         for lag in range(1, 4):
#             monthly[f"{col}_lag{lag}"] = monthly.groupby("station_id")[col].shift(lag)
#         # rolling average over last 3 months
#         monthly[f"{col}_ma3"] = monthly.groupby("station_id")[col].transform(lambda x: x.rolling(3).mean())

#     monthly = monthly.dropna()

#     # delta = change in pollution score
#     monthly["delta"] = monthly["next_score"] - monthly["pollution_score"]

#     # Normalize delta globally to [-1, 1]
#     delta_min = monthly["delta"].min()
#     delta_max = monthly["delta"].max()
#     if delta_max - delta_min != 0:
#         monthly["target"] = 2 * (monthly["delta"] - delta_min) / (delta_max - delta_min) - 1
#     else:
#         monthly["target"] = monthly["delta"] * 0  # constant case

#     # ===============================
#     # 6. TRAIN / TEST SPLIT
#     # ===============================
#     features = [
#         "temperature_lag1", "temperature_lag2", "temperature_lag3", "temperature_ma3",
#         "pH_lag1", "pH_lag2", "pH_lag3", "pH_ma3",
#         "turbidity_lag1", "turbidity_lag2", "turbidity_lag3", "turbidity_ma3"
#     ]

#     X = monthly[features]
#     y = monthly["target"]

#     split = int(len(monthly) * 0.8)
#     X_train, X_test = X.iloc[:split], X.iloc[split:]
#     y_train, y_test = y.iloc[:split], y.iloc[split:]

#     # ===============================
#     # 7. MODEL
#     # ===============================
#     model = RandomForestRegressor(
#         n_estimators=500,
#         max_depth=10,
#         min_samples_leaf=3,
#         random_state=42
#     )
#     model.fit(X_train, y_train)

#     # ===============================
#     # 8. EVALUATION
#     # ===============================
#     y_pred = model.predict(X_test)

#     r2 = r2_score(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     print("R2 Score:", r2)
#     print("RMSE:", rmse)

#     print("\nFeature Importance:")
#     for f, i in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
#         print(f"{f}: {i:.3f}")

#     # ===============================
#     # 9. LATEST PREDICTION
#     # ===============================
#     latest = monthly.sort_values(["station_id", "month"]).groupby("station_id").tail(1)
#     latest["predicted_delta"] = model.predict(latest[features])

#     print("\nNext Month Predicted Change (-1: better, +1: worse):")
#     print(latest[["station_id", "month", "predicted_delta"]])


# if __name__ == "__main__":
#     main()



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # ===============================
    # 1. LOAD DATA
    # ===============================
    df = pd.read_csv("taiwan_river_data.csv")

    # Rename columns from EPA dataset
    df = df.rename(columns={
        "監測站代碼": "station_id",
        "採樣日期": "date",
        "採樣時間": "time",
        "水溫_溫度": "temperature",
        "pH值_統計": "pH",
        "懸浮固體_mg-L": "turbidity"
    })

    # ===============================
    # 2. TIMESTAMP ALIGNMENT
    # ===============================
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

    # ===============================
    # 3. MONTHLY AGGREGATION
    # ===============================
    monthly = df.groupby(["station_id", "month"], as_index=False).agg({
        "temperature": "mean",
        "pH": "mean",
        "turbidity": "mean"
    })

    # ===============================
    # 4. POLLUTION SCORE
    # ===============================
    monthly["pollution_score"] = 0.6 * monthly["turbidity"] + 0.4 * (monthly["pH"] - 7).abs()

    # Standardize main features
    scaler = StandardScaler()
    monthly[["temperature", "pH", "turbidity"]] = scaler.fit_transform(
        monthly[["temperature", "pH", "turbidity"]]
    )

    # ===============================
    # 5. LAGS, ROLLING AVERAGES, SEASONALITY
    # ===============================
    monthly = monthly.sort_values(["station_id", "month"])
    monthly["next_score"] = monthly.groupby("station_id")["pollution_score"].shift(-1)

    for col in ["temperature", "pH", "turbidity"]:
        # 6-month lags
        for lag in range(1, 7):
            monthly[f"{col}_lag{lag}"] = monthly.groupby("station_id")[col].shift(lag)
        # rolling averages
        monthly[f"{col}_ma3"] = monthly.groupby("station_id")[col].transform(lambda x: x.rolling(3).mean())
        monthly[f"{col}_ma6"] = monthly.groupby("station_id")[col].transform(lambda x: x.rolling(6).mean())

    # seasonality
    monthly['month_num'] = monthly['month'].dt.month
    monthly['month_sin'] = np.sin(2 * np.pi * monthly['month_num']/12)
    monthly['month_cos'] = np.cos(2 * np.pi * monthly['month_num']/12)

    monthly = monthly.dropna()

    # ===============================
    # 6. TARGET DELTA SCALING
    # ===============================
    monthly["delta"] = monthly["next_score"] - monthly["pollution_score"]

    # Percentile scaling to [-1, 1]
    p5, p95 = np.percentile(monthly['delta'], [5, 95])
    monthly['target'] = np.clip(monthly['delta'], p5, p95)
    monthly['target'] = 2 * (monthly['target'] - p5) / (p95 - p5) - 1

    # ===============================
    # 7. TRAIN / TEST SPLIT
    # ===============================
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

    # ===============================
    # 8. RANDOM FOREST MODEL
    # ===============================
    model = RandomForestRegressor(
        n_estimators=700,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ===============================
    # 9. EVALUATION
    # ===============================
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2)
    print("RMSE:", rmse)

    print("\nFeature Importance:")
    for f, i in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"{f}: {i:.3f}")

    # ===============================
    # 10. LATEST PREDICTION
    # ===============================
    latest = monthly.sort_values(["station_id", "month"]).groupby("station_id").tail(1)
    latest["predicted_delta"] = model.predict(latest[features])

    print("\nNext Month Predicted Change (-1: better, +1: worse):")
    print(latest[["station_id", "month", "predicted_delta"]])

if __name__ == "__main__":
    main()
