import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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
    
    print(df[["station_id", "timestamp"]])


    df = df.dropna(subset=["timestamp"])

    # Keep only needed columns

    # df = df[[
    #     "station_id",
    #     "timestamp",
    #     "temperature",
    #     "pH",
    #     "turbidity"
    # ]].dropna()

    for col in ["temperature", "pH", "turbidity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[[
        "station_id",
        "timestamp",
        "temperature",
        "pH",
        "turbidity"
    ]]

    df = df.dropna()


    # Align to month start
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

    # ===============================
    # 3. MONTHLY AGGREGATION
    # ===============================
    monthly = (
        df
        .groupby(["station_id", "month"], as_index=False)
        .agg({
            "temperature": "mean",
            "pH": "mean",
            "turbidity": "mean"
        })
    )

    # ===============================
    # 4. POLLUTION SCORE (NO SS)
    # ===============================
    monthly["pollution_score"] = (
        0.6 * monthly["turbidity"] +
        0.4 * (monthly["pH"] - 7).abs()
    )

    # ===============================
    # 5. LABEL CREATION
    # ===============================
    monthly = monthly.sort_values(["station_id", "month"])

    monthly["next_score"] = (
        monthly
        .groupby("station_id")["pollution_score"]
        .shift(-1)
    )

    # for col in ["turbidity", "pH", "temperature"]:
    #     monthly[f"{col}_lag1"] = (
    #         monthly
    #         .groupby("station_id")[col]
    #         .shift(1)
    #     )

    # monthly = monthly.dropna()
    monthly["delta"] = monthly["next_score"] - monthly["pollution_score"]

    print("Monthly rows:", len(monthly))
    print("Stations with >=2 months:",
        (monthly.groupby("station_id").size() >= 2).sum())


    def make_label(delta, threshold=0.3):
        if delta > threshold:
            return 1     # more polluted
        elif delta < -threshold:
            return -1    # less polluted
        return 0        # stable

    monthly["target"] = monthly["delta"].apply(make_label)
    monthly = monthly.dropna()

    # ===============================
    # 6. TRAIN / TEST SPLIT
    # ===============================
    features = ["temperature", "pH", "turbidity"]

    X = monthly[features]
    y = monthly["target"]

    split = int(len(monthly) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ===============================
    # 7. MODEL
    # ===============================
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ===============================
    # 8. EVALUATION
    # ===============================
    pred = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, pred))

    print("\nFeature Importance:")
    for f, i in sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{f}: {i:.3f}")

    # ===============================
    # 9. LATEST PREDICTION
    # ===============================
    latest = (
        monthly
        .sort_values(["station_id", "month"])
        .groupby("station_id")
        .tail(1)
    )

    latest["prediction"] = model.predict(latest[features])

    print("\nNext Month Prediction:")
    print(latest[["station_id", "month", "prediction"]])

if __name__ == "__main__":
    main()
