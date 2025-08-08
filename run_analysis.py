# run_analysis.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from math import sqrt

def main():
    # 1) Load the data
    df = pd.read_csv("strategy_results.csv")
    print("Loaded", len(df), "rows.")

    # 2) Quick group-by summary per strategy
    summary = (
        df
        .groupby(["buy_orange","buy_light_blue","buy_stations","buy_utilities","max_development_level"])
        .agg(avg_roi=("roi","mean"), win_rate=("win","mean"))
        .reset_index()
        .sort_values("avg_roi", ascending=False)
    )
    print("\nTop 5 strategies by average ROI:")
    print(summary.head(), "\n")

    # 3) Prepare features & target
    features = ["buy_orange", "buy_light_blue", "buy_stations", "buy_utilities", "max_development_level"]
    X = df[features]
    y = df["roi"]

    # 4) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5) Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 6) Evaluate
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}\n")

    # 7) Feature importances
    importances = sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1])
    print("Feature importances:")
    for feat, imp in importances:
        print(f"  {feat}: {imp:.3f}")

    # 8) Plot Actual vs Predicted ROI
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--")
    plt.xlabel("Actual ROI")
    plt.ylabel("Predicted ROI")
    plt.title("Actual vs Predicted ROI")
    plt.tight_layout()
    plt.savefig("roi_actual_vs_predicted.png")
    print("\nScatter plot saved to roi_actual_vs_predicted.png")

if __name__ == "__main__":
    main()
