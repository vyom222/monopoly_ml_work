# shap_explain.py
"""
Load model.joblib and aggregated_strategies.csv, compute SHAP values and save summary plot + CSV.
"""

import joblib
import pandas as pd
import os

MODEL_PATH = "rf_strategy_winrate_model.joblib"
AGG_CSV = "aggregated_strategies.csv"
SHAP_SUMMARY_PNG = "shap_summary.png"
SHAP_VALUES_CSV = "shap_values_summary.csv"

STRATEGY_COLS = [
    'buy_brown', 'buy_light_blue', 'buy_pink', 'buy_orange',
    'buy_red', 'buy_yellow', 'buy_green', 'buy_indigo',
    'buy_stations', 'buy_utilities',
    'max_development_level', 'unspendable_cash'
]

try:
    import shap
    import matplotlib.pyplot as plt
except Exception as e:
    print("shap or matplotlib not available:", e)
    print("Install shap with: pip install shap matplotlib")
    raise

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run aggregate_and_train.py first.")

    if not os.path.exists(AGG_CSV):
        raise FileNotFoundError(f"Aggregated CSV not found at {AGG_CSV}. Run aggregate_and_train.py first.")

    model = joblib.load(MODEL_PATH)
    grouped = pd.read_csv(AGG_CSV)

    X = grouped[STRATEGY_COLS].astype(float)

    # SHAP TreeExplainer (works well for tree models)
    explainer = shap.TreeExplainer(model)
    print("Computing SHAP values (this may take a little while)...")
    shap_values = explainer.shap_values(X)  # shape: (n_samples, n_features)

    # Summary plot (beeswarm)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, feature_names=STRATEGY_COLS, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PNG, dpi=150)
    print("Saved SHAP summary plot:", SHAP_SUMMARY_PNG)

    # Save mean(|shap|) per feature as a CSV for a concise importance table
    mean_abs = pd.DataFrame({
        'feature': STRATEGY_COLS,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    mean_abs.to_csv(SHAP_VALUES_CSV, index=False)
    print("Saved SHAP feature importances to:", SHAP_VALUES_CSV)

if __name__ == "__main__":
    import numpy as np
    main()
