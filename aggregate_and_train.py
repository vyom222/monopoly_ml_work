# aggregate_and_train.py
"""
Aggregate per-game CSV into per-strategy rows, train a model to predict strategy win-rate (ignoring ROI),
and save outputs:
 - aggregated_strategies.csv      (aggregated stats per strategy)
 - strategies_by_predicted.csv    (per-strategy predicted and actual win rates)
 - model.joblib                   (trained sklearn model)
 - feature_importances.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# CONFIG
INPUT_CSV = "strategy_results.csv"
AGG_CSV = "aggregated_strategies.csv"
OUT_PRED_CSV = "strategies_by_predicted.csv"
MODEL_PATH = "rf_strategy_winrate_model.joblib"
FEATURE_IMPORTANCES_CSV = "feature_importances.csv"

# Strategy-defining columns (these will be used as features)
STRATEGY_COLS = [
    'buy_brown', 'buy_light_blue', 'buy_pink', 'buy_orange',
    'buy_red', 'buy_yellow', 'buy_green', 'buy_indigo',
    'buy_stations', 'buy_utilities',
    'max_development_level', 'unspendable_cash'
]

# Columns to aggregate for diagnostics (not used as predictors)
AGG_DIAG_COLS = ['props', 'houses', 'hotels', 'turns']

# Smoothing parameter for win-rate (small n_games -> shrink toward global mean)
SMOOTH_K = 5  # increase to shrink more; set to 0 to disable smoothing

RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_and_aggregate(path):
    print("Loading:", path)
    df = pd.read_csv(path)
    # Ensure 'win' is integer (0/1)
    df['win'] = df['win'].astype(int)

    # Convert booleans/strings to ints for strategy columns (safety)
    for c in STRATEGY_COLS:
        if c in df.columns:
            if df[c].dtype == 'bool' or df[c].dtype == object:
                # object could be "True"/"False" strings; astype(int) will fail; handle explicitly
                df[c] = df[c].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(df[c])
            df[c] = df[c].astype(int)
        else:
            raise KeyError(f"Missing expected strategy column: {c}")

    # Aggregate by strategy
    grouped = df.groupby(STRATEGY_COLS).agg(
        n_games=('win', 'count'),
        wins=('win', 'sum'),
        avg_props=('props', 'mean') if 'props' in df.columns else ('props', 'count'),
        avg_houses=('houses', 'mean') if 'houses' in df.columns else ('houses', 'count'),
        avg_hotels=('hotels', 'mean') if 'hotels' in df.columns else ('hotels', 'count'),
        avg_turns=('turns', 'mean') if 'turns' in df.columns else ('turns', 'count'),
    ).reset_index()

    # Compute raw win rate and optional smoothing
    grouped['raw_win_rate'] = grouped['wins'] / grouped['n_games']
    global_mean = (grouped['wins'].sum()) / (grouped['n_games'].sum())
    if SMOOTH_K > 0:
        grouped['smoothed_win_rate'] = (grouped['wins'] + SMOOTH_K * global_mean) / (grouped['n_games'] + SMOOTH_K)
    else:
        grouped['smoothed_win_rate'] = grouped['raw_win_rate']

    return df, grouped

def train_model_on_strategy_table(grouped):
    # Features: strategy columns (already numeric)
    X = grouped[STRATEGY_COLS].astype(float)
    # Target: smoothed_win_rate (to reduce label noise)
    y = grouped['smoothed_win_rate'].values

    # Use n_games as sample_weight so strategies with more games matter more
    sample_weight = grouped['n_games'].values

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weight, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    # quick (optional) small-grid tune for max_depth / min_samples_leaf (fast)
    param_grid = {
        "max_depth": [8, 12, None],
        "min_samples_leaf": [1, 2, 4]
    }
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    print("Running quick grid-search (3-fold) to pick good params (this is optional but recommended)...")
    grid.fit(X_train, y_train, sample_weight=w_train)
    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    # Evaluate
    y_pred_test = best.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test, sample_weight=w_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test, sample_weight=w_test)
    print(f"Test RMSE: {rmse:.4f}, Test R2: {r2:.4f}")

    return best

def save_outputs(model, grouped, raw_df):
    # Save aggregated data
    grouped.to_csv(AGG_CSV, index=False)
    print("Saved aggregated strategies:", AGG_CSV)

    # Predict for all strategies
    X_all = grouped[STRATEGY_COLS].astype(float)
    grouped['predicted_win_rate'] = model.predict(X_all)

    # Order by prediction and by actual for comparison
    by_pred = grouped.sort_values('predicted_win_rate', ascending=False)
    by_actual = grouped.sort_values('raw_win_rate', ascending=False)

    by_pred.to_csv(OUT_PRED_CSV, index=False)
    print("Saved strategies ordered by predicted win rate:", OUT_PRED_CSV)

    # Feature importances
    importances = model.feature_importances_
    fi = pd.DataFrame({'feature': STRATEGY_COLS, 'importance': importances}).sort_values('importance', ascending=False)
    fi.to_csv(FEATURE_IMPORTANCES_CSV, index=False)
    print("Saved feature importances:", FEATURE_IMPORTANCES_CSV)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print("Saved model to:", MODEL_PATH)

def main():
    raw_df, grouped = load_and_aggregate(INPUT_CSV)
    print(f"Aggregated to {len(grouped)} unique strategies.")
    model = train_model_on_strategy_table(grouped)
    save_outputs(model, grouped, raw_df)
    print("Done.")

if __name__ == "__main__":
    main()
