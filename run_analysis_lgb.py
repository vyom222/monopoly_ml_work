#!/usr/bin/env python3
"""
run_analysis_lgb.py

- Trains a LightGBM regressor on aggregated strategies to predict average win rate.
- Trains a LightGBM classifier on per-game rows to predict win/loss with probability calibration.
- Writes out a CSV of all distinct strategies with predicted win probabilities, ordered by prediction.
- Saves models to disk (joblib).

Expected CSV columns (per-game):
buy_brown,buy_light_blue,buy_pink,buy_orange,buy_red,buy_yellow,
buy_green,buy_indigo,buy_stations,buy_utilities,max_development_level,
unspendable_cash,roi,win,props,houses,hotels,turns,game_seed

The script ignores "roi" (as requested) and uses the boolean buy flags + max_development_level + unspendable_cash.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, classification_report, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt

# Try to use LightGBM if available
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

RANDOM_STATE = 42
DATA_FILE = "strategy_results.csv"

# Features to use (match your CSV booleans)
BUY_FLAGS = [
    "buy_brown", "buy_light_blue", "buy_pink", "buy_orange",
    "buy_red", "buy_yellow", "buy_green", "buy_indigo",
    "buy_stations", "buy_utilities"
]
OTHER_FEATURES = ["max_development_level", "unspendable_cash"]
FEATURES = BUY_FLAGS + OTHER_FEATURES

OUT_DIR = "analysis_output"
os.makedirs(OUT_DIR, exist_ok=True)


def prepare_df(path=DATA_FILE):
    print("Loading:", path)
    df = pd.read_csv(path)
    print("Raw rows:", len(df))
    # Drop rows that don't have 'win' or features
    if "win" not in df.columns:
        raise ValueError("Input CSV must have 'win' column (binary 0/1).")
    # Ensure booleans are numeric 0/1 (some CSVs may have True/False strings)
    for col in BUY_FLAGS:
        if col in df.columns:
            df[col] = df[col].astype(int)
        else:
            raise KeyError(f"Missing expected column: {col}")

    # ensure other features present
    for col in OTHER_FEATURES:
        if col not in df.columns:
            raise KeyError(f"Missing expected column: {col}")

    # Drop ROI field as requested
    if "roi" in df.columns:
        df = df.drop(columns=["roi"])

    return df


def train_regressor_on_aggregated(df):
    """
    Aggregate per unique strategy (groupby FEATURES), compute avg win,
    and train a regressor to predict avg_win.
    """
    print("\n== Aggregating strategies and training regressor ==")
    group_cols = FEATURES
    df_agg = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(avg_win=("win", "mean"), n_games=("win", "size"))
    )

    print("Unique strategies:", len(df_agg))
    X = df_agg[FEATURES].astype(float)
    y = df_agg["avg_win"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Model: LightGBM regressor if available, else use sklearn HistGradientBoostingRegressor
    if LGB_AVAILABLE:
        model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05,
                                  random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  early_stopping_rounds=50, verbose=False)
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"Regressor Test RMSE: {rmse:.4f}")
    print(f"Regressor Test R²: {r2:.4f}")

    # Feature importances (LGB has feature_importances_)
    try:
        fi = getattr(model, "feature_importances_", None)
        if fi is not None:
            importances = sorted(zip(FEATURES, fi), key=lambda x: -x[1])
            print("\nRegressor feature importances:")
            for k, v in importances:
                print(f"  {k}: {v:.4f}")
    except Exception:
        pass

    # Save model & aggregated table
    joblib.dump(model, os.path.join(OUT_DIR, "regressor_model.joblib"))
    df_agg.to_csv(os.path.join(OUT_DIR, "aggregated_strategies.csv"), index=False)
    return model, df_agg


def train_classifier_on_per_game(df):
    """
    Train a classifier to predict win/loss for individual games.
    Use class balancing and probability calibration.
    """
    print("\n== Training classifier on per-game rows ==")
    X = df[FEATURES].astype(float)
    y = df["win"].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    if LGB_AVAILABLE:
        # Use scale_pos_weight to account for class imbalance
        scale_pos_weight = max(1.0, (len(y_train) - y_train.sum()) / max(1.0, y_train.sum()))
        clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05,
                                 random_state=RANDOM_STATE, n_jobs=-1,
                                 class_weight="balanced")
        # Fit with early stopping
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                early_stopping_rounds=50, verbose=False)
    else:
        clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)

    # Calibrate probabilities (sigmoid is a good default)
    calib = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
    calib.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = calib.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    print(f"Classifier accuracy: {acc:.4f}")
    print(f"Classifier ROC AUC: {roc:.4f}")
    print(f"Classifier Brier score: {brier:.6f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Feature importances from the underlying estimator if available
    base = calib.base_estimator if hasattr(calib, "base_estimator") else None
    if base is None and hasattr(clf, 'booster_'):
        base = clf
    try:
        fi = getattr(clf, "feature_importances_", None)
        if fi is not None:
            importances = sorted(zip(FEATURES, fi), key=lambda x: -x[1])
            print("\nClassifier feature importances:")
            for k, v in importances:
                print(f"  {k}: {v:.4f}")
    except Exception:
        pass

    # Save calibrated classifier
    joblib.dump(calib, os.path.join(OUT_DIR, "classifier_calibrated.joblib"))

    return calib


def predict_all_strategies_and_save(reg_model, clf_calibrated, df):
    """
    For every distinct strategy (unique combination of FEATURES),
    compute predicted avg win (from regressor) and predicted win prob (from calibrated classifier by averaging
    predicted per-game probabilities for that strategy).
    Write ordered CSVs.
    """
    print("\n== Predicting for all distinct strategies ==")
    # Unique strategy combos from data (if you'd like to explore whole grid you can generate cartesian product)
    unique_strats = df[FEATURES].drop_duplicates().reset_index(drop=True)
    unique_strats = unique_strats.sort_values(by=FEATURES).reset_index(drop=True)

    Xu = unique_strats[FEATURES].astype(float)

    # Regressor predicted avg win
    reg_pred = reg_model.predict(Xu)

    # For classifier -> average predicted per-game prob (we use classifier to predict probability for "a single game")
    clf_proba = None
    try:
        clf_proba = clf_calibrated.predict_proba(Xu)[:, 1]
    except Exception:
        # if calibration object cannot predict on aggregated rows, fall back to base estimator
        try:
            base = clf_calibrated.base_estimator
            clf_proba = base.predict_proba(Xu)[:, 1]
        except Exception:
            clf_proba = np.zeros(len(Xu))

    unique_strats["pred_avg_win_regressor"] = reg_pred
    unique_strats["pred_game_win_prob_classifier"] = clf_proba

    # Save ordered CSVs
    unique_strats_sorted = unique_strats.sort_values(
        "pred_game_win_prob_classifier", ascending=False
    ).reset_index(drop=True)
    unique_strats_sorted.to_csv(os.path.join(OUT_DIR, "strategies_predicted_winrate.csv"), index=False)
    print("Wrote strategies_predicted_winrate.csv with", len(unique_strats_sorted), "rows.")


def main():
    df = prepare_df(DATA_FILE)

    # 1) Regressor on aggregated strategies
    reg_model, df_agg = train_regressor_on_aggregated(df)

    # 2) Classifier per-game
    clf_calib = train_classifier_on_per_game(df)

    # 3) Predict all unique strategies and save
    predict_all_strategies_and_save(reg_model, clf_calib, df)

    print("\nAll done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
