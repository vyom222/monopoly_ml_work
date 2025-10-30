# two_stage_lgb.py (fixed: removed verbose arg in .fit())
import os
import sys
import math
import pickle
from collections import defaultdict
import inspect

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, brier_score_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from math import sqrt

# Try to import LightGBM
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except Exception as e:
    print("Failed to import LightGBM. If you see a libomp / dylib error on mac, run:")
    print("  brew install libomp")
    print("then reinstall lightgbm in the same python environment.")
    raise

# ---------- User config ----------
CSV_PATH = "strategy_results.csv"
OUT_DIR = "analysis_output"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5

# Groups we expect in the game (these should match the 'max_owned_<group>' columns in your CSV)
GROUP_PREFIX = "max_owned_"  # we'll detect columns with this prefix

# The buy flags (these should match your CSV header)
BUY_FLAGS = [
    "buy_brown","buy_light_blue","buy_pink","buy_orange","buy_red",
    "buy_yellow","buy_green","buy_indigo","buy_stations","buy_utilities"
]

# Pre-game numeric features to include if present
PRENUM = ["max_development_level", "unspendable_cash", "trade_max_diff_absolute", "set_completion_trade_bonus"]

# Stage1 model settings
STAGE1_PARAMS = dict(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)

# Stage2 model settings
STAGE2_PARAMS = dict(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)

# ---------- Helpers ----------
def safe_bool_to_int(series):
    """Map typical boolean-like CSV values to integer 0/1."""
    s = series.copy().astype(str).str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    mapped = s.map(mapping)
    if mapped.isna().any():
        coerced = pd.to_numeric(series, errors="coerce")
        mapped = mapped.fillna(coerced)
    return mapped.fillna(0).astype(int)

def wilson_lower_bound(k, n, z=1.96):
    """Wilson score lower bound for binomial proportion (conservative lower bound)."""
    if n == 0:
        return 0.0
    phat = k / n
    z2 = z*z
    denom = 1 + z2/n
    numer = phat + z2/(2*n) - z * math.sqrt((phat*(1-phat) + z2/(4*n))/n)
    return numer / denom

# ---------- Load data ----------
print("Loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print("Raw rows:", len(df))

# Ensure required target exists
if "win" not in df.columns:
    raise RuntimeError("CSV is missing 'win' column")

# Detect the max_owned_* columns automatically
max_owned_cols = [c for c in df.columns if c.startswith(GROUP_PREFIX)]
if not max_owned_cols:
    raise RuntimeError(f"No columns found with prefix '{GROUP_PREFIX}'. Add max_owned_* columns to CSV or adjust GROUP_PREFIX.")

print("Detected target groups (max_owned):", max_owned_cols)

# Coerce buy flags to 0/1 integers
for col in BUY_FLAGS:
    if col in df.columns:
        df[col] = safe_bool_to_int(df[col])
    else:
        print(f"Warning: expected buy flag column '{col}' not found in CSV; skipping it.")

# Coerce pre-numeric features
present_prenum = [c for c in PRENUM if c in df.columns]
print("Using pre-game numeric features:", present_prenum)

# Convert win to int (some CSVs might contain floats)
df['win'] = df['win'].astype(int)

# Build list of pre-game features we will use for Stage1
pregame_features = [c for c in BUY_FLAGS if c in df.columns] + present_prenum
print("Pregame features used for Stage1 & Stage2:", pregame_features)

# ---------- Stage 1: predict max_owned_<group> from pre-game features (OOF) ----------
print("\n== Stage 1: predicting max_owned_* targets (OOF predictions) ==")
# --- Replace the Stage1 block in two_stage_lgb.py with the following ---

# Build DataFrame X (keep as DataFrame to keep feature names consistent)
X = df[pregame_features].fillna(0)

stage1_oof = pd.DataFrame(index=df.index)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

stage1_rmse = {}
for target_col in max_owned_cols:
    y = df[target_col].fillna(0).astype(float).values
    oof = np.zeros(len(df), dtype=float)
    fold = 0
    print(f" Stage1 target: {target_col}")
    for train_idx, val_idx in kf.split(X, y):
        fold += 1
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = LGBMRegressor(**STAGE1_PARAMS)
        # fit with DataFrames so LightGBM sees feature names consistently
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        oof[val_idx] = model.predict(X_val)
    # compute RMSE in a portable way (avoid squared= kwarg)
    rmse = math.sqrt(mean_squared_error(y, oof))
    stage1_rmse[target_col] = rmse
    colname = "pred_" + target_col
    stage1_oof[colname] = oof
    print(f"  OOF RMSE: {rmse:.6f}")

# Attach Stage1 OOF prediction columns to df (these are safe to use in Stage2 training)
df = pd.concat([df, stage1_oof], axis=1)
stage1_oof.to_csv(os.path.join(OUT_DIR, "stage1_oof_preds.csv"), index=False)
print("Saved Stage1 OOF predictions ->", os.path.join(OUT_DIR, "stage1_oof_preds.csv"))

# ---------- Stage 2: train classifier on per-game rows using pre-game features + Stage1 preds ----------
print("\n== Stage 2: training classifier (win) using pregame + Stage1-predicted max_owned == ")

stage1_pred_cols = ["pred_" + c for c in max_owned_cols]
stage2_features = pregame_features + [c for c in stage1_pred_cols if c in df.columns]
X2 = df[stage2_features].fillna(0)
y2 = df['win']

print("Stage2 features:", stage2_features)

# Train/holdout split for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.20, random_state=RANDOM_STATE, stratify=y2)

base_clf = LGBMClassifier(**STAGE2_PARAMS)
print("Training Stage2 base classifier (LGBM)...")
# removed verbose from fit call (not needed)
base_clf.fit(X_train, y_train)

print("Calibrating probabilities with CalibratedClassifierCV (sigmoid)...")
sig = inspect.signature(CalibratedClassifierCV)
if "estimator" in sig.parameters:
    clf_kw = "estimator"
elif "base_estimator" in sig.parameters:
    clf_kw = "base_estimator"
else:
    raise RuntimeError("Unexpected CalibratedClassifierCV signature; cannot find 'estimator' nor 'base_estimator' parameter")

calibrated = CalibratedClassifierCV(**{clf_kw: base_clf, "method": "sigmoid", "cv": 3})
calibrated.fit(X_train, y_train)

y_prob = calibrated.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

print(f"Stage2 Test Accuracy: {acc:.4f}")
print(f"Stage2 Test ROC AUC : {roc:.4f}")
print(f"Stage2 Test Brier   : {brier:.6f}")

# ---------- Retrain Stage1 and Stage2 on FULL data for final predictions ----------
print("\n== Retraining Stage1 & Stage2 on FULL data for final predictions ==")
X_full = df[pregame_features].fillna(0).values
stage1_full_preds = {}
for target_col in max_owned_cols:
    y_full = df[target_col].fillna(0).astype(float).values
    model = LGBMRegressor(**STAGE1_PARAMS)
    model.fit(X_full, y_full)
    pred = model.predict(X_full)
    stage1_full_preds["pred_" + target_col] = pred

# Add full-data Stage1 preds to df
for k, v in stage1_full_preds.items():
    df[k] = v

# Retrain Stage2 on full data
X2_full = df[stage2_features].fillna(0)
y2_full = df['win'].values
final_clf = LGBMClassifier(**STAGE2_PARAMS)
final_clf.fit(X2_full, y2_full)


sig = inspect.signature(CalibratedClassifierCV)
if "estimator" in sig.parameters:
    clf_kw = "estimator"
elif "base_estimator" in sig.parameters:
    clf_kw = "base_estimator"
else:
    raise RuntimeError("Unexpected CalibratedClassifierCV signature; cannot find 'estimator' nor 'base_estimator' parameter")

final_calib = CalibratedClassifierCV(**{clf_kw: final_clf, "method": "sigmoid", "cv": 3})
final_calib.fit(X2_full, y2_full)

df["pred_win_prob"] = final_calib.predict_proba(X2_full)[:, 1]

# ---------- Aggregate per unique strategy (group by strategy fields) ----------
strategy_keys = [c for c in BUY_FLAGS if c in df.columns] + [c for c in present_prenum]
print("Aggregating predictions per unique strategy with keys:", strategy_keys)

grouped = df.groupby(strategy_keys).agg(
    games_played=("win", "count"),
    wins=("win", "sum"),
    empirical_win_rate=("win", "mean"),
    avg_pred_win_prob=("pred_win_prob", "mean")
).reset_index()

grouped["wilson_lower"] = grouped.apply(lambda r: wilson_lower_bound(int(r["wins"]), int(r["games_played"])), axis=1)

grouped_sorted_by_pred = grouped.sort_values("avg_pred_win_prob", ascending=False)
out_pred_path = os.path.join(OUT_DIR, "strategies_predicted_winrate_by_pred.csv")
grouped_sorted_by_pred.to_csv(out_pred_path, index=False)
print("Wrote predicted-ranked strategies ->", out_pred_path)

grouped_sorted_by_wilson = grouped.sort_values("wilson_lower", ascending=False)
out_wilson_path = os.path.join(OUT_DIR, "strategies_ranked_by_wilson.csv")
grouped_sorted_by_wilson.to_csv(out_wilson_path, index=False)
print("Wrote empirically-ranked strategies (Wilson lower bound) ->", out_wilson_path)

# Diagnostics summary
print("\n== Diagnostics summary ==")
print("Stage1 RMSE per max_owned target:")
for k, v in stage1_rmse.items():
    print(f"  {k}: RMSE = {v:.4f}")

print(f"\nStage2 holdout (test) metrics (from earlier eval):")
print(f"  Accuracy: {acc:.4f}, ROC AUC: {roc:.4f}, Brier: {brier:.6f}")

print("\nTop 10 strategies by predicted win prob:")
print(grouped_sorted_by_pred.head(10).to_string(index=False))

print("\nTop 10 strategies by Wilson lower bound (empirical conservative):")
print(grouped_sorted_by_wilson.head(10).to_string(index=False))

df_out_path = os.path.join(OUT_DIR, "per_game_with_preds.csv")
df.to_csv(df_out_path, index=False)
print("\nSaved per-game rows with predictions ->", df_out_path)

print("\nAll done. Outputs in:", OUT_DIR)
