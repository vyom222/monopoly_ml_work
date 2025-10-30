# ===== File: train_models.py =====
"""
Train Stage 1 + Stage 2 LightGBM models and persist them for analysis.
- Stage 1: one binary LGBMClassifier per set -> P(complete set)
- Stage 2: LGBMClassifier (tree) + sigmoid calibration -> P(win)

Outputs (created under final_work/):
- stage1_models.pkl                 (dict of per-set LGBM models)
- stage1_targets.json               (metadata: set names, target column names)
- stage2_lgbm.pkl                   (uncalibrated LGBM model used for SHAP)
- stage2_calibrated.pkl             (CalibratedClassifierCV)
- features.json                     (pregame feature names)
- model_metrics_summary.csv         (OOF CV metrics for Stage 2)
- two_stage_predictions.csv         (OOF predictions for each row)

This script keeps your previous v3 structure but adds model persistence.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

RANDOM_STATE = 42
DATA_FILE = "strategy_results.csv"
OUT_DIR = "final_work"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Stage 1 target spec ---
SET_SIZES = {
    "brown": 2,
    "lightblue": 3,
    "pink": 3,
    "orange": 3,
    "red": 3,
    "yellow": 3,
    "green": 3,
    "indigo": 2,
    "railroads": 4,
    "utilities": 2,
}
STAGE1_TARGETS = {f"max_owned_{k}": v for k, v in SET_SIZES.items()}

PREGAME_FEATURES = [
    "buy_brown", "buy_light_blue", "buy_pink", "buy_orange",
    "buy_red", "buy_yellow", "buy_green", "buy_indigo",
    "buy_stations", "buy_utilities",
    "max_development_level", "unspendable_cash",
]


def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    # Ensure bools -> ints
    for col in [c for c in PREGAME_FEATURES if c.startswith("buy_")]:
        if col not in df.columns:
            raise KeyError(f"Missing expected column: {col}")
        df[col] = df[col].astype(int)
    for c in ["max_development_level", "unspendable_cash"]:
        if c not in df.columns:
            raise KeyError(f"Missing expected column: {c}")
    if "win" not in df.columns:
        raise KeyError("CSV must contain 'win' column (0/1)")

    # Build Stage 1 binary targets
    for col, need in STAGE1_TARGETS.items():
        if col not in df.columns:
            raise KeyError(f"Missing column for stage1 target: {col}")
        df[f"{col}_complete"] = (df[col] == need).astype(int)

    df["did_win"] = (df["win"] == 1).astype(int)
    return df


# --- Cross-validated OOF helper ---

def cv_stage1_models(df):
    """Train one LGBM per set target and return dict of final models + OOF preds dataframe."""
    X = df[PREGAME_FEATURES]
    stage1_models = {}
    oof = pd.DataFrame(index=df.index)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for set_col, _ in STAGE1_TARGETS.items():
        target = f"{set_col}_complete"
        y = df[target].values
        oof_col = np.zeros(len(df), dtype=float)

        for tr, va in skf.split(X, y):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y[tr], y[va]
            clf = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            clf.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                eval_metric="binary_logloss",
                callbacks=[early_stopping(50), log_evaluation(0)],
            )
            oof_col[va] = clf.predict_proba(Xva)[:, 1]
        oof[f"p_{target}"] = oof_col

        # Fit final model on full data for persistence
        final_clf = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        final_clf.fit(X, y, eval_set=[(X, y)], eval_metric="binary_logloss", callbacks=[log_evaluation(0)])
        stage1_models[target] = final_clf

        acc = accuracy_score(y, (oof_col > 0.5).astype(int))
        ll = log_loss(y, oof_col)
        print(f"Stage1 {target}: Acc={acc:.3f}, LogLoss={ll:.3f}")

    return stage1_models, oof


def cv_stage2_and_fit_final(df, oof_stage1):
    """Build Stage 2 features, produce OOF predictions with CV, then fit & save final models."""
    X2 = pd.concat([df[PREGAME_FEATURES], oof_stage1], axis=1)
    y2 = df["did_win"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof2 = np.zeros(len(df), dtype=float)

    # We'll keep a reference to the last fold's uncalibrated model for SHAP during training,
    # but for persistence we will train a final model on full data after CV.
    for tr, va in skf.split(X2, y2):
        Xtr, Xva = X2.iloc[tr], X2.iloc[va]
        ytr, yva = y2[tr], y2[va]
        base = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=63,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        base.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(50), log_evaluation(0)],
        )
        calib = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, n_jobs=-1)
        calib.fit(Xtr, ytr)
        oof2[va] = calib.predict_proba(Xva)[:, 1]

    # OOF metrics
    acc = accuracy_score(y2, (oof2 > 0.5).astype(int))
    ll = log_loss(y2, oof2)
    roc = roc_auc_score(y2, oof2)
    brier = brier_score_loss(y2, oof2)
    print(f"Stage2 OOF Acc={acc:.4f} ROC AUC={roc:.4f} LogLoss={ll:.4f} Brier={brier:.6f}")

    # Fit final models on FULL data
    base_full = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    base_full.fit(X2, y2, eval_set=[(X2, y2)], eval_metric="binary_logloss", callbacks=[log_evaluation(0)])
    calib_full = CalibratedClassifierCV(estimator=base_full, method="sigmoid", cv=5, n_jobs=-1)
    calib_full.fit(X2, y2)

    # Persist
    joblib.dump(base_full, os.path.join(OUT_DIR, "stage2_lgbm.pkl"))
    joblib.dump(calib_full, os.path.join(OUT_DIR, "stage2_calibrated.pkl"))

    # Save OOF predictions
    out = df.copy()
    out = pd.concat([out, oof_stage1], axis=1)
    out["p_win_oof"] = oof2
    out.to_csv(os.path.join(OUT_DIR, "two_stage_predictions.csv"), index=False)

    # Save metrics
    pd.DataFrame({
        "metric": ["accuracy", "roc_auc", "logloss", "brier"],
        "value": [acc, roc, ll, brier],
    }).to_csv(os.path.join(OUT_DIR, "model_metrics_summary.csv"), index=False)

    return base_full, calib_full


def main():
    df = load_data(DATA_FILE)

    # Stage 1 CV + final fit
    stage1_models, oof_stage1 = cv_stage1_models(df)

    # Persist Stage 1 and metadata
    joblib.dump(stage1_models, os.path.join(OUT_DIR, "stage1_models.pkl"))
    with open(os.path.join(OUT_DIR, "stage1_targets.json"), "w") as f:
        json.dump({"targets": list(stage1_models.keys())}, f, indent=2)
    with open(os.path.join(OUT_DIR, "features.json"), "w") as f:
        json.dump({"pregame_features": PREGAME_FEATURES}, f, indent=2)

    # Stage 2 CV + final fit
    base_full, calib_full = cv_stage2_and_fit_final(df, oof_stage1)

    print("\nSaved models to:")
    print(" -", os.path.join(OUT_DIR, "stage1_models.pkl"))
    print(" -", os.path.join(OUT_DIR, "stage2_lgbm.pkl"))
    print(" -", os.path.join(OUT_DIR, "stage2_calibrated.pkl"))
    print("Metrics & OOF predictions saved in:")
    print(" -", os.path.join(OUT_DIR, "model_metrics_summary.csv"))
    print(" -", os.path.join(OUT_DIR, "two_stage_predictions.csv"))


if __name__ == "__main__":
    main()