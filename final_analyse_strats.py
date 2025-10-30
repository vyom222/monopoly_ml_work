

# ===== File: analyze_strategies.py =====
"""
Analysis & reporting for trained two-stage Monopoly model.
Loads models from final_work/ and produces:
- Strategy rankings CSV (all strategies by predicted win rate)
- Per-set completion/win table
- Evaluation plots for Stage 2 (ROC/PR/Calibration/Confusion/Distributions)
- Optional SHAP using uncalibrated Stage 2 LGBM

Outputs are written under final_work/ and final_work/plots/.
"""

import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

RANDOM_STATE = 42
DATA_FILE = "strategy_results.csv"
OUT_DIR = "final_work"
PLOTS = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS, exist_ok=True)

# Same feature spec as training
PREGAME_FEATURES = [
    "buy_brown", "buy_light_blue", "buy_pink", "buy_orange",
    "buy_red", "buy_yellow", "buy_green", "buy_indigo",
    "buy_stations", "buy_utilities",
    "max_development_level", "unspendable_cash",
]
SETS = ["brown","lightblue","pink","orange","red","yellow","green","indigo","railroads","utilities"]
NEEDED = {s: (4 if s=="railroads" else 2 if s in ("utilities","indigo","brown") else 3) for s in SETS}


def load_models():
    stage1 = joblib.load(os.path.join(OUT_DIR, "stage1_models.pkl"))
    stage2_lgbm = joblib.load(os.path.join(OUT_DIR, "stage2_lgbm.pkl"))
    stage2_cal = joblib.load(os.path.join(OUT_DIR, "stage2_calibrated.pkl"))
    return stage1, stage2_lgbm, stage2_cal


def stage1_predict_proba(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe of per-set completion probabilities for rows in X."""
    preds = {}
    for target, model in models.items():
        preds[f"p_{target}"] = model.predict_proba(X)[:, 1]
    return pd.DataFrame(preds, index=X.index)


def stage2_predict_proba(stage2_cal, X2: pd.DataFrame) -> np.ndarray:
    return stage2_cal.predict_proba(X2)[:, 1]


def evaluate_stage2_on_holdout(df, stage1_models, stage2_cal):
    # Build y
    y = (df["win"] == 1).astype(int)

    # Train/test split
    X = df[PREGAME_FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    # Predict Stage 1 on both splits (using trained full models)
    p1_tr = stage1_predict_proba(stage1_models, Xtr)
    p1_te = stage1_predict_proba(stage1_models, Xte)

    # Stage 2 predictions (calibrated)
    X2_tr = pd.concat([Xtr, p1_tr], axis=1)
    X2_te = pd.concat([Xte, p1_te], axis=1)

    probs = stage2_predict_proba(stage2_cal, X2_te)
    preds = (probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(yte, preds)
    roc = roc_auc_score(yte, probs)
    ll = log_loss(yte, probs)
    brier = brier_score_loss(yte, probs)
    ap = average_precision_score(yte, probs)

    # Save / print summary
    summ = pd.DataFrame({
        "model": ["stage2_calibrated"],
        "accuracy": [acc],
        "roc_auc": [roc],
        "avg_precision": [ap],
        "logloss": [ll],
        "brier": [brier],
    })
    summ.to_csv(os.path.join(OUT_DIR, "model_metrics_summary_holdout.csv"), index=False)

    # Plots
    def save_fig(fig, name):
        path = os.path.join(PLOTS, name)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print("Saved:", path)

    # ROC
    fpr, tpr, _ = roc_curve(yte, probs)
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(fpr, tpr, label=f"Stage2 Cal (AUC={roc:.3f})")
    ax.plot([0,1],[0,1],"k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve"); ax.legend()
    save_fig(fig, "roc_curve_stage2.png"); plt.close(fig)

    # PR
    prec, rec, _ = precision_recall_curve(yte, probs)
    apv = average_precision_score(yte, probs)
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(rec, prec, label=f"Stage2 Cal (AP={apv:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall Curve"); ax.legend()
    save_fig(fig, "pr_curve_stage2.png"); plt.close(fig)

    # Calibration
    pt, pp = calibration_curve(yte, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(pp, pt, marker='o', label="Stage2 Cal")
    ax.plot([0,1],[0,1],"k--", alpha=0.3)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives"); ax.set_title("Calibration curve"); ax.legend()
    save_fig(fig, "calibration_curve_stage2.png"); plt.close(fig)

    # Probability distributions
    bins = np.linspace(0,1,50)
    fig, ax = plt.subplots(figsize=(7,6))
    ax.hist(probs[yte==1], bins=bins, alpha=0.6, label="Winners", density=True)
    ax.hist(probs[yte==0], bins=bins, alpha=0.4, label="Non-winners", density=True)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Density"); ax.set_title("Predicted win probability distribution"); ax.legend()
    save_fig(fig, "predicted_prob_distributions_stage2.png"); plt.close(fig)

    # Confusion matrices at thresholds
    def plot_confusion(y_true, probs, thr, title, fname):
        preds = (probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        fig, ax = plt.subplots(figsize=(5,4))
        cm = np.array([[tn, fp],[fn, tp]])
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f"{val}", ha="center", va="center", color="k")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["True 0","True 1"])
        ax.set_title(f"{title}\nthreshold={thr}")
        save_fig(fig, fname); plt.close(fig)

    for thr in [0.5, 0.2, 0.1, 0.05]:
        plot_confusion(yte, probs, thr, "Stage2 Confusion Matrix", f"confusion_stage2_thr_{thr:.2f}.png")

    # Save predictions snapshot
    snap = X2_te.copy()
    snap["y_true"] = yte.values
    snap["p_stage2"] = probs
    snap.to_csv(os.path.join(OUT_DIR, "predictions_with_features.csv"))

    return {
        "accuracy": acc, "roc_auc": roc, "avg_precision": apv, "logloss": ll, "brier": brier
    }


def compute_set_completion_win_table(df):
    results = []
    y = (df["win"]==1).astype(int)
    for s in SETS:
        col = f"max_owned_{s}"
        if col not in df.columns:
            continue
        need = NEEDED[s]
        comp = (df[col]==need)
        p_complete = comp.mean()
        p_win_comp = y[comp].mean() if comp.any() else np.nan
        p_win_not = y[~comp].mean() if (~comp).any() else np.nan
        diff = (p_win_comp - p_win_not) if (p_win_comp==p_win_comp and p_win_not==p_win_not) else np.nan
        results.append({"Set": s, "P(complete)": p_complete, "P(win|complete)": p_win_comp, "P(win|not)": p_win_not, "abs_win_gain": diff})
    tab = pd.DataFrame(results).set_index("Set").sort_values("abs_win_gain", ascending=False)
    tab.to_csv(os.path.join(OUT_DIR, "set_completion_win_table.csv"))
    return tab


def generate_strategy_grid(df, full_grid=True):
    """Return a DataFrame of strategies to score.
    full_grid=True -> all 2^10 buy-flag combos; set max_development_level & unspendable_cash to dataset medians.
    full_grid=False -> use unique combinations observed in data.
    """
    if full_grid:
        buys = [c for c in PREGAME_FEATURES if c.startswith("buy_")]
        combos = list(product([0,1], repeat=len(buys)))
        base = {"max_development_level": int(df["max_development_level"].median()),
                "unspendable_cash": float(df["unspendable_cash"].median())}
        rows = []
        for comb in combos:
            row = dict(zip(buys, comb))
            row.update(base)
            rows.append(row)
        grid = pd.DataFrame(rows)
    else:
        grid = df[PREGAME_FEATURES].drop_duplicates().reset_index(drop=True)
    return grid[PREGAME_FEATURES]


def rank_strategies(strategy_df, stage1_models, stage2_cal):
    # Stage 1 -> Stage 2
    p1 = stage1_predict_proba(stage1_models, strategy_df)
    X2 = pd.concat([strategy_df, p1], axis=1)
    pwin = stage2_predict_proba(stage2_cal, X2)

    out = strategy_df.copy()
    out["predicted_win_prob"] = pwin

    # Provide human-readable list of purchased groups
    buy_cols = [c for c in strategy_df.columns if c.startswith("buy_")]
    def owned_sets(row):
        return ", ".join([c.replace("buy_","") for c in buy_cols if row[c]==1]) or "(none)"

    out["owned_sets"] = out.apply(owned_sets, axis=1)
    out = out.sort_values("predicted_win_prob", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out)+1)

    # Points: 1 for best .. N for worst
    out["points"] = out["rank"]

    out.to_csv(os.path.join(OUT_DIR, "strategy_rankings.csv"), index=False)

    # Also save top 20 nicely
    out.head(20).to_csv(os.path.join(OUT_DIR, "top20_strategies.csv"), index=False)
    return out


def shap_stage2(stage2_lgbm, df):
    if not SHAP_AVAILABLE:
        print("SHAP not available; skipping")
        return
    X = df[PREGAME_FEATURES]
    p1 = stage1_predict_proba(joblib.load(os.path.join(OUT_DIR, "stage1_models.pkl")), X)
    X2 = pd.concat([X, p1], axis=1)
    explainer = shap.TreeExplainer(stage2_lgbm)
    sv = explainer.shap_values(X2)
    if isinstance(sv, list):
        sv = sv[1]
    shap_df = pd.DataFrame({"feature": X2.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(os.path.join(OUT_DIR, "stage2_shap_importance.csv"), index=False)
    plt.figure(figsize=(10,6))
    shap.summary_plot(sv, X2, plot_type="bar", show=False)
    plt.title("Stage 2 Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "shap_bar_stage2.png"))
    plt.close()


def main():
    # Load data & models
    assert os.path.exists(DATA_FILE), f"{DATA_FILE} not found"
    df = pd.read_csv(DATA_FILE)
    stage1_models, stage2_lgbm, stage2_cal = load_models()

    # Evaluation on holdout split
    _ = evaluate_stage2_on_holdout(df, stage1_models, stage2_cal)

    # Per-set table (descriptive)
    tab = compute_set_completion_win_table(df)

    # Strategy ranking (full grid of 2^10 buy choices)
    strat_df = generate_strategy_grid(df, full_grid=True)
    rankings = rank_strategies(strat_df, stage1_models, stage2_cal)

    # Optional SHAP using uncalibrated LGBM (more stable with TreeExplainer)
    shap_stage2(stage2_lgbm, df)

    print("\nAll artifacts saved under:", OUT_DIR)
    print(" - Plots in:", PLOTS)
    print(" - Strategy rankings:", os.path.join(OUT_DIR, "strategy_rankings.csv"))
    print(" - Top 20:", os.path.join(OUT_DIR, "top20_strategies.csv"))
    print(" - Set table:", os.path.join(OUT_DIR, "set_completion_win_table.csv"))


if __name__ == "__main__":
    main()
