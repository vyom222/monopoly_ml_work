"""
Evaluation & plotting script for Monopoly-win models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_recall_curve,
                             average_precision_score, log_loss, brier_score_loss,
                             roc_curve, confusion_matrix)
from sklearn.calibration import calibration_curve

from lightgbm import LGBMClassifier
import shap

# Load data
CSV = "strategy_results.csv"
df = pd.read_csv(CSV)

y_col = "did_win" if "did_win" in df.columns else "win"
sets = ["brown","lightblue","pink","orange","red","yellow","green","indigo","railroads","utilities"]
needed_map = {s: (4 if s=="railroads" else 2 if s in ("utilities","indigo","brown") else 3) for s in sets}

# Compute set completion and conditional win probabilities
results = []
for s in sets:
    max_col = f"max_owned_{s}"
    if max_col not in df.columns:
        continue
    needed = needed_map[s]
    completed = (df[max_col] == needed)
    p_complete = completed.mean()
    p_win_given_complete = df.loc[completed, y_col].mean() if completed.any() else np.nan
    p_win_given_not = df.loc[~completed, y_col].mean() if (~completed).any() else np.nan
    results.append({"Set": s,
                    "P(complete)": float(p_complete),
                    "P(win | complete)": float(p_win_given_complete) if not pd.isna(p_win_given_complete) else np.nan,
                    "P(win | not complete)": float(p_win_given_not) if not pd.isna(p_win_given_not) else np.nan,
                    "abs_win_gain": float(p_win_given_complete - p_win_given_not) if (not pd.isna(p_win_given_complete) and not pd.isna(p_win_given_not)) else np.nan})
prob_table = pd.DataFrame(results).set_index("Set")
prob_table.to_csv("set_completion_win_table.csv")

# Prepare features
X = pd.DataFrame({s: (df[f"max_owned_{s}"] == needed_map[s]).astype(int) if f"max_owned_{s}" in df.columns else 0
                  for s in sets})
y = df[y_col].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
logreg = LogisticRegression(max_iter=2000, solver="lbfgs")
logreg.fit(X_train, y_train)
probs_log = logreg.predict_proba(X_test)[:, 1]
preds_log = (probs_log >= 0.5).astype(int)

metrics_log = {
    "accuracy": accuracy_score(y_test, preds_log),
    "roc_auc": roc_auc_score(y_test, probs_log),
    "avg_precision": average_precision_score(y_test, probs_log),
    "logloss": log_loss(y_test, probs_log),
    "brier": brier_score_loss(y_test, probs_log)
}

pd.DataFrame({"model": ["logistic"], **metrics_log}).to_csv("model_metrics_summary.csv", index=False)

# LightGBM + SHAP
lgb = LGBMClassifier(n_estimators=200, n_jobs=-1, random_state=42)
lgb.fit(X_train, y_train)
probs_lgb = lgb.predict_proba(X_test)[:, 1]
preds_lgb = (probs_lgb >= 0.5).astype(int)

metrics_lgb = {
    "model": "lightgbm",
    "accuracy": accuracy_score(y_test, preds_lgb),
    "roc_auc": roc_auc_score(y_test, probs_lgb),
    "avg_precision": average_precision_score(y_test, probs_lgb),
    "logloss": log_loss(y_test, probs_lgb),
    "brier": brier_score_loss(y_test, probs_lgb)
}

dfm = pd.read_csv("model_metrics_summary.csv")
dfm = pd.concat([dfm, pd.DataFrame([metrics_lgb])], ignore_index=True)
dfm.to_csv("model_metrics_summary.csv", index=False)

explainer = shap.TreeExplainer(lgb)
shap_vals = explainer.shap_values(X_test)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
shap_df = pd.DataFrame({"Set": X.columns, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)
shap_df.to_csv("lgb_shap_feature_importance.csv", index=False)

# Plots
PNG_DIR = "diagnostic_plots"
os.makedirs(PNG_DIR, exist_ok=True)

def save_fig(fig, name):
    fig.savefig(os.path.join(PNG_DIR, name), bbox_inches="tight", dpi=150)
    plt.close(fig)

# ROC Curve
fig, ax = plt.subplots(figsize=(7,6))
fpr, tpr, _ = roc_curve(y_test, probs_log)
ax.plot(fpr, tpr, label=f"Logistic (AUC={metrics_log['roc_auc']:.3f})")
fpr2, tpr2, _ = roc_curve(y_test, probs_lgb)
ax.plot(fpr2, tpr2, label=f"LGBM (AUC={metrics_lgb['roc_auc']:.3f})")
ax.plot([0,1],[0,1],"k--", alpha=0.3)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
save_fig(fig, "roc_curve.png")

# Precision-Recall Curve
fig, ax = plt.subplots(figsize=(7,6))
prec, rec, _ = precision_recall_curve(y_test, probs_log)
ax.plot(rec, prec, label=f"Logistic (AP={metrics_log['avg_precision']:.3f})")
prec2, rec2, _ = precision_recall_curve(y_test, probs_lgb)
ax.plot(rec2, prec2, label=f"LGBM (AP={metrics_lgb['avg_precision']:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend()
save_fig(fig, "pr_curve.png")

# Calibration Curve
fig, ax = plt.subplots(figsize=(7,6))
pt, pp = calibration_curve(y_test, probs_log, n_bins=10)
ax.plot(pp, pt, marker='o', label="Logistic")
pt2, pp2 = calibration_curve(y_test, probs_lgb, n_bins=10)
ax.plot(pp2, pt2, marker='o', label="LGBM")
ax.plot([0,1],[0,1],"k--", alpha=0.3)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve")
ax.legend()
save_fig(fig, "calibration_curve.png")

# Predicted probability distributions
fig, ax = plt.subplots(figsize=(7,6))
bins = np.linspace(0,1,50)
ax.hist(probs_log[y_test==1], bins=bins, alpha=0.6, label="Winners (logistic)", density=True)
ax.hist(probs_log[y_test==0], bins=bins, alpha=0.4, label="Non-winners (logistic)", density=True)
ax.hist(probs_lgb[y_test==1], bins=bins, alpha=0.4, label="Winners (LGBM)", density=True, histtype='step')
ax.hist(probs_lgb[y_test==0], bins=bins, alpha=0.3, label="Non-winners (LGBM)", density=True, histtype='step')
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Density")
ax.set_title("Predicted Probabilities Distribution")
ax.legend()
save_fig(fig, "predicted_prob_distributions.png")

# Confusion matrices
def plot_confusion(y_true, probs, thr, title, fname):
    preds = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    cm = np.array([[tn, fp],[fn, tp]])
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center", color="k")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["True 0","True 1"])
    ax.set_title(f"{title}\nthreshold={thr}")
    save_fig(fig, fname)

for thr in [0.5, 0.1, 0.05, 0.02]:
    plot_confusion(y_test, probs_log, thr, "Logistic Confusion Matrix", f"confusion_logistic_thr_{thr:.3f}.png")
    plot_confusion(y_test, probs_lgb, thr, "LGBM Confusion Matrix", f"confusion_lgb_thr_{thr:.3f}.png")

# Logistic coefficients + bootstrap CI
coefs = logreg.coef_[0]
odds = np.exp(coefs)
coef_df = pd.DataFrame({"Set": X.columns, "coef": coefs, "odds_ratio": odds})

n_boot = 500
boot_coefs = np.zeros((n_boot, len(X.columns)))
rng = np.random.RandomState(123)
for i in range(n_boot):
    idx = rng.choice(len(X_train), size=len(X_train), replace=True)
    Xb, yb = X_train.iloc[idx], y_train.iloc[idx]
    modelb = LogisticRegression(max_iter=2000, solver="lbfgs").fit(Xb, yb)
    boot_coefs[i, :] = modelb.coef_[0]

ci_lower = np.percentile(np.exp(boot_coefs), 2.5, axis=0)
ci_upper = np.percentile(np.exp(boot_coefs), 97.5, axis=0)
coef_df["odds_ci_lower"] = ci_lower
coef_df["odds_ci_upper"] = ci_upper
coef_df.to_csv("logistic_odds_and_ci.csv", index=False)

fig, ax = plt.subplots(figsize=(8,6))
ypos = np.arange(len(coef_df))
ax.errorbar(coef_df["odds_ratio"], ypos,
            xerr=[np.abs(coef_df["odds_ratio"]-coef_df["odds_ci_lower"]),
                  np.abs(coef_df["odds_ci_upper"]-coef_df["odds_ratio"])], fmt='o')
ax.set_yticks(ypos); ax.set_yticklabels(coef_df["Set"])
ax.set_xscale("log")
ax.set_xlabel("Odds ratio (log scale)")
ax.set_title("Logistic regression odds ratios with 95% CI")
save_fig(fig, "odds_ratios_ci.png")

# Strategy bubble chart
buy_col_map = {
    "brown": "buy_brown",
    "lightblue": "buy_light_blue",
    "pink": "buy_pink",
    "orange": "buy_orange",
    "red": "buy_red",
    "yellow": "buy_yellow",
    "green": "buy_green",
    "indigo": "buy_indigo",
    "railroads": "buy_stations",
    "utilities": "buy_utilities"
}

results_cond = []

for s in sets:
    max_col = f"max_owned_{s}"
    buy_col = buy_col_map[s]

    if max_col not in df.columns or buy_col not in df.columns:
        print(f"Skipping {s} because columns missing")
        continue

    needed = needed_map[s]
    df_intended = df[df[buy_col] == 1]

    completed = (df_intended[max_col] == needed)
    cond_complete_prob = completed.mean() if len(df_intended) > 0 else 0
    p_win_given_complete = df_intended.loc[completed, y_col].mean() if completed.any() else 0

    importance = np.abs(coefs[X.columns.get_loc(s)]) if s in X.columns else 0.01

    results_cond.append({
        "Set": s,
        "cond_complete_prob": cond_complete_prob,
        "P(win | complete)": p_win_given_complete,
        "importance": importance
    })

merge_df_cond = pd.DataFrame(results_cond)

# Bubble plot
fig, ax = plt.subplots(figsize=(10,7))
sizes = (merge_df_cond["importance"] / merge_df_cond["importance"].max()) * 800
sc = ax.scatter(merge_df_cond["cond_complete_prob"],
                merge_df_cond["P(win | complete)"],
                s=sizes, alpha=0.6, edgecolors='k')

for i, row in merge_df_cond.iterrows():
    ax.text(row["cond_complete_prob"] + 0.002,
            row["P(win | complete)"] + 0.002,
            row["Set"], fontsize=9)

ax.set_xlabel("Conditional Probability of Completing Set (given intention)")
ax.set_ylabel("Probability of Winning | Completed Set")
ax.set_title("Conditional Completion vs Win Probability (bubble size ~ importance)")
save_fig(fig, "strategy_bubble_chart_conditional.png")

# SHAP Beeswarm plot
explainer = shap.TreeExplainer(lgb)
shap_values = explainer.shap_values(X)

fig = plt.figure(figsize=(10,7))
shap.summary_plot(shap_values, X, plot_type="dot", max_display=10)
plt.tight_layout()
save_fig(fig, "beeswarm_shap.png")
plt.close(fig)

# SHAP Interaction Heatmap (sampled)
sample_size = min(2000, len(X))
X_sample = X.sample(n=sample_size, random_state=42)

explainer = shap.TreeExplainer(lgb)
shap_interaction_values = explainer.shap_interaction_values(X_sample)

if isinstance(shap_interaction_values, list):
    shap_interaction_values = shap_interaction_values[1]

mean_interaction = np.abs(shap_interaction_values).mean(axis=0)
interaction_df = pd.DataFrame(mean_interaction, index=X.columns, columns=X.columns)

fig, ax = plt.subplots(figsize=(10,8))
im = ax.imshow(interaction_df, cmap='coolwarm', interpolation='nearest')
ax.set_xticks(np.arange(len(X.columns)))
ax.set_yticks(np.arange(len(X.columns)))
ax.set_xticklabels(X.columns, rotation=45, ha='right')
ax.set_yticklabels(X.columns)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Mean absolute SHAP interaction", rotation=-90, va="bottom")
ax.set_title("SHAP Interaction Heatmap: Monopoly Sets (sampled)")
plt.tight_layout()
save_fig(fig, "shap_interaction_heatmap_sampled.png")
plt.close(fig)
