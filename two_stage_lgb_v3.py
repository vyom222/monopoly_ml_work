import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import shap
import matplotlib.pyplot as plt

df = pd.read_csv("strategy_results.csv")

# Stage 1 targets
set_sizes = {
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
stage1_targets = {f"max_owned_{k}": v for k, v in set_sizes.items()}

pregame_features = [
    "buy_brown", "buy_light_blue", "buy_pink", "buy_orange",
    "buy_red", "buy_yellow", "buy_green", "buy_indigo",
    "buy_stations", "buy_utilities",
    "max_development_level", "unspendable_cash"
]

# Convert Stage 1 targets to binary labels
for col, full_size in stage1_targets.items():
    df[f"{col}_complete"] = (df[col] == full_size).astype(int)

# Stage 2 target
df["did_win"] = (df["win"] == 1).astype(int)

# Stage 1: Multitask LightGBM
X1 = df[pregame_features]
Y1 = df[[f"{col}_complete" for col in stage1_targets]]

oof_stage1 = pd.DataFrame(index=df.index)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for target in Y1.columns:
    y = Y1[target].values
    oof = np.zeros(len(df))
    
    for train_idx, val_idx in skf.split(X1, y):
        X_train, X_val = X1.iloc[train_idx], X1.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            n_jobs=-1,
            random_state=42
        )

        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
        )

        oof[val_idx] = clf.predict_proba(X_val)[:, 1]

    oof_stage1[f"p_{target}"] = oof
    acc = accuracy_score(y, (oof > 0.5).astype(int))
    print(f"{target}: Acc={acc:.3f}, LogLoss={log_loss(y, oof):.3f}")

# Stage 2: Predict win probability
X2 = pd.concat([X1, oof_stage1], axis=1)
y2 = df["did_win"].values

oof2 = np.zeros(len(df))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models_stage2 = []

for train_idx, val_idx in skf.split(X2, y2):
    X_train, X_val = X2.iloc[train_idx], X2.iloc[val_idx]
    y_train, y_val = y2[train_idx], y2[val_idx]

    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
    )

    # Calibrate Stage 2 probabilities
    calibrated = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv=3, n_jobs=-1)
    calibrated.fit(X_train, y_train)

    oof2[val_idx] = calibrated.predict_proba(X_val)[:, 1]
    models_stage2.append(calibrated)

print("Stage2 Acc:", accuracy_score(y2, (oof2 > 0.5).astype(int)))
print("Stage2 LogLoss:", log_loss(y2, oof2))

df["p_win"] = oof2
df = pd.concat([df, oof_stage1], axis=1)
df.to_csv("two_stage_predictions_final.csv", index=False)

# --- SHAP Analysis for Stage 2 ---
final_model = models_stage2[0] 
explainer = shap.TreeExplainer(final_model.estimator)
shap_values = explainer.shap_values(X2)

# Summary bar plot (average absolute contribution)
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X2, plot_type="bar", show=False)
plt.title("Stage 2 Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("shap_barplot.png")

# Use mean absolute SHAP for each set completion feature
set_features = [f"p_{col}_complete" for col in stage1_targets]
shap_means = pd.DataFrame({
    "set": set_features,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)[[X2.columns.get_loc(f) for f in set_features]]
})
shap_means = shap_means.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
shap_means["rank"] = range(1, len(shap_means)+1)
shap_means.to_csv("set_rankings.csv", index=False)

print("\n=== Final Set Rankings ===")
print(shap_means)

# Compute per-game strategy score
scores = pd.DataFrame(index=df.index)
for f in set_features:
    scores[f] = df[f"{f}"] * shap_means.loc[shap_means['set']==f, 'rank'].values[0]

df["strategy_score"] = scores.sum(axis=1)
df.to_csv("strategy_scores.csv", index=False)
print("\nStrategy scores saved to strategy_scores.csv")

# Beeswarm plot
plt.figure(figsize=(12,8))
shap.summary_plot(shap_values, X2, show=False)
plt.title("Stage 2 SHAP Beeswarm")
plt.tight_layout()
plt.savefig("shap_beeswarm.png")
