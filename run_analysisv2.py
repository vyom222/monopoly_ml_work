# run_analysis_v2.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    classification_report, mean_squared_error, r2_score
)
from math import sqrt
from collections import OrderedDict

# ----------------- config -----------------
CSV_PATH = "strategy_results.csv"
BUY_FLAGS = ['buy_brown', 'buy_light_blue', 'buy_pink', 'buy_orange',
             'buy_red', 'buy_yellow', 'buy_green', 'buy_indigo',
             'buy_stations', 'buy_utilities']
STRATEGY_FEATURES = BUY_FLAGS + ["max_development_level", "unspendable_cash"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
# ------------------------------------------

def wilson_lower_bound(wins, n, z=1.96):
    """Wilson score interval lower bound (95% by default). Handles n==0."""
    if n == 0:
        return 0.0
    phat = wins / n
    denom = 1 + z*z / n
    num = phat + z*z/(2*n) - z * np.sqrt((phat*(1-phat) + z*z/(4*n))/n)
    return num/denom

def laplace_mean(wins, n):
    return (wins + 1) / (n + 2)

def make_strategy_id(row):
    # deterministic tuple key (booleans -> True/False, ints unchanged)
    return tuple([row[f] for f in BUY_FLAGS] + [int(row["max_development_level"]), int(row["unspendable_cash"])])

def main():
    print("Loading:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    print("Raw rows:", len(df))
    # Drop ROI column if present
    if "roi" in df.columns:
        df = df.drop(columns=["roi"])

    # Basic class distribution
    print("\n== Class distribution (win variable) ==")
    print(df["win"].value_counts(normalize=True).rename("proportion"))

    # Build strategy_id
    df["strategy_id"] = df.apply(make_strategy_id, axis=1)

    # Aggregate per-strategy
    agg = df.groupby("strategy_id").agg(
        games_played=("win", "size"),
        wins=("win", "sum"),
    ).reset_index()
    agg["win_rate"] = agg["wins"] / agg["games_played"]
    agg["laplace_mean"] = agg.apply(lambda r: laplace_mean(r["wins"], r["games_played"]), axis=1)
    agg["wilson_lower"] = agg.apply(lambda r: wilson_lower_bound(r["wins"], r["games_played"]), axis=1)

    # For diagnostics: add averaged outcome-based fields (partial_* and ever_*)
    # If these columns exist in df, compute per-strategy means for them (useful for regressor B)
    outcome_cols = [c for c in df.columns if c.startswith("partial_") or c.startswith("ever_") or c.startswith("max_owned_")]
    if outcome_cols:
        outcome_agg = df.groupby("strategy_id")[outcome_cols].mean().reset_index()
        agg = agg.merge(outcome_agg, on="strategy_id", how="left")

    # Save aggregated
    agg_sorted = agg.sort_values(["wilson_lower", "win_rate"], ascending=False)
    agg_sorted.to_csv("strategies_aggregated.csv", index=False)
    print(f"\n== Per-strategy aggregation saved to strategies_aggregated.csv (unique strategies: {len(agg)}) ==")

    # ----------------- Train classifier on per-game rows -----------------
    print("\n== Training classifier on per-game rows ==")
    X = df[STRATEGY_FEATURES].astype(float)
    y = df["win"].astype(int)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # RandomForest with class balancing and calibration
    clf_base = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    clf = CalibratedClassifierCV(clf_base, cv=5, method="sigmoid")
    print("Fitting classifier (this can take some time)...")
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    print(f"Classifier accuracy: {acc:.4f}")
    print(f"Classifier ROC AUC: {auc:.4f}")
    print(f"Classifier Brier score: {brier:.6f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Feature importances via base estimator
    try:
        importances = clf_base.feature_importances_
        fi = pd.Series(importances, index=STRATEGY_FEATURES).sort_values(ascending=False)
        print("\nFeature importances (from RandomForest):")
        print(fi)
    except Exception:
        pass

    # ----------------- Predict per-strategy with classifier -----------------
    # Two ways: (A) average predicted prob across all games for strategy (use rows) (B) predict from canonical feature vector per strategy
    print("\n== Predicting per-strategy win-probability using classifier ==")
    # A) average over rows (use df rows)
    df["pred_prob"] = clf.predict_proba(df[STRATEGY_FEATURES].astype(float))[:, 1]
    avg_pred = df.groupby("strategy_id")["pred_prob"].mean().reset_index().rename(columns={"pred_prob": "avg_pred_prob"})
    agg = agg.merge(avg_pred, on="strategy_id", how="left")

    # B) single representative strategy row (features are deterministic from id) -> compute predicted prob
    # Build a DataFrame of representative features from the unique strategy_id rows
    def id_to_row(sid):
        # sid is tuple: buy flags..., max_dev, unspendable
        vals = list(sid)
        d = OrderedDict()
        for i, flag in enumerate(BUY_FLAGS):
            d[flag] = int(vals[i])
        d["max_development_level"] = int(vals[len(BUY_FLAGS)])
        d["unspendable_cash"] = int(vals[len(BUY_FLAGS) + 1])
        return d

    repr_rows = pd.DataFrame([id_to_row(sid) for sid in agg["strategy_id"]])
    repr_rows_pred = clf.predict_proba(repr_rows[STRATEGY_FEATURES].astype(float))[:, 1]
    agg["repr_pred_prob"] = repr_rows_pred

    # ----------------- Regressor on aggregated strategies (diagnostic) -----------------
    print("\n== Training regressors on aggregated strategies (diagnostic) ==")
    # Regressor A: only strategy features (pre-game)
    X_agg_A = pd.DataFrame([id_to_row(sid) for sid in agg["strategy_id"]])[STRATEGY_FEATURES].astype(float)
    y_agg = agg["win_rate"].values
    # split
    X_A_train, X_A_test, yA_train, yA_test = train_test_split(X_agg_A, y_agg, test_size=0.2, random_state=RANDOM_STATE)
    regA = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    regA.fit(X_A_train, yA_train)
    yA_pred = regA.predict(X_A_test)
    rmseA = sqrt(mean_squared_error(yA_test, yA_pred))
    r2A = r2_score(yA_test, yA_pred)
    print(f"Regressor A (strategy features) Test RMSE: {rmseA:.4f}, R^2: {r2A:.4f}")

    # Regressor B: strategy features + averaged outcome features (this leaks outcome data but good for diagnostics)
    outcome_cols_present = [c for c in agg.columns if c.startswith("partial_") or c.startswith("ever_") or c.startswith("max_owned_")]
    if outcome_cols_present:
        X_agg_B = pd.concat([X_agg_A.reset_index(drop=True), agg[outcome_cols_present].reset_index(drop=True)], axis=1).astype(float)
        X_B_train, X_B_test, yB_train, yB_test = train_test_split(X_agg_B, y_agg, test_size=0.2, random_state=RANDOM_STATE)
        regB = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
        regB.fit(X_B_train, yB_train)
        yB_pred = regB.predict(X_B_test)
        rmseB = sqrt(mean_squared_error(yB_test, yB_pred))
        r2B = r2_score(yB_test, yB_pred)
        print(f"Regressor B (strategy + outcome features) Test RMSE: {rmseB:.4f}, R^2: {r2B:.4f}")
    else:
        print("No aggregated outcome columns found; skipping Regressor B.")

    # ----------------- Final ranking + save -----------------
    # choose primary ranking: wilson_lower (conservative) and repr_pred_prob
    agg_out = agg.copy()
    agg_out["repr_pred_prob"] = agg_out["repr_pred_prob"].fillna(agg_out["avg_pred_prob"])
    # create columns for human-readable strategy breakdown
    agg_out = agg_out.sort_values(["wilson_lower", "repr_pred_prob"], ascending=False)
    # expand strategy_id back to columns for convenience
    strat_cols = pd.DataFrame([id_to_row(sid) for sid in agg_out["strategy_id"]]).reset_index(drop=True)
    final = pd.concat([strat_cols, agg_out.reset_index(drop=True)], axis=1)
    FINAL_COLS = strat_cols.columns.tolist() + ["games_played", "wins", "win_rate", "laplace_mean", "wilson_lower", "avg_pred_prob", "repr_pred_prob"]
    for c in FINAL_COLS:
        if c not in final.columns:
            final[c] = final.get(c, np.nan)
    final[FINAL_COLS].to_csv("strategies_ranked.csv", index=False)
    print("Wrote strategies_ranked.csv (ranked by Wilson lower bound then model pred prob).")

    # Show top 10
    print("\n== Top 10 strategies by Wilson lower bound (conservative) ==")
    print(final[FINAL_COLS].head(10).to_string(index=False))

    # Save classifier and regressors if desired (skipped here)
    print("\nAll done. Outputs: strategies_aggregated.csv, strategies_ranked.csv")

if __name__ == "__main__":
    main()
