# analyze_and_export_strategies.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIG ---
INPUT_CSV = "strategy_results.csv"
OUTPUT_CSV_BY_ACTUAL = "strategies_by_win_rate.csv"
OUTPUT_CSV_BY_PREDICTED = "strategies_by_predicted_win_prob.csv"

# the columns that define a "strategy"
STRATEGY_COLS = [
    'buy_brown', 'buy_light_blue', 'buy_pink', 'buy_orange',
    'buy_red', 'buy_yellow', 'buy_green', 'buy_indigo',
    'buy_stations', 'buy_utilities',
    'max_development_level', 'unspendable_cash'
]

# additional columns to report in the summary
REPORT_AGG_COLS = ['roi', 'props', 'houses', 'hotels', 'turns']

# --- LOAD DATA ---
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df):,} rows.")

# Make sure win is numeric 0/1
df['win'] = df['win'].astype(int)

# --- TRAIN A MODEL (for predicted probabilities) ---
X = df[STRATEGY_COLS]
y = df['win']

# small preprocessing: ensure numeric types (booleans -> ints)
X = X.copy()
for c in STRATEGY_COLS:
    if X[c].dtype == 'bool' or X[c].dtype == object:
        X[c] = X[c].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
print("Training RandomForestClassifier...")
clf.fit(X_train, y_train)

# quick evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))

# add predicted win probability to original dataframe
df['predicted_win_prob'] = clf.predict_proba(X)[:, 1]

# --- AGGREGATE BY UNIQUE STRATEGY ---
agg_funcs = {
    'predicted_win_prob': 'mean',
    'win': 'mean',
    'roi': 'mean',
    'props': 'mean',
    'houses': 'mean',
    'hotels': 'mean',
    'turns': 'mean',
    'game_seed': 'count'   # use game_seed as a proxy for number of rows/games
}

grouped = df.groupby(STRATEGY_COLS).agg(agg_funcs).reset_index()
grouped = grouped.rename(columns={'win': 'actual_win_rate', 'game_seed': 'n_games', 'predicted_win_prob': 'avg_predicted_win_prob'})

# Re-order columns for human readability
cols_order = STRATEGY_COLS + ['n_games', 'actual_win_rate', 'avg_predicted_win_prob', 'roi', 'props', 'houses', 'hotels', 'turns']
grouped = grouped[cols_order]

# Sort by actual_win_rate (descending) and write to CSV
grouped_sorted_by_actual = grouped.sort_values(by='actual_win_rate', ascending=False).reset_index(drop=True)
grouped_sorted_by_actual.to_csv(OUTPUT_CSV_BY_ACTUAL, index=False)
print(f"Wrote strategies ordered by ACTUAL win rate -> {OUTPUT_CSV_BY_ACTUAL}")

# Optionally also save ordered by model's predicted win prob
grouped_sorted_by_pred = grouped.sort_values(by='avg_predicted_win_prob', ascending=False).reset_index(drop=True)
grouped_sorted_by_pred.to_csv(OUTPUT_CSV_BY_PREDICTED, index=False)
print(f"Wrote strategies ordered by PREDICTED win prob -> {OUTPUT_CSV_BY_PREDICTED}")
