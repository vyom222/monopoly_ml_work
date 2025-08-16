import pandas as pd
import numpy as np
from scipy.stats import beta

# ==== Load Data ====
df = pd.read_csv("strategy_results.csv")
print(f"Raw rows: {len(df)}")

# ==== Create strategy_id from feature columns ====
strategy_cols = [
    'buy_brown','buy_light_blue','buy_pink','buy_orange','buy_red','buy_yellow',
    'buy_green','buy_indigo','buy_stations','buy_utilities',
    'max_development_level','unspendable_cash'
]
df['strategy_id'] = df[strategy_cols].apply(tuple, axis=1)

# ==== Class distribution ====
class_dist = df['win'].value_counts(normalize=True)
print("\n== Class distribution (win variable) ==")
print(class_dist)

# ==== Group stats per strategy ====
strategy_stats = df.groupby('strategy_id').agg(
    games_played=('win', 'count'),
    wins=('win', 'sum')
).reset_index()

# ==== Beta-Binomial lower bound ====
# Prior: uniform Beta(1,1)
alpha_prior = 1
beta_prior = 1
confidence = 0.95

strategy_stats['win_rate'] = strategy_stats['wins'] / strategy_stats['games_played']
strategy_stats['lower_bound'] = strategy_stats.apply(
    lambda row: beta.ppf((1-confidence),
                         row['wins'] + alpha_prior,
                         row['games_played'] - row['wins'] + beta_prior),
    axis=1
)

# ==== Rank and select top-k strategies ====
top_k = 10
strategy_stats = strategy_stats.sort_values('lower_bound', ascending=False)
top_strategies = strategy_stats.head(top_k)

print(f"\n== Top {top_k} strategies by Beta-Binomial lower bound ==")
print(top_strategies)

# ==== Optional: save top strategies to CSV ====
top_strategies.to_csv("top_strategies.csv", index=False)

# ==== Feature importance (mean abs SHAP) ====
# If you already have SHAP values from another step, load them
try:
    shap_df = pd.read_csv("shap_values_summary.csv")  # feature,mean_abs_shap
    shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
    print("\n== Feature Importance from SHAP ==")
    print(shap_df)
except FileNotFoundError:
    print("\n[Note] No shap_values.csv found, skipping SHAP feature importances.")
