import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load CSV
df = pd.read_csv("strategy_results.csv")

# Features for the model
feature_cols = [
    'buy_brown', 'buy_light_blue', 'buy_pink', 'buy_orange',
    'buy_red', 'buy_yellow', 'buy_green', 'buy_indigo',
    'buy_stations', 'buy_utilities', 'max_development_level',
    'unspendable_cash'
]

X = df[feature_cols]
y = df['win']  # Target variable (1 = win, 0 = loss)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols)
print("\nFeature Importances:\n", importances.sort_values(ascending=False))

# Predict win probability for every row in the dataset
df['predicted_win_prob'] = model.predict_proba(X)[:, 1]

# Group by strategy for predicted win probability
predicted_group = (
    df.groupby(feature_cols)['predicted_win_prob']
    .mean()
    .reset_index()
    .rename(columns={'predicted_win_prob': 'avg_predicted_win_prob'})
)

# Group by strategy for actual win rate
actual_group = (
    df.groupby(feature_cols)['win']
    .mean()
    .reset_index()
    .rename(columns={'win': 'actual_win_rate'})
)

# Merge predicted and actual win rates
comparison = pd.merge(predicted_group, actual_group, on=feature_cols)

# Sort by predicted win probability
comparison = comparison.sort_values(by='avg_predicted_win_prob', ascending=False)
    

print("\nTop 5 strategies (predicted vs actual win rate):")
print(comparison.head(5))
