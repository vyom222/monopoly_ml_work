import pandas as pd

# Load dataset
df = pd.read_csv("strategy_results.csv")

# Dictionary of all sets
sets = ["brown","lightblue","pink","orange","red","yellow","green","indigo","railroads","utilities"]

results = []
for s in sets:
    ever_col = f"ever_{s}"
    max_col = f"max_owned_{s}"

    # A set is complete if max_owned = 2 (utilities), 4 (railroads), else 3
    needed = 2 if s == "utilities" or s == "indigo" or s == "brown" else (4 if s == "railroads" else 3)

    completed = (df[max_col] == needed)
    prob_complete = completed.mean()                 # P(complete)
    prob_win_given_complete = df.loc[completed, "win"].mean() if completed.any() else 0
    prob_win_given_not = df.loc[~completed, "win"].mean() if (~completed).any() else 0

    results.append({
        "Set": s,
        "P(complete)": prob_complete,
        "P(win | complete)": prob_win_given_complete,
        "P(win | not complete)": prob_win_given_not
    })

prob_table = pd.DataFrame(results)
print(prob_table)



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

# Features = completed sets (binary)
X = pd.DataFrame()
for s in sets:
    needed = 2 if s == "utilities" or s == "indigo" or s == "brown" else (4 if s == "railroads" else 3)
    X[s] = (df[f"max_owned_{s}"] == needed).astype(int)

y = df["win"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict_proba(X_test)[:,1]
print("Accuracy:", accuracy_score(y_test, preds > 0.5))
print("ROC AUC:", roc_auc_score(y_test, preds))

# Feature importance (coefficients)
import numpy as np
importance = pd.DataFrame({
    "Set": X.columns,
    "Importance": np.round(model.coef_[0], 3)
}).sort_values(by="Importance", ascending=False)

print(importance)
