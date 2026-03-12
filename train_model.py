"""
Train and save the loan recommendation model.
Run once: python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("training_data.csv")

# Encode categoricals
cat_cols = ["subcategory", "region"]
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[cat_cols] = enc.fit_transform(df[cat_cols])

X = df.drop(columns=["log_loan_amount"])
y = df["log_loan_amount"]

# Train — use 200 estimators, store individual tree predictions for intervals
model = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

oob_model = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    oob_score=True
)
oob_model.fit(X, y)
print(f"OOB R²: {oob_model.oob_score_:.3f}")

with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "encoder": enc, "feature_cols": list(X.columns)}, f)

print("Model saved to model.pkl")
