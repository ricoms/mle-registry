#!/usr/bin/env python3
"""
Simple Machine Learning experiment with DVC tracking
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import yaml
import json
from pathlib import Path

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Load the wine dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

# Save raw data (this will be tracked by DVC)
print("Saving raw data...")
wine_data = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_data['target'] = wine.target
wine_data.to_csv("data/wine_data.csv", index=False)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save train and test datasets
train_data = X_train.copy()
train_data['target'] = y_train.values
test_data = X_test.copy()
test_data['target'] = y_test.values

train_data.to_csv("data/train.csv", index=False)
test_data.to_csv("data/test.csv", index=False)

# Define parameters (normally these would be in params.yaml)
params = {
    "n_estimators": 100, 
    "max_depth": 10,
    "min_samples_split": 2,
    "random_state": 42
}

# Save params for DVC to track
with open("params.yaml", "w") as f:
    yaml.dump({"train": params}, f)

print(f"Training model with parameters: {params}")
# Train model
model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"], 
    min_samples_split=params["min_samples_split"],
    random_state=params["random_state"]
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred, average="weighted")),
    "precision": float(precision_score(y_test, y_pred, average="weighted")),
    "recall": float(recall_score(y_test, y_pred, average="weighted"))
}

print(f"Model metrics: {metrics}")

# Save metrics for DVC to track
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f)

# Save model for DVC to track
import joblib
joblib.dump(model, "models/model.pkl")

# Generate and save feature importance plot
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importances_}
)
feature_importance = feature_importance.sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance["feature"], feature_importance["importance"])
plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("plots/feature_importance.png")

# Save feature importance as CSV
feature_importance.to_csv("plots/feature_importance.csv", index=False)

print("Experiment completed successfully!")
print("Run 'dvc add data/' to track data changes")
print("Run 'dvc add models/' to track model changes")
print("Run 'dvc add metrics/' to track metrics")
print("Run 'dvc add plots/' to track plots")
print("Run 'dvc push' to push to remote storage")