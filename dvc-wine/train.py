import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


class InferenceModel:
    """
    A class to handle model operations including instantiation, persistence,
    loading and inference.
    """

    def __init__(self, model_params=None, model=None):
        """
        Initialize the model with parameters or use an existing model instance

        Parameters:
        model_params (dict): Parameters for model instantiation
        model: An already instantiated model
        """
        self.model = model
        self.model_params = model_params

        if model is None and model_params:
            self.instantiate_model()

    def instantiate_model(self):
        """
        Instantiate a RandomForestClassifier model with the specified parameters
        """
        if not self.model_params:
            raise ValueError(
                "Model parameters must be provided to instantiate the model"
            )

        self.model = RandomForestClassifier(
            n_estimators=self.model_params.get("n_estimators", 100),
            max_depth=self.model_params.get("max_depth", None),
            min_samples_split=self.model_params.get("min_samples_split", 2),
            random_state=self.model_params.get("random_state", 42),
        )
        return self.model

    def fit(self, X, y):
        """
        Fit the model with training data

        Parameters:
        X: Features
        y: Target
        """
        if self.model is None:
            self.instantiate_model()
        self.model.fit(X, y)
        return self

    def save_model(self, filepath="models/model.pkl"):
        """
        Persist the entire InferenceModel instance to disk using joblib

        Parameters:
        filepath (str): Path where model will be saved
        """
        if self.model is None:
            raise ValueError("No model to save. Instantiate or load a model first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save the entire InferenceModel instance instead of just the underlying model
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
        return filepath

    @classmethod
    def load_model(cls, filepath="models/model.pkl"):
        """
        Load a model from disk

        Parameters:
        filepath (str): Path to the saved model

        Returns:
        InferenceModel: Loaded instance from disk
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load the entire InferenceModel instance
        return joblib.load(filepath)

    def predict(self, X):
        """
        Run inference on data

        Parameters:
        X: Features for prediction

        Returns:
        array: Model predictions
        """
        if self.model is None:
            raise ValueError(
                "No model available for inference. Instantiate or load a model first."
            )

        return self.model.predict(X)


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
wine_data["target"] = wine.target
wine_data.to_csv("data/wine_data.csv", index=False)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save train and test datasets
train_data = X_train.copy()
train_data["target"] = y_train.values
test_data = X_test.copy()
test_data["target"] = y_test.values

train_data.to_csv("data/train.csv", index=False)
test_data.to_csv("data/test.csv", index=False)

# Define parameters (normally these would be in params.yaml)
params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "random_state": 42,
}

# Save params for DVC to track
with open("params.yaml", "w") as f:
    yaml.dump({"train": params}, f)

print(f"Training model with parameters: {params}")

# Use InferenceModel for model training
inference_model = InferenceModel(model_params=params)
inference_model.fit(X_train, y_train)

# Make predictions using the inference model
y_pred = inference_model.predict(X_test)

# Calculate metrics
metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred, average="weighted")),
    "precision": float(precision_score(y_test, y_pred, average="weighted")),
    "recall": float(recall_score(y_test, y_pred, average="weighted")),
}

print(f"Model metrics: {metrics}")

# Save metrics for DVC to track
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f)

# Save model for DVC to track using the InferenceModel
inference_model.save_model("models/model.pkl")

# Example of loading a saved model and running inference
print("Demonstrating model loading and inference...")
loaded_model = InferenceModel.load_model("models/model.pkl")
test_predictions = loaded_model.predict(X_test[:5])
print(f"Sample predictions from loaded model: {test_predictions}")

# Generate and save feature importance plot
# (accessing the underlying model within our InferenceModel)
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": inference_model.model.feature_importances_}
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
