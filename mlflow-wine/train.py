import os

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
mlflow.set_tracking_uri("http://localhost:5001")

EXPERIMENT_NAME = "wine-classification"
mlflow.set_experiment(EXPERIMENT_NAME)

# Load the wine dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define some hyperparameters to try
n_estimators_options = [50, 100, 200]
max_depth_options = [None, 10, 20]
min_samples_split_options = [2, 5, 10]

# Run different model configurations and log to MLflow
for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        for min_samples_split in min_samples_split_options:
            # Start an MLflow run
            with mlflow.start_run():
                # Create and train the model
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                )
                model.fit(X_train, y_train)

                # Make predictions and calculate metrics
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")

                # Log model parameters
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("min_samples_split", min_samples_split)

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

                # Log feature importance as artifacts
                feature_importances = pd.Series(
                    model.feature_importances_, index=X.columns
                ).sort_values(ascending=False)

                # Create a feature importance CSV
                feature_imp_df = pd.DataFrame(
                    {"feature": X.columns, "importance": model.feature_importances_}
                ).sort_values("importance", ascending=False)

                # Save feature importance to CSV
                feature_imp_path = "feature_importance.csv"
                feature_imp_df.to_csv(feature_imp_path, index=False)
                mlflow.log_artifact(feature_imp_path)

                # Log the model
                mlflow.sklearn.log_model(model, "random_forest_model")

                # Print run info
                print(
                    f"Run completed with parameters: n_estimators={n_estimators}, "
                    f"max_depth={max_depth}, min_samples_split={min_samples_split}"
                )
                print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
                print("-" * 50)

print(
    "All model runs completed. View results in the MLflow UI at http://localhost:5000"
)
