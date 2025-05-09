import os
import argparse

import pandas as pd
from sklearn.datasets import load_wine

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"


def get_latest_model_version(model_name, stage="Production"):
    """
    Get the latest model version from MLflow model registry

    Parameters:
    model_name (str): Name of the registered model
    stage (str): Model stage to use (defaults to "Production")

    Returns:
    str: URI of the model
    """
    client = MlflowClient()

    try:
        # Try to get the model by stage
        model_versions = client.get_latest_versions(model_name, stages=[stage])
        if model_versions:
            return f"models:/{model_name}/{stage}"

        # If no model in the requested stage, get the latest version
        model_versions = client.search_model_versions(f"name='{model_name}'")
        if model_versions:
            latest_version = max([int(mv.version) for mv in model_versions])
            return f"models:/{model_name}/{latest_version}"

        return None

    except Exception as e:
        print(f"Error fetching model: {e}")
        return None


def get_best_run_model(experiment_name):
    """
    Get the best model from experiment runs based on accuracy

    Parameters:
    experiment_name (str): Name of the experiment

    Returns:
    str: URI of the model artifact
    """
    # Get experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found")
        return None

    # Get all runs
    runs = mlflow.search_runs(
        experiment_ids=experiment.experiment_id, order_by=["metrics.accuracy DESC"]
    )

    if runs.empty:
        print("No runs found with metrics")
        return None

    # Get the run with highest accuracy
    best_run_id = runs.iloc[0]["run_id"]
    return f"runs:/{best_run_id}/random_forest_model"


def load_sample_data():
    """
    Load sample wine data for prediction

    Returns:
    DataFrame: Sample data for prediction
    """
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target)

    # Return first 5 samples
    return X.iloc[:5], y.iloc[:5]


def main(args):
    # Set MLflow tracking URI to connect to the MLflow server
    mlflow.set_tracking_uri(args.tracking_uri)

    model_uri = None

    # Try to get model from registry if name is provided
    if args.model_name:
        model_uri = get_latest_model_version(args.model_name, args.stage)
        if model_uri:
            print(f"Using model from registry: {model_uri}")

    # Fall back to getting best model from experiment
    if not model_uri:
        model_uri = get_best_run_model(args.experiment)
        if model_uri:
            print(f"Using model from experiment run: {model_uri}")

    if not model_uri:
        print("Could not find any model to use for inference")
        return

    # Load the model
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load sample data
    X_sample, y_true = load_sample_data()

    # Make predictions
    try:
        predictions = model.predict(X_sample)
        probabilities = model.predict_proba(X_sample)

        # Display results
        print("\nSample Inference Results:")
        print("-" * 50)
        print("Features:")
        print(X_sample)
        print("\nActual labels:", y_true.values)
        print("Predicted labels:", predictions)

        # Display probabilities for each class
        print("\nClass probabilities:")
        pd.set_option("display.precision", 3)
        proba_df = pd.DataFrame(
            probabilities, columns=[f"Class {i}" for i in range(len(probabilities[0]))]
        )
        print(proba_df)

        # Calculate accuracy on the sample
        accuracy = (predictions == y_true).mean()
        print(f"\nSample accuracy: {accuracy:.2f}")

    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using MLflow model")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://localhost:5001",
        help="MLflow tracking URI (default: http://localhost:5001)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="wine-classification",
        help="Experiment name to find model from (default: wine-classification)",
    )
    parser.add_argument(
        "--model-name", type=str, help="Registered model name in MLflow Model Registry"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Production",
        help="Model stage (default: Production)",
    )

    args = parser.parse_args()
    main(args)
