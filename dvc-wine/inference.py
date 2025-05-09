#!/usr/bin/env python3
"""
DVC Model Inference Script

This script demonstrates how to load a model versioned with DVC
and use it for inference.
"""

import argparse
import os
import subprocess

import joblib
import pandas as pd
from sklearn.datasets import load_wine


def ensure_dvc_pull(model_path="models/model.pkl", quiet=False):
    """
    Ensure the model file is available locally by running dvc pull if needed

    Parameters:
    model_path (str): Path to the model file
    quiet (bool): Whether to suppress output

    Returns:
    bool: True if the model is available, False otherwise
    """
    if os.path.exists(model_path):
        if not quiet:
            print(f"Model already exists at {model_path}")
        return True

    if not quiet:
        print(
            f"Model not found at {model_path}, attempting to pull from DVC storage..."
        )

    try:
        # Run dvc pull on the model file
        cmd = ["dvc", "pull", model_path]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE if quiet else None)

        if os.path.exists(model_path):
            if not quiet:
                print("Successfully pulled model from DVC storage")
            return True
        else:
            if not quiet:
                print(
                    "Failed to retrieve model: File still doesn't exist after dvc pull"
                )
            return False
    except subprocess.CalledProcessError as e:
        if not quiet:
            print(f"Error running dvc pull: {e}")
        return False
    except Exception as e:
        if not quiet:
            print(f"Unexpected error: {e}")
        return False


def load_model(model_path="models/model.pkl", quiet=False):
    """
    Load an InferenceModel instance from file

    Parameters:
    model_path (str): Path to the saved model
    quiet (bool): Whether to suppress output

    Returns:
    InferenceModel: Loaded model instance or None if loading fails
    """
    # First make sure the model is available locally
    if not ensure_dvc_pull(model_path, quiet):
        return None

    try:
        # Load the InferenceModel instance
        model = joblib.load(model_path)
        if not quiet:
            print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        if not quiet:
            print(f"Error loading model: {e}")
        return None


def get_model_metrics(metrics_path="metrics/metrics.json", quiet=False):
    """
    Get model metrics from the metrics file tracked by DVC

    Parameters:
    metrics_path (str): Path to the metrics file
    quiet (bool): Whether to suppress output

    Returns:
    dict: Metrics dictionary
    """
    import json

    if not os.path.exists(metrics_path):
        try:
            # Try to pull metrics from DVC storage
            cmd = ["dvc", "pull", metrics_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE if quiet else None)
        except Exception as e:
            if not quiet:
                print(f"Error pulling metrics: {e}")
            return None

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except Exception as e:
            if not quiet:
                print(f"Error reading metrics file: {e}")
            return None
    else:
        if not quiet:
            print(f"Metrics file not found: {metrics_path}")
        return None


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


def load_test_data(test_path="data/test.csv", quiet=False):
    """
    Load test data from CSV file

    Parameters:
    test_path (str): Path to the test CSV file
    quiet (bool): Whether to suppress output

    Returns:
    tuple: (X, y) dataframes
    """
    if not os.path.exists(test_path):
        try:
            # Try to pull test data from DVC storage
            cmd = ["dvc", "pull", test_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE if quiet else None)
        except Exception as e:
            if not quiet:
                print(f"Error pulling test data: {e}")
            return None, None

    if not os.path.exists(test_path):
        if not quiet:
            print(f"Test data file not found: {test_path}")
        return None, None

    try:
        # Load the test data
        df = pd.read_csv(test_path)

        # Separate features and target
        if "target" in df.columns:
            X = df.drop("target", axis=1)
            y = df["target"]
            return X, y
        else:
            if not quiet:
                print("No 'target' column found in test data")
            return df, None
    except Exception as e:
        if not quiet:
            print(f"Error loading test data: {e}")
        return None, None


def main(args):
    # Load the model
    model = load_model(args.model_path, args.quiet)
    if model is None:
        return

    # Get metrics if available
    if args.show_metrics:
        metrics = get_model_metrics(args.metrics_path, args.quiet)
        if metrics:
            print("\nModel Metrics:")
            print("-" * 50)
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")
            print("-" * 50)

    # Load data for prediction
    if args.use_test_data:
        X_sample, y_true = load_test_data(args.test_path, args.quiet)
        if X_sample is None:
            print("Using sample data instead of test data")
            X_sample, y_true = load_sample_data()
    else:
        X_sample, y_true = load_sample_data()

    # Make predictions
    try:
        predictions = model.predict(X_sample)

        # Display results
        print("\nSample Inference Results:")
        print("-" * 50)
        print("Features:")
        print(X_sample)

        if y_true is not None:
            print("\nActual labels:", y_true.values)
        print("Predicted labels:", predictions)

        # Calculate accuracy on the sample if true values are available
        if y_true is not None:
            accuracy = (predictions == y_true).mean()
            print(f"\nSample accuracy: {accuracy:.2f}")

        # Try to get class probabilities if model supports it
        try:
            if hasattr(model.model, "predict_proba"):
                probabilities = model.model.predict_proba(X_sample)

                # Display probabilities for each class
                print("\nClass probabilities:")
                pd.set_option("display.precision", 3)
                proba_df = pd.DataFrame(
                    probabilities,
                    columns=[f"Class {i}" for i in range(len(probabilities[0]))],
                )
                print(proba_df)
        except Exception as e:
            if not args.quiet:
                print(f"Could not calculate probabilities: {e}")

    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference using DVC-tracked model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/model.pkl",
        help="Path to the model file (default: models/model.pkl)",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="metrics/metrics.json",
        help="Path to metrics file (default: metrics/metrics.json)",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/test.csv",
        help="Path to test data file (default: data/test.csv)",
    )
    parser.add_argument(
        "--use-test-data",
        action="store_true",
        help="Use test data instead of sample data",
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show model metrics from the metrics file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output",
    )

    args = parser.parse_args()
    main(args)
