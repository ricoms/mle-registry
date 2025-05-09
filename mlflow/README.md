# MLflow Experiment

This directory contains a setup for running MLflow tracking server locally using Docker Compose and a sample machine learning experiment that integrates with MLflow.

## Setup

### Prerequisites

- Docker and Docker Compose installed
- Python 3.8+ installed (for running the experiment)
- pip (Python package manager)

### Running MLflow Server

To start the MLflow server and its dependencies (PostgreSQL and MinIO):

```bash
docker-compose up -d
```

This will start:
- PostgreSQL: For MLflow's backend store (metrics, parameters, etc.)
- MinIO: For artifact storage (models, plots, etc.)
- MLflow Server: UI and tracking server

### MLflow UI

Once running, the MLflow UI is available at:
- http://localhost:5000

MinIO Console (for inspecting artifacts) is available at:
- http://localhost:9001 (login with minio/minio123)

## Running the Sample Experiment

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the experiment:
```bash
python ml_experiment.py
```

The experiment trains multiple Random Forest classifiers on the wine dataset with different hyperparameters and logs the results to MLflow.

## Experiment Structure

The experiment:
1. Loads the wine dataset from scikit-learn
2. Splits the data into training and testing sets
3. Trains multiple Random Forest models with different hyperparameters
4. For each model configuration, it logs:
   - Parameters (n_estimators, max_depth, min_samples_split)
   - Metrics (accuracy, F1 score, precision, recall)
   - The trained model itself
   - Feature importance as a CSV artifact

## Viewing Results

After running the experiment, visit the MLflow UI at http://localhost:5000 to:
- Compare different runs
- View metrics and parameters
- Download or load trained models
- View artifacts like feature importance data

## Shutting Down

To stop all containers:

```bash
docker-compose down
```

To remove all data (including databases and artifacts):

```bash
docker-compose down -v
```