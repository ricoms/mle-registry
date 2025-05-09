# DVC Machine Learning Experiment

This directory contains a setup for running a simple machine learning experiment using Data Version Control (DVC). The example uses the Wine dataset from scikit-learn to train a Random Forest classifier with various hyperparameters.

## Setup

### Prerequisites

- Docker and Docker Compose installed
- Python 3.8+ installed (for running the experiment)
- pip (Python package manager)
- DVC installed (`pip install dvc dvc[s3]`)

### Running DVC Storage Backend

To start the MinIO storage backend for DVC:

```bash
docker-compose up -d
```

This will start:
- MinIO: For artifact storage (datasets, models, metrics, etc.)
- A setup container that creates the necessary bucket

MinIO Console (for inspecting artifacts) is available at:
- http://localhost:9001 (login with minioadmin/minioadmin)

## Running the Experiment

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Initialize DVC (if not already done):
```bash
dvc init --no-scm
```

3. Configure DVC to use the MinIO storage:
```bash
dvc remote add minio-remote s3://dvcstore
dvc remote modify minio-remote endpointurl http://localhost:9000
dvc remote modify minio-remote access_key_id minioadmin
dvc remote modify minio-remote secret_access_key minioadmin
dvc remote default minio-remote
```

4. Run the experiment:
```bash
python train.py
```

5. Track the generated files with DVC:
```bash
dvc add data/
dvc add models/
dvc add metrics/
dvc add plots/
```

6. Push the tracked files to DVC remote storage:
```bash
dvc push
```

## Experiment Structure

The experiment:
1. Loads the wine dataset from scikit-learn
2. Splits the data into training and testing sets
3. Trains a Random Forest model with specified hyperparameters
4. Evaluates the model and logs:
   - Metrics (accuracy, F1 score, precision, recall)
   - The trained model itself
   - Feature importance as a CSV artifact and visualization

## Working with DVC

### Retrieving Data

To retrieve tracked data from the remote:

```bash
dvc pull
```

### Viewing Metrics

To view the logged metrics:

```bash
dvc metrics show
```

### Comparing Experiments

For comparing multiple experiment runs (after making changes and re-running):

```bash
dvc exp diff
```

## Shutting Down

To stop all containers:

```bash
docker-compose down
```

To remove all data (including MinIO storage):

```bash
docker-compose down -v
```