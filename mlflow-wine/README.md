# MLflow Experiment: Individual-Centric ML Tracking

This directory contains a setup for running MLflow tracking server locally using Docker Compose and a sample machine learning experiment that integrates with MLflow. MLflow provides a comprehensive UI-focused approach to experiment tracking that optimizes for individual productivity.

## MLflow's Individual-Focused Approach

MLflow provides an experimentation workflow that:
- **Centralizes tracking**: All experiments visible in one UI dashboard
- **Prioritizes visualization**: Rich UI for comparing parameters and metrics
- **Focuses on individual productivity**: Optimized for single data scientist workflows
- **Simplifies experiment logging**: Automatic tracking through Python API

While powerful, this approach contrasts with DVC's Git-based collaborative model that aligns more closely with continuous delivery principles and emphasis on engineering discipline.

## Setup

### Prerequisites

- Docker and Docker Compose installed
- Python 3.13+ installed (for running the experiment)
- uv (Python package manager)

### Running MLflow Server

To start the MLflow server and its dependencies:

```bash
docker-compose up -d
```

This will start three separate services:
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
python train.py
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

## Comparison with Collaborative Approaches

While MLflow excels at individual experimentation, it presents challenges for team-based development:

1. **Infrastructure Complexity**: Requires maintaining multiple services
2. **Limited Git Integration**: Less natural fit with software engineering workflows
3. **CI/CD Challenges**: Custom work needed to integrate with continuous delivery pipelines
4. **Collaboration Model**: Favors independent work followed by sharing, rather than collaborative development
5. **Custom Packaging**: Uses its own model packaging format, separate from Python standards

As in "Continuous Delivery," production systems benefit from frequent integration, automated testing, and engineering discipline—areas where Git-based workflows typically excel.

## Shutting Down

To stop all containers:

```bash
docker-compose down
```

To remove all data (including databases and artifacts):

```bash
docker-compose down -v
```