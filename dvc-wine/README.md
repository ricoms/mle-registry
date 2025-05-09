# DVC Machine Learning Experiment: Git-Based Collaboration

This directory demonstrates a simple yet powerful machine learning workflow using Data Version Control (DVC) - a lightweight, Git-integrated approach that emphasizes collaboration and software engineering best practices.

## Why DVC?

DVC brings software engineering discipline to machine learning by:
- **Git Integration**: Leverages existing Git workflows familiar to engineers
- **Simplicity**: Minimal infrastructure (just MinIO for remote storage)
- **Collaboration**: Enables cross-functional teams to work together through Git
- **DevOps Ready**: Naturally fits into CI/CD pipelines and deployment workflows
- **Continuous Delivery for ML**: Supports Martin Fowler's principles of incremental, tested changes

## Setup

### Prerequisites

- Docker and Docker Compose installed
- Python 3.13+ installed
- uv (Python package manager)

### Running DVC Storage Backend

To start the MinIO storage backend:

```bash
docker-compose up -d
```

This starts **only MinIO** for artifact storage - notice the simplicity compared to MLflow's multi-container setup!

MinIO Console is available at:
- http://localhost:9001 (login with minioadmin/minioadmin)

## Running the Experiment

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Initialize DVC:
```bash
dvc init --no-scm
```

3. Configure DVC remote storage:
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

5. Track files with DVC:
```bash
dvc add data/
dvc add models/
dvc add metrics/
dvc add plots/
```

6. Push artifacts to remote storage:
```bash
dvc push
```

## Collaborative Engineering Benefits

With DVC, you gain these advantages:

1. **Git-Centered Workflow**: Changes to code, data, and models are tracked together through standard Git practices
2. **Team Collaboration**: Multiple data scientists can work on branches and merge changes using familiar Git workflows
3. **DevOps Integration**: CI/CD systems can pull code and data for automated testing and deployment
4. **Engineering Discipline**: As Dave Farley advocates, brings rigor and repeatability to ML processes
5. **Incremental Development**: Supports Martin Fowler's continuous delivery principles with small, validated changes

## Working with DVC

### Retrieving Data

```bash
dvc pull
```

### Viewing Metrics

```bash
dvc metrics show
```

### Comparing Experiments

```bash
dvc exp diff
```

### Supporting Team Collaboration

DVC excels at enabling collaboration across roles:

1. **Data Scientists**: Work on feature branches to experiment with algorithms
2. **Software Engineers**: Review changes and integrate ML code with applications 
3. **DevOps/MLOps**: Build automation pipelines using Git triggers
4. **Product Managers**: Track progress through metrics stored in Git

## Shutting Down

To stop all containers:

```bash
docker-compose down
```

To remove all data:

```bash
docker-compose down -v
```