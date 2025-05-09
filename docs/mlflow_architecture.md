# MLflow Architecture

MLflow provides a comprehensive experiment tracking and model registry solution with a server-based architecture.

```mermaid
graph TD
    subgraph "MLflow Architecture"
        A[Data Scientist] -->|Logs experiments| B[MLflow Tracking Server]
        B <-->|Stores metadata| C[PostgreSQL]
        B <-->|Stores artifacts| D[MinIO]
        E[Data Engineer] -->|Views experiments| B
        F[ML Engineer] -->|Registers & deploys models| B
        B -->|Serves UI| G[Web Browser]
    end
    
    style B fill:#85C1E9,stroke:#5499C7,stroke-width:2px
    style C fill:#F9E79F,stroke:#F7DC6F,stroke-width:2px
    style D fill:#F5B041,stroke:#E67E22,stroke-width:2px
```

## Component Descriptions

1. **MLflow Tracking Server**: Central server that receives experiment logs, serves the UI, and manages the model registry
2. **PostgreSQL Database**: Stores experiment metadata, parameters, metrics, and model registry information
3. **MinIO Object Storage**: Stores experiment artifacts like models, datasets, and other large files
4. **Web UI**: Provides visualization and comparison of experiments and models

## Key Features

- Centralized experiment tracking
- Model registry for versioning and staging
- Artifact storage
- Parameter and metric logging
- UI for experiment comparison
- **Custom Model Packaging**: MLflow uses its own packaging format and conventions

## Packaging Approach

MLflow implements a custom packaging system that diverges from standard Python packaging practices:

- Uses its own `MLmodel` format for model metadata
- Creates custom model packaging formats (MLflow Models)
- Imposes specific structure for saving and loading models
- Handles dependencies through custom model flavor configurations
- May require adaptations when integrating with existing Python projects or deployment pipelines
- Can lead to additional complexity when working with projects that follow standard Python packaging conventions