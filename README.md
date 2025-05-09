# Machine Learning Experiment Registry

This repository provides a comparison of two popular machine learning experiment tracking and model registry tools: MLflow and DVC (Data Version Control). Both setups use the same machine learning example (wine classification) to demonstrate their different approaches to infrastructure and data science workflows.

## MLflow vs DVC: Infrastructure Comparison

For a visual representation of each architecture:
- [MLflow Architecture Diagram](./docs/mlflow_architecture.md)
- [DVC Architecture Diagram](./docs/dvc_architecture.md)

| Feature | MLflow | DVC |
|---------|--------|-----|
| **Core Philosophy** | Centralized experiment tracking and model registry | Git-based version control for ML projects with focus on data |
| **Architecture** | Server-based with dedicated UI and API | Command-line based, integrated with Git |
| **Infrastructure Components** | PostgreSQL, MinIO, MLflow Tracking Server | MinIO only (for remote storage) |
| **Container Requirements** | 3 containers (PostgreSQL, MinIO, MLflow Server) | 1 container (MinIO for storage) |
| **Storage Strategy** | Metrics in database, artifacts in object storage | Everything in object storage, references in Git |
| **UI Access** | Built-in web interface | Uses Git and optional external tools |
| **Entry Barrier** | Setup multiple services | Simpler initial setup |
| **Scaling Complexity** | Higher (multiple services to manage) | Lower (fewer components) |
| **Packaging Approach** | Custom MLflow packaging format (MLmodel) | Compatible with native Python packaging |

## Impact on Data Science Workflow

### MLflow Approach

**Advantages:**
- Centralized experiment tracking with rich UI for comparing runs
- Automatic parameter, metric, and artifact logging
- Built-in model registry for deployment
- Less Git knowledge required
- Unified view of all experiments and models
- Real-time tracking during experiment execution
- API-first design enables integration with various tools

**Disadvantages:**
- More complex infrastructure setup and maintenance
- Requires running server components
- More resource-intensive
- Dependency on external services (PostgreSQL, object storage)
- Custom packaging system that's separate from standard Python practices
- May require adaptations when integrating with existing Python projects

### DVC Approach

**Advantages:**
- Git-integrated workflow fits with existing development practices
- Data version control alongside code
- Works offline (no server dependency)
- Lighter infrastructure footprint
- Pipeline definition for reproducible workflows
- Natural integration with CI/CD systems
- Seamless compatibility with Python's native packaging ecosystem
- No need to adapt to proprietary packaging formats

**Disadvantages:**
- Steeper learning curve for Git-unfamiliar data scientists
- Missing centralized UI for experiment comparison
- Manual tracking of experiments
- Requires more CLI operations
- Less real-time visibility into running experiments

## Choosing Between MLflow and DVC

**Consider MLflow when:**
- Working in teams with diverse technical backgrounds
- Requiring comprehensive experiment visualization and comparison
- Managing a large number of experiments and models
- Needing a model registry for deployment
- Having infrastructure resources to support multiple services
- Unified packaging approach is preferred over standard Python packaging

**Consider DVC when:**
- Working in Git-savvy teams with software engineering practices
- Focusing on reproducibility and data version control
- Working in environments with limited infrastructure
- Needing offline operation capabilities
- Integrating tightly with CI/CD pipelines
- Preferring to use standard Python packaging practices
- Requiring flexibility in model deployment pipelines

## Getting Started

Both tools are set up to run the same wine classification example with a Random Forest classifier. See the respective directories for setup instructions:

- [MLflow Example](./mlflow-wine/README.md)
- [DVC Example](./dvc-wine/README.md)
