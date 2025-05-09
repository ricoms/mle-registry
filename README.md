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
| **Development Model** | Individual-centric with centralized sharing | Team-oriented with Git collaboration |
| **Continuous Integration** | Requires custom integration | Native Git-based CI/CD compatibility |

## Impact on Data Science Workflow

### MLflow Approach: Individual-Centric Experimentation

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
- Encourages siloed work with after-the-fact sharing
- Creates friction when integrating with modern CI/CD pipelines

### DVC Approach: Collaborative Engineering

**Advantages:**
- Git-integrated workflow fits with existing development practices
- Data version control alongside code
- Works offline (no server dependency)
- Lighter infrastructure footprint
- Pipeline definition for reproducible workflows
- Natural integration with CI/CD systems
- Seamless compatibility with Python's native packaging ecosystem
- Promotes collaborative workflows through Git's branching and merging
- Enables "Continuous Delivery for Machine Learning" (CD4ML) principles
- Brings engineering discipline to data science processes

**Disadvantages:**
- Steeper learning curve for Git-unfamiliar data scientists
- Missing centralized UI for experiment comparison
- Manual tracking of experiments
- Requires more CLI operations
- Less real-time visibility into running experiments

## Choosing Between MLflow and DVC

### Consider MLflow when:
- Working in teams with diverse technical backgrounds
- Requiring comprehensive experiment visualization and comparison
- Managing a large number of experiments and models
- Needing a model registry for deployment
- Having infrastructure resources to support multiple services
- Team is mostly composed of data scientists with limited software engineering background

### Consider DVC when:
- Working in Git-savvy teams with software engineering practices
- Focusing on reproducibility and data version control
- Working in environments with limited infrastructure
- Needing offline operation capabilities
- Applying "Continuous Delivery" principles to ML workflows
- Promoting collaboration between data scientists, engineers, and DevOps
- Building rigorous, production-grade ML systems
- Implementing Martin Fowler's and Dave Farley's principles of engineering discipline

## Software Engineering Principles Applied to ML

The choice between MLflow and DVC reflects a broader philosophy about how machine learning systems should be built:

**MLflow** emphasizes exploratory data science with tools that make individual experimentation efficient but with less focus on software engineering practices.

**DVC** emphasizes Martin Fowler's concept of "Continuous Delivery" applied to ML, where:
- Version control is fundamental (not just for code but for data and models too)
- Reproducibility is a first-class concern
- Automation of testing and deployment is built into the workflow
- Cross-functional collaboration is enabled through shared tools and practices
- Changes are made incrementally through small, validated steps

As Dave Farley emphasizes, treating machine learning as software engineering means bringing the same rigor, automated testing, and continuous integration practices that have proven successful in traditional software development. DVC's approach aligns closely with these principles.

## Getting Started

Both tools are set up to run the same wine classification example with a Random Forest classifier. See the respective directories for setup instructions:

- [MLflow Example](./mlflow-wine/README.md)
- [DVC Example](./dvc-wine/README.md)
