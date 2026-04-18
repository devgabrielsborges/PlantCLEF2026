<!--
Sync Impact Report:
- Version change: [CONSTITUTION_VERSION] → 1.0.0
- List of modified principles:
  - [PRINCIPLE_1_NAME] → I. Clean Code & Clean Architecture
  - [PRINCIPLE_2_NAME] → II. Scientific Methodology & Experiment Rigor
  - [PRINCIPLE_3_NAME] → III. Data Scientist & ML Engineer Mindset
  - [PRINCIPLE_4_NAME] → IV. Test-Driven ML Development (TD-MLD)
  - [PRINCIPLE_5_NAME] → V. Reproducible Infrastructure & MLOps
- Added sections:
  - Modeling & Data Constraints
  - Development Workflow
- Removed sections: None
- Templates requiring updates: ✅ updated (generic placeholders remain valid)
- Follow-up TODOs: None
-->

# PlantCLEF 2026 Constitution

## Core Principles

### I. Clean Code & Clean Architecture
Adhere to a clean architecture where core ML logic (data models, species prediction logic, custom metrics) is decoupled from the training framework (PyTorch) or tracking infrastructure (MLflow). Code must be DRY, self-documenting, and maintain high readability. Business logic MUST remain independent of external dependencies to ensure long-term maintainability.

### II. Scientific Methodology & Experiment Rigor
Every experiment MUST be reproducible. Mandatory use of MLflow to track all parameters, metrics (Macro-F1, Precision, Recall, ROC AUC), and model artifacts. Standardize validation splits to ensure comparability between different model versions and hyperparameter configurations. Empirical evidence MUST drive all modeling decisions.

### III. Data Scientist & ML Engineer Mindset
Prioritize deep data understanding over model complexity. Rigorously analyze the domain shift between training data (single plant images) and testing data (complex vegetation quadrats). Implement spatial tiling, data augmentation, and domain adaptation strategies based on rigorous analysis of the underlying data distributions.

### IV. Test-Driven ML Development (TD-MLD)
Core ML components, including custom data loaders, evaluation metrics, and preprocessing transformations, MUST have unit tests. Implement automated integration tests for the full training and inference pipelines using small synthetic or sample datasets to ensure pipeline integrity before large-scale runs.

### V. Reproducible Infrastructure & MLOps
Maintain a fully reproducible environment using Docker Compose for backend services (PostgreSQL, MinIO, MLflow) and `uv` for strict dependency management. Use standardized Makefiles for environment setup, data preparation, and pipeline execution to minimize operational friction and human error.

## Modeling & Data Constraints

- **Backbones**: Prefer Vision Transformers (e.g., DINOv2) via the `timm` library for robust and scalable feature extraction.
- **Domain Adaptation**: Explicitly address the disparity between training and testing distributions through tiling inference and unsupervised learning on LUCAS Cover Photos.
- **Resource Efficiency**: Optimize for high-resolution quadrat images while maintaining efficient training and inference times.

## Development Workflow

- **Environment Consistency**: Always sync dependencies using `uv sync` before starting development.
- **Quality Gates**: Mandatory execution of `pre-commit` hooks (including black, isort, flake8, and vulture) before every commit.
- **Experiment Lifecycle**: Define new experiments in structured scripts or notebooks, log all outputs to the local MLflow instance, and document significant findings.

## Governance

- The Constitution supersedes all other project practices and standards.
- All Pull Requests MUST be reviewed for compliance with these principles.
- Significant changes to the project's architectural direction or core principles require a formal amendment to this document.
- Use the `scripts/` directory and `Makefile` for any repetitive or complex operational tasks to ensure consistency.

**Version**: 1.0.0 | **Ratified**: 2026-04-18 | **Last Amended**: 2026-04-18
