# Implementation Plan: Unsupervised Domain Adaptation (DANN)

**Branch**: `001-unsupervised-domain-adaptation` | **Date**: April 18, 2026 | **Spec**: [specs/001-unsupervised-domain-adaptation/spec.md](spec.md)
**Input**: Feature specification from `/specs/001-unsupervised-domain-adaptation/spec.md`

## Summary
The goal is to implement an Unsupervised Domain Adaptation (UDA) strategy using a Domain-Adversarial Neural Network (DANN) to handle background differences between single-plant photos (source) and dense quadrat photos (target). We will simultaneously train the model to classify single plants and train a "domain discriminator" to distinguish between source and target images, with the feature extractor penalized for discriminating domain identity. This promotes domain-invariant features. Configuration will be handled via `.env` and integrated into an interactive notebook workflow (compatible with `papermill`).

## Technical Context

**Language/Version**: Python 3.11+ (from `pyproject.toml`)  
**Primary Dependencies**: PyTorch (>=2.10), `timm` (>=1.0.12), `mlflow` (3.9.0), `python-dotenv`, `papermill`  
**Storage**: Local files for data, MLflow for experiment tracking, PostgreSQL/MinIO for backend services  
**Testing**: `pytest` (implied by constitution), unit tests for custom loaders/metrics  
**Target Platform**: Linux server (SSH via `chuva`)  
**Project Type**: ML Training Pipeline & Experimentation Notebook  
**Performance Goals**: >10% relative improvement in target validation accuracy over non-DANN baseline  
**Constraints**: Configurable backbone (preference for DINOv2 via `timm`), Domain-Adversarial Loss Weight via `.env`  
**Scale/Scope**: ~15,000 images, domain adaptation across heterogeneous background contexts  

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Clean Code & Architecture**: ML logic must be decoupled from tracking and specific frameworks. Custom `DANN` model and loss modules will reside in `src/`.
- **Scientific Methodology**: MLflow mandatory for reproducibility. All experiments must be logged.
- **Experiment Rigor**: Standard validation splits required.
- **TD-MLD**: Custom layers (e.g., GRL) and training loops must have unit tests.
- **Reproducible Infrastructure**: Use `uv`, Makefiles, and Docker-compose as defined in `constitution.md`.

## Project Structure

### Documentation (this feature)

```text
specs/001-unsupervised-domain-adaptation/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # (By /speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── config/
│   └── mlflow_init.py   # Existing MLflow configuration
├── data/
│   ├── datasets.py      # Source/Target combined loader
│   └── metadata.py
├── models/
│   ├── __init__.py
│   └── dann.py          # DANN architecture (Feature Extractor, Predictor, Discriminator)
└── utils/
    ├── metrics.py       # Custom domain loss monitoring
    └── ...
notebooks/
└── 001-dann-training.ipynb  # Primary experimentation/training notebook
```

**Structure Decision**: Option 1: Single project. We will extend the existing `src/` hierarchy to include DANN-specific modules and update `notebooks/`.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None      |            |                                     |
