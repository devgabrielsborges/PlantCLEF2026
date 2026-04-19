# Quickstart: DANN Training for PlantCLEF 2026

This guide provides the steps to execute the Domain-Adversarial Neural Network (DANN) training pipeline.

## Prerequisites

1.  **Environment Setup**:
    ```bash
    uv sync
    source .venv/bin/activate
    ```
2.  **Configuration**:
    Copy `.env.example` to `.env` and configure the dataset paths:
    ```bash
    SOURCE_DATA_PATH=/absolute/path/to/single-plant/data
    TARGET_DATA_PATH=/absolute/path/to/quadrat/data
    ```

## Local Execution

Run the training notebook using Papermill for a headless execution. Parameters like `BATCH_SIZE` and `NUM_EPOCHS` will override the `.env` defaults if passed via `-p`.

```bash
uv run papermill notebooks/001-dann-training.ipynb notebooks/001-dann-training-output.ipynb \
    -p BATCH_SIZE 16 \
    -p NUM_EPOCHS 5
```

## Remote Execution (via SSH)

Use the project `Makefile` to execute training on the remote cluster (`chuva`). Ensure `.env` is correctly populated on the remote side.

```bash
ssh chuva "cd dev/PlantCLEF2026 && make run-dann"
```

## Monitoring Results

1.  **MLflow**: Open the MLflow UI to track real-time progress of classification and domain losses.
    ```bash
    mlflow ui --port 5000
    ```
2.  **Visualization**: After training, check the generated `notebooks/001-dann-training-output.ipynb` for the t-SNE plot and classification metrics on both domains. The t-SNE plot (`tsne_overlap.png`) is also logged as an MLflow artifact.
