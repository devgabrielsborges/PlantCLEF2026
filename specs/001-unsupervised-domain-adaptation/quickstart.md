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

Run the training notebook using Papermill for a headless execution:
```bash
papermill notebooks/001-dann-training.ipynb output.ipynb \
    -p BATCH_SIZE 16 \
    -p NUM_EPOCHS 5
```

## Remote Execution (via SSH)

Use the project `Makefile` or a simple SSH tunnel to execute training on the remote cluster (`chuva`):
```bash
ssh chuva "cd dev/PlantCLEF2026 && make run-dann"
```

## Monitoring Results

1.  **MLflow**: Open the MLflow UI to track real-time progress.
    ```bash
    mlflow ui --port 5000
    ```
2.  **Visualization**: After training, check the generated `output.ipynb` for the t-SNE plot and classification metrics on both domains.
