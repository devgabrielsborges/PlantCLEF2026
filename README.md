# PlantCLEF 2026 @ LifeCLEF & CVPR-FGVC

Identify Multi-species Plants in Images of Vegetation Plots.
This repository contains the baseline code, MLflow tracking infrastructure, and experiment pipeline for the [PlantCLEF 2026 Kaggle Competition](https://www.kaggle.com/competitions/plantclef-2026).

## Competition Overview
The PlantCLEF 2026 challenge focuses on predicting plant species present in high-resolution vegetation plot images (quadrats). 

**The Challenge (Domain Shift)**
- **Training Data:** ~1.4 million labeled images of single plants covering 7.8k species (from Pl@ntNet & GBIF).
- **Test Data:** Complex, high-resolution vegetation quadrat scenes where multiple plant species overlap naturally.
- **Bonus Data:** ~212,000 unannotated high-resolution "pseudo-quadrat" images (LUCAS Cover Photos) provided for unsupervised/self-supervised domain adaptation.

**Evaluation Metric**
Submissions are evaluated using the **Macro-averaged F1 Score per sample (quadrat)**, designed to balance precision and recall for every image individually, ensuring models don't over-predict species (low precision) or miss them (low recall).

## Stack and Architecture
- **Computer Vision:** PyTorch, `timm` (Vision Transformers / DINOv2).
- **Tracking & MLOps:** MLflow (logging step-metrics, batch times, model architectures, and tracking output artifacts).
- **Infrastructure:** Docker Compose (PostgreSQL for backend store, MinIO for S3-compatible artifact storage).
- **Environment:** Python managed via `uv`.

## Project Structure
```text
├── data/                                 # (Not in VC) Kaggle datasets & split outputs
├── notebooks/
│   ├── baseline.ipynb                    # PyTorch ViT/DINOv2 Tiling Inference with MLflow
│   ├── compression_baseline.ipynb        # Compression-aware end-to-end baseline (train + inference)
│   ├── official-notebook-tiling-inference.ipynb
│   └── official-notebook-tiling-inference_out.ipynb
├── scripts/
│   ├── generate_val_split.py             # Script to split training metadata into train/val
│   └── up.sh                             # Script to start docker-compose
├── src/
│   └── config/             
│       └── mlflow_init.py                # Local MLflow & MinIO connection setup
├── docker-compose.yml                    # Local MLOps infrastructure
├── .pre-commit-config.yaml               # Code formatting and linting config
├── Makefile                              # Command shortcuts for Docker & Data Prep
├── pyproject.toml                        # uv dependencies & project scripts
└── README.md
```

## Quick Start

### 1. Start the MLflow Infrastructure
Spin up PostgreSQL, MinIO, and the MLflow dashboard locally:
```bash
make up
```
| Service | URL | Credentials |
|---|---|---|
| MLflow UI | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | `minioadmin` / `minioadmin` |

### 2. Configure Environment
Ensure you have the required `.env` file or export your environment variables for MLflow to connect to MinIO correctly:
```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"
```

### 3. Download and Prepare Data
You can download the metadata files via the Kaggle API and generate a local 10% validation ground-truth split (to measure the Macro-F1 score locally):
```bash
make init
```
This runs `make download` pulling directly from Kaggle and natively unzipping the files into `data/`, then triggers `make generate-val` (running `scripts/generate_val_split.py`).

> **Note:** Requires Kaggle API credentials configured at `~/.kaggle/kaggle.json`.

### 4. Run the Baseline
Open `notebooks/baseline.ipynb` to execute the spatial tiling inference on the test set using the DINOv2 vision transformer backbone. The PyTorch model, validation metrics (F1, Precision, Recall, ROC AUC), and parameter tracking will automatically be routed to your local MLflow instance.

To run the compression-aware pipeline notebook in batch mode:
```bash
make run-compression-baseline
```

### 5. Tear down
```bash
make down     # stop services
make clean    # stop services and delete volumes
```

## Development
This project uses `uv` for dependency management and `pre-commit` for code quality checks (`vulture`, `autoflake`, `black`, `isort`).
```bash
# Sync dependencies
uv sync

# Install and run pre-commit hooks
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Authors
- [devgabrielsborges](https://github.com/devgabrielsborges)
- [JuanLir4](https://github.com/JuanLir4)
