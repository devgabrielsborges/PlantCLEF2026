.PHONY: help up down restart logs clean init download preprocess generate-val split-data run-dann run-compression-baseline

help:
	@echo "Infrastructure"
	@echo "  make up             Start Postgres + MinIO + MLflow"
	@echo "  make down           Stop all services"
	@echo "  make restart        Restart all services"
	@echo "  make logs           Tail service logs"
	@echo "  make clean          Stop services and delete volumes"
	@echo ""
	@echo "Data & Preprocessing"
	@echo "  make init           Download data and generate validation split"
	@echo "  make download       Download main competition metadata and datasets"
	@echo "  make generate-val   Generate local validation split from train metadata"
	@echo ""
	@echo "Modeling"
	@echo "  make run-dann       Run DANN training via papermill"
	@echo "  make run-compression-baseline Run compression-aware baseline via papermill"

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

init: download generate-val

download:
	@echo "Downloading PlantCLEF 2026 metadata..."
	uvx kaggle competitions download -c plantclef-2026 -f PlantCLEF2024_single_plant_training_metadata.csv -p data/
	uvx kaggle competitions download -c plantclef-2026 -f PlantCLEF2025_test.csv -p data/
	@echo "Extracting datasets..."
	unzip -n data/PlantCLEF2024_single_plant_training_metadata.csv.zip -d data/ || true
	unzip -n data/PlantCLEF2025_test.csv.zip -d data/ || true

generate-val:
	@echo "Generating 10% validation ground-truth split..."
	uv run scripts/generate_val_split.py

run-dann:
	@echo "Running DANN training pipeline..."
	uv run papermill notebooks/001-dann-training.ipynb notebooks/001-dann-training-output.ipynb

run-compression-baseline:
	@echo "Running compression baseline pipeline..."
	uv run papermill notebooks/compression_baseline.ipynb notebooks/compression_baseline-output.ipynb

clean:
	docker compose down -v
