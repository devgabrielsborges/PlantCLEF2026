from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


@dataclass(slots=True)
class CompressionConfig:
    data_dir: Path
    train_image_root: Path
    train_metadata_path: Path
    test_image_root: Path
    test_csv_path: Path
    output_dir: Path
    submission_path: Path
    compressed_cache_dir: Path
    image_size: int
    jpeg_normalize: bool
    jpeg_quality: int
    compression_enabled: bool
    compressai_model: str
    compressai_quality: int
    compression_pretrained: bool
    compression_device: str
    train_batch_size: int
    num_workers: int
    max_train_samples: int
    max_val_samples: int
    learning_rate: float
    epochs: int
    backbone_name: str
    min_score: float
    top_k_tile: int
    mlflow_experiment_name: str

    @classmethod
    def from_env(cls) -> "CompressionConfig":
        load_dotenv(override=True)
        data_dir = Path(os.getenv("DATA_DIR", "data")).expanduser()
        train_image_root_default = _first_existing(
            [
                data_dir / "train_images",
                data_dir / "images_max_side_800",
                data_dir / "labeled",
                data_dir / "PlantCLEF2024_single_plant_training_images",
                data_dir
                / "PlantCLEF2024_single_plant_training_images"
                / "PlantCLEF2024_single_plant_training_images",
            ]
        ) or (data_dir / "train_images")
        test_image_root_default = _first_existing(
            [
                data_dir / "PlantCLEF2025_test_images",
                data_dir / "PlantCLEF2025_test_images" / "PlantCLEF2025_test_images",
                data_dir / "test_images",
            ]
        ) or (data_dir / "PlantCLEF2025_test_images")
        output_dir = Path(
            os.getenv("COMPRESSION_OUTPUT_DIR", str(data_dir / "outputs"))
        ).expanduser()
        submission_path = Path(
            os.getenv("COMPRESSION_SUBMISSION_PATH", str(output_dir / "submission.csv"))
        ).expanduser()
        compressed_cache_dir = Path(
            os.getenv(
                "COMPRESSED_CACHE_DIR",
                str(data_dir / "cache" / "compressed_quadrats"),
            )
        ).expanduser()

        return cls(
            data_dir=data_dir,
            train_image_root=Path(
                os.getenv("TRAIN_IMAGE_ROOT", str(train_image_root_default))
            ).expanduser(),
            train_metadata_path=Path(
                os.getenv(
                    "TRAIN_METADATA_PATH",
                    str(data_dir / "PlantCLEF2024_single_plant_training_metadata.csv"),
                )
            ).expanduser(),
            test_image_root=Path(
                os.getenv("TEST_IMAGE_ROOT", str(test_image_root_default))
            ).expanduser(),
            test_csv_path=Path(
                os.getenv("TEST_CSV_PATH", str(data_dir / "PlantCLEF2025_test.csv"))
            ).expanduser(),
            output_dir=output_dir,
            submission_path=submission_path,
            compressed_cache_dir=compressed_cache_dir,
            image_size=_get_int("IMAGE_SIZE", 512),
            jpeg_normalize=_get_bool("JPEG_NORMALIZE", True),
            jpeg_quality=_get_int("JPEG_QUALITY", 95),
            compression_enabled=_get_bool("COMPRESSION_ENABLED", True),
            compressai_model=os.getenv(
                "COMPRESSAI_MODEL", "bmshj2018-factorized"
            ).strip(),
            compressai_quality=_get_int("COMPRESSAI_QUALITY", 3),
            compression_pretrained=_get_bool("COMPRESSION_PRETRAINED", True),
            compression_device=os.getenv("COMPRESSION_DEVICE", "cuda").strip(),
            train_batch_size=_get_int("TRAIN_BATCH_SIZE", 8),
            num_workers=_get_int("NUM_WORKERS", 4),
            max_train_samples=_get_int("MAX_TRAIN_SAMPLES", 10000),
            max_val_samples=_get_int("MAX_VAL_SAMPLES", 1000),
            learning_rate=_get_float("LEARNING_RATE", 3e-4),
            epochs=_get_int("EPOCHS", 2),
            backbone_name=os.getenv("BACKBONE_NAME", "vit_base_patch14_dinov2"),
            min_score=_get_float("MIN_SCORE", 0.1),
            top_k_tile=_get_int("TOP_K_TILE", 2),
            mlflow_experiment_name=os.getenv(
                "EXPERIMENT_NAME", "compression-baseline"
            ).strip(),
        )

    def validate(self) -> None:
        missing = []
        for path in (
            self.train_image_root,
            self.train_metadata_path,
            self.test_image_root,
            self.test_csv_path,
        ):
            if not path.exists():
                missing.append(str(path))
        if missing:
            joined = "\n".join(f"- {item}" for item in missing)
            raise FileNotFoundError(
                "Required paths for compression baseline are missing:\n" + joined
            )

        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG_QUALITY must be between 1 and 100.")
        if not 1 <= self.compressai_quality <= 8:
            raise ValueError("COMPRESSAI_QUALITY must be between 1 and 8.")
        if self.image_size <= 0:
            raise ValueError("IMAGE_SIZE must be positive.")
        if self.train_batch_size <= 0:
            raise ValueError("TRAIN_BATCH_SIZE must be positive.")
        if self.epochs <= 0:
            raise ValueError("EPOCHS must be positive.")
        if self.top_k_tile <= 0:
            raise ValueError("TOP_K_TILE must be positive.")

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compressed_cache_dir.mkdir(parents=True, exist_ok=True)
        self.submission_path.parent.mkdir(parents=True, exist_ok=True)
