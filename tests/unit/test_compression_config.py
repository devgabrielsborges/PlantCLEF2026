from pathlib import Path

import pytest

from src.config.compression import CompressionConfig


def test_config_from_env_defaults(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    cfg = CompressionConfig.from_env()
    assert isinstance(cfg.data_dir, Path)
    assert cfg.image_size == 512
    assert cfg.compressai_model == "bmshj2018-factorized"


def test_config_validate_rejects_invalid_jpeg_quality(tmp_path):
    cfg = CompressionConfig(
        data_dir=tmp_path,
        train_image_root=tmp_path / "train",
        train_metadata_path=tmp_path / "train.csv",
        test_image_root=tmp_path / "test",
        test_csv_path=tmp_path / "test.csv",
        output_dir=tmp_path / "out",
        submission_path=tmp_path / "out" / "submission.csv",
        compressed_cache_dir=tmp_path / "cache",
        image_size=512,
        jpeg_normalize=True,
        jpeg_quality=0,
        compression_enabled=True,
        compressai_model="bmshj2018-factorized",
        compressai_quality=3,
        compression_pretrained=True,
        compression_device="cpu",
        train_batch_size=4,
        num_workers=0,
        max_train_samples=10,
        max_val_samples=10,
        learning_rate=1e-3,
        epochs=1,
        backbone_name="resnet18",
        min_score=0.1,
        top_k_tile=2,
        mlflow_experiment_name="test",
    )
    for path in (
        cfg.train_image_root,
        cfg.test_image_root,
    ):
        path.mkdir(parents=True, exist_ok=True)
    cfg.train_metadata_path.write_text("image_id;species_id\n")
    cfg.test_csv_path.write_text("plot_id\n")

    with pytest.raises(ValueError):
        cfg.validate()

