# Environment Variables Schema

This document defines the configuration parameters for the DANN implementation.

## .env Parameters

| Variable Name | Type | Description | Default |
|---------------|------|-------------|---------|
| `BACKBONE` | string | The `timm` model string (e.g., `vit_base_patch14_dinov2.lvd142m`). | `vit_base_patch14_dinov2` |
| `BATCH_SIZE` | int | Batch size per domain (e.g., 32 source + 32 target = 64 total). | `32` |
| `LEARNING_RATE` | float | Base learning rate for the training process. | `0.001` |
| `BACKBONE_LR_FACTOR` | float | Multiplier for the backbone's learning rate (e.g., 0.1 means $LR/10$). | `0.01` |
| `DOMAIN_LOSS_WEIGHT` | float | Initial weight for the adversarial domain loss ($\lambda$). | `1.0` |
| `NUM_EPOCHS` | int | Total number of training epochs. | `20` |
| `SOURCE_DATA_PATH` | string | Absolute path to the labeled single-plant dataset. | REQUIRED |
| `TARGET_DATA_PATH` | string | Absolute path to the unlabeled quadrat dataset. | REQUIRED |
| `MLFLOW_TRACKING_URI` | string | URI for the MLflow server. | `http://localhost:5000` |
| `USE_AMP` | bool | Whether to use Automatic Mixed Precision for faster training. | `true` |
