# MLflow Metrics & Experiment Tracking

This document defines the standard metrics and parameters to be logged in MLflow.

## Logged Parameters

- `feature_extractor_backbone`
- `domain_adaptation_method` (e.g., "DANN")
- `adversarial_loss_weight`
- `optimizer_type`
- `scheduler_type`

## Logged Metrics

### Training Phase
- `train/loss/classification`: Loss from the plant species predictor.
- `train/loss/domain_source`: Discriminator loss on source domain images.
- `train/loss/domain_target`: Discriminator loss on target domain images.
- `train/accuracy/species`: Classification accuracy on the source training set.
- `train/accuracy/domain`: Discriminator accuracy (should approach 0.5 over time).

### Validation Phase (Source)
- `val/source/accuracy/top1`
- `val/source/accuracy/top5`
- `val/source/macro_f1`

### Validation Phase (Target - Labeled Subset)
- `val/target/accuracy/top1`
- `val/target/accuracy/top5`
- `val/target/macro_f1`

## Artifacts

- `model/checkpoints`: Saved model weights for each epoch.
- `plots/tsne_feature_space`: 2D visualization of the latent domain overlap.
- `output_notebook.ipynb`: The fully executed results notebook from Papermill.
