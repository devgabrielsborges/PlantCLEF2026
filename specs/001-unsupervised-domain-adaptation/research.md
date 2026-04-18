# Research: Unsupervised Domain Adaptation (DANN)

This document outlines the technical research and architectural decisions for implementing Domain-Adversarial Neural Networks (DANN) for the PlantCLEF 2026 competition.

## 1. Gradient Reversal Layer (GRL) Implementation

### Decision
Implement the GRL as a custom `torch.autograd.Function`.

### Rationale
DANN requires the gradient from the domain discriminator to be multiplied by $-\lambda$ when backpropagating through the feature extractor. A custom `autograd.Function` is the idiomatic PyTorch way to achieve this, as it allows the forward pass to act as an identity while the backward pass performs the gradient reversal.

### Implementation Detail
```python
from torch.autograd import Function

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)
```

### Alternatives
- **Manual Gradient Negation**: Manually negating gradients after `loss.backward()` but before `optimizer.step()`. This is prone to errors with multiple optimizers and complex architectures.
- **Hook-based Approach**: Using `register_backward_hook`. This is generally discouraged for modern PyTorch (preferring `register_full_backward_hook`), and can be harder to debug than a clean functional interface.

---

## 2. DANN with Vision Transformer (DINOv2) Backbone

### Decision
Use `timm`'s `vit_base_patch14_dinov2` as the feature extractor, utilizing the `[CLS]` token for both classification and domain discrimination.

### Rationale
DINOv2 features are highly robust to background noise. The `[CLS]` token provides a global summary of the image, which is suitable for both identifying the plant species and identifying the global domain (single-plant vs. quadrat).
- **Learning Rates**: The backbone should use a significantly lower learning rate (e.g., $10^{-5}$) compared to the task-specific heads ($10^{-3}$).
- **Lambda Scheduling**: Gradually increase the reversal strength $\lambda$ from 0 to 1 during training to prevent the discriminator from destabilizing the feature extractor early on.

### Alternatives
- **Global Average Pooling (GAP)**: Averaging all patch tokens. While effective for some tasks, DINOv2 is specifically optimized for `[CLS]` token usage in classification.
- **Multi-scale Feature Fusion**: Concatenating features from different layers. This adds complexity and may lead to overfitting given the limited labeled target data.

---

## 3. Interleaved Dataset Loading

### Decision
Use a zipped iterator with `itertools.cycle` to ensure every training batch contains both source and target domain data.

### Rationale
DANN training requires a simultaneous forward pass for source data (calculating class and domain loss) and target data (calculating domain loss). Zipping the loaders ensures the model sees both domains in every optimization step, which is crucial for adversarial stability.

### Implementation Detail
```python
import itertools

source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)

# Loop over the longer/primary dataset
for (source_images, source_labels), (target_images, _) in zip(source_loader, itertools.cycle(target_loader)):
    # ... training step ...
```

### Alternatives
- **Concatenated Datasets**: Using a single loader with a "domain" label. This makes it harder to ensure equal representation of source and target in every batch, especially if dataset sizes are imbalanced.
- **Random Sampling**: Randomly picking a batch from either source or target. This leads to high variance in the adversarial gradient.

---

## 4. MLflow Multi-Loss Tracking

### Decision
Log `loss_classification`, `loss_domain_source`, `loss_domain_target`, and `total_adversarial_loss` separately using `mlflow.log_metrics`.

### Rationale
Monitoring the individual components of the adversarial loss is essential for diagnosing "discriminator collapse" or "feature extractor dominance."
- **Metric Grouping**: Group metrics by domain in the MLflow UI (e.g., `train/source/loss_class`, `train/target/loss_domain`).
- **Step Coordination**: Ensure all metrics for a single optimization step are logged with the same `step` index.

### Alternatives
- **Single Total Loss**: Logging only the scalar sum of losses. This hides the dynamics of the adversarial game.

---

## 5. Papermill and Environment Management

### Decision
Use `python-dotenv` for local and remote environment variable management. Pass sensitive parameters (like S3 keys or DB passwords) via shell environment variables when calling Papermill via SSH.

### Rationale
- **Security**: Papermill records all input parameters in the output notebook's metadata. Storing secrets in `.env` files (ignored by git) or injecting them via the shell ensures they are not leaked into the project history.
- **SSH Workflow**: Use a wrapper script or a `Makefile` to ensure the remote server has the correct environment context before execution.

### Implementation Detail
```bash
# Example remote execution via SSH
ssh user@remote "export $(cat .env | xargs) && papermill notebook.ipynb output.ipynb"
```

### Alternatives
- **Papermill Parameters for Secrets**: Insecure.
- **Hardcoded Config Files**: Breaks portability and security.
