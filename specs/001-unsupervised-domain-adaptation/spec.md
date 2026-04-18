# Feature Specification: Unsupervised Domain Adaptation for Plant Classification

**Feature Branch**: `001-unsupervised-domain-adaptation`  
**Created**: April 18, 2026  
**Status**: Draft  
**Input**: User description: "train a model to actively ignore the background difference between a single-plant photo and a dense quadrat photo. Technique: Unsupervised Domain Adaptation (e.g., Domain-Adversarial Neural Networks - DANN). Workflow: As you train your model to classify single plants, you simultaneously feed it the unannotated quadrat images and train a "domain discriminator." The feature extractor is penalized if the discriminator can tell whether an image came from the single-plant set or the quadrat set. This forces the model to learn features that work equally well on both. . create a notebook for it, and configurable things via .env and src/"

## User Scenarios & Testing *(mandatory)*

### Clarifications

### Session 2026-04-18
- **Q**: How should the papermill execution be integrated with the environment configuration for the DANN training? → **A**: Notebook loads .env internally (e.g., via python-dotenv).
- **Q**: Should the dataset paths be defined as absolute paths on the remote server or relative? → **A**: Absolute remote paths (e.g., /home/ppgec/Documentos/GabrielBorges/PlantCLEF2026/data/...).
- **Q**: Which library should be used for loading environment variables? → **A**: python-dotenv.
- **Q**: Should the system standardize on MLflow for experiment tracking? → **A**: Yes, use the existing MLflow setup.
- **Q**: Confirming the core task and competition alignment. → **A**: Focus on Plant Classification for PlantCLEF 2026 (Computer Vision).

### User Story 1 - Train Domain-Invariant Classifier (Priority: P1)

As a researcher, I want to train a plant classifier that focuses on plant morphology rather than background context (e.g., soil vs. dense vegetation), so that the model performs reliably on diverse field images (quadrats) despite being trained primarily on clean single-plant photos.

**Why this priority**: This is the core functional requirement. Without this, the model will likely overfit to the simple backgrounds of single-plant photos.

**Independent Test**: Can be tested by training the model using the DANN workflow and verifying that the "domain loss" decreases/stabilizes while classification accuracy on the source domain remains high.

**Acceptance Scenarios**:

1. **Given** a labeled dataset of single-plant photos and an unlabeled dataset of quadrat photos, **When** the DANN training pipeline is executed, **Then** the system produces a feature extractor that is "confusing" to the domain discriminator.
2. **Given** a trained model, **When** evaluating on a hold-out set of single-plant photos, **Then** the classification accuracy is within 5% of a baseline non-DANN model.

---

### User Story 2 - Configure Experiment via Environment (Priority: P2)

As an engineer, I want to toggle domain adaptation parameters (like loss weights) and dataset paths via configuration files and environment variables, so that I can run hyperparameter sweeps without modifying the core notebook code.

**Why this priority**: Ensures reproducibility and ease of experimentation across different environments.

**Independent Test**: Change a value in `.env` (e.g., `DOMAIN_LOSS_WEIGHT`) and verify that the training loop uses the new value without code changes.

**Acceptance Scenarios**:

1. **Given** a `.env` file with training parameters, **When** the training notebook is launched, **Then** it correctly overrides default values with the environment settings.
2. **Given** a configuration module in `src/`, **When** the notebook imports parameters, **Then** all paths and hyperparameters are centralized and documented.

---

### User Story 3 - Validate Domain Invariance (Priority: P3)

As a researcher, I want to visualize the feature space of both domains, so that I can qualitatively confirm that single-plant and quadrat features are overlapping.

**Why this priority**: Provides visual proof that the UDA technique is working as intended beyond just accuracy metrics.

**Independent Test**: Run a visualization step (e.g., t-SNE) on features extracted from both domains and observe the degree of cluster overlap.

**Acceptance Scenarios**:

1. **Given** a trained feature extractor, **When** extracting features for 500 images from each domain, **Then** a 2D projection shows significant overlap between the two domains.

### Edge Cases

- **Zero Overlap**: What happens when the quadrat photos contain species not present in the single-plant labeled set?
- **Domain Dominance**: How does the system handle a situation where the source domain is significantly larger than the target domain?
- **Discriminator Overfitting**: How to handle cases where the discriminator becomes too strong too early, preventing the feature extractor from learning?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST implement a DANN architecture consisting of a Feature Extractor, a Label Predictor, and a Domain Discriminator.
- **FR-002**: System MUST support training with labeled data from the source domain and unlabeled data from the target domain (e.g., bonus dataset) simultaneously.
- **FR-003**: System MUST implement a mechanism to penalize the feature extractor based on discriminator performance (e.g., adversarial loss).
- **FR-004**: System MUST allow configuration of the domain adaptation loss weight via external environment settings loaded internally.
- **FR-005**: System MUST provide an interactive execution environment (e.g., a workbook or notebook) that demonstrates the full training and evaluation workflow, compatible with papermill execution.
- **FR-006**: System MUST use a configurable backbone architecture defined in the project configuration.
- **FR-007**: System MUST log both classification loss and domain loss to MLflow for monitoring and experiment tracking.
- **FR-008**: All remote execution MUST be performed via SSH commands (`ssh chuva + command`).
- **FR-009**: Pre-commit checks MUST be executed successfully before any commits are finalized.

### Key Entities *(include if feature involves data)*

- **Source Image**: Labeled plant image with associated species classification.
- **Target Image**: Unlabeled plant image with complex background.
- **Domain Label**: Binary indicator used by the discriminator to distinguish between datasets.
- **Feature Vector**: The latent representation produced by the shared feature extractor.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Model achieves a measurable improvement (at least 10% relative) in classification accuracy on a target-domain validation set compared to a baseline model trained only on source-domain data.
- **SC-002**: Domain discriminator accuracy approaches 50% (random guessing) for the final feature extractor, indicating domain-invariant features.
- **SC-003**: Training process completes without errors for a dataset of at least 15,000 total images.
- **SC-004**: All training hyperparameters can be modified via configuration settings without requiring changes to the execution scripts.

## Assumptions

- **Labeled Data**: A significant labeled dataset of "single-plant" photos is available for the species classification task.
- **Target Domain Volume**: There is enough unlabeled quadrat data to allow the discriminator to learn domain-specific features.
- **Shared Classes**: The species present in the quadrat photos are a subset of the species present in the single-plant labeled set.
- **Hardware**: Training will occur on a system with CUDA-compatible GPU acceleration.
- **Validation**: A small, manually labeled subset of quadrat images will be used for final performance validation.
