---
description: "Task list for Unsupervised Domain Adaptation (DANN) implementation"
---

# Tasks: Unsupervised Domain Adaptation (DANN)

**Input**: Design documents from `/specs/001-unsupervised-domain-adaptation/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Unit tests are required for core ML components per project constitution (TD-MLD).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create feature directory structure in `src/models/`, `src/data/`, `src/utils/`, and `tests/unit/`
- [ ] T002 [P] Verify `python-dotenv` and `papermill` are correctly configured in `pyproject.toml`
- [ ] T003 [P] Initialize `.env.example` with DANN-specific parameters (Backbone, Loss weights, Paths)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Implement Gradient Reversal Layer (GRL) as `torch.autograd.Function` in `src/models/dann.py`
- [ ] T005 [P] Implement interleaved Dataset loader (Source + Target) in `src/data/datasets.py`
- [ ] T006 [P] Setup MLflow metric logging schema in `src/utils/metrics.py` (classification vs domain losses)
- [ ] T007 [P] Create unit test for Gradient Reversal Layer in `tests/unit/test_dann.py`
- [ ] T008 [P] Create unit test for interleaved data loading in `tests/unit/test_datasets.py`

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Train Domain-Invariant Classifier (Priority: P1) 🎯 MVP

**Goal**: Train a plant classifier that focuses on plant morphology rather than background context by using DANN.

**Independent Test**: Training loop runs, `loss_classification` decreases on source, and `loss_domain` stabilizes as discriminator accuracy approaches 50%.

### Implementation for User Story 1

- [ ] T009 [P] [US1] Implement DANN Model class with Feature Extractor (DINOv2), Label Predictor, and Domain Discriminator in `src/models/dann.py`
- [ ] T010 [US1] Create training step logic with adversarial loss calculation and GRL alpha scheduling in `notebooks/001-dann-training.ipynb`
- [ ] T011 [US1] Integrate MLflow session initialization and metric logging in `notebooks/001-dann-training.ipynb`
- [ ] T012 [US1] Implement base evaluation loop for source validation set in `notebooks/001-dann-training.ipynb`
- [ ] T013 [P] [US1] Add unit test for DANN model forward pass in `tests/unit/test_dann.py`

**Checkpoint**: User Story 1 is functional - model can be trained with domain adaptation.

---

## Phase 4: User Story 2 - Configure Experiment via Environment (Priority: P2)

**Goal**: Toggle training parameters and dataset paths via `.env` without modifying code.

**Independent Test**: Change `DOMAIN_LOSS_WEIGHT` in `.env` and verify the change is reflected in the notebook's training behavior.

### Implementation for User Story 2

- [ ] T014 [US2] Implement `.env` loading and parameter injection into the training pipeline in `notebooks/001-dann-training.ipynb`
- [ ] T015 [P] [US2] Update `src/data/datasets.py` to use dataset paths from environment variables via `python-dotenv`
- [ ] T016 [US2] Add validation check in notebook to ensure all required `.env` variables are present before training starts

**Checkpoint**: Experiment configuration is decoupled from code.

---

## Phase 5: User Story 3 - Validate Domain Invariance (Priority: P3)

**Goal**: Visualize the feature space of both domains to qualitatively confirm overlap.

**Independent Test**: Notebook generates a 2D projection (t-SNE) showing overlapping clusters of source and target features.

### Implementation for User Story 3

- [ ] T017 [US3] Implement feature extraction and t-SNE projection logic for source/target domains in `notebooks/001-dann-training.ipynb`
- [ ] T018 [US3] Add visualization cells to display domain overlap and example predictions in `notebooks/001-dann-training.ipynb`
- [ ] T019 [US3] Implement final evaluation on the target-domain labeled validation subset in `notebooks/001-dann-training.ipynb`

**Checkpoint**: Domain invariance is visually and quantitatively validated.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T020 [P] Update `quickstart.md` with final `papermill` command examples for DANN training
- [ ] T021 [P] Ensure all code follows project linting standards (black, isort, flake8)
- [ ] T022 Document final adversarial training hyperparameters and results in `specs/001-unsupervised-domain-adaptation/research.md`
- [ ] T023 [P] Run final validation of the full pipeline via a headless `papermill` execution

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup (Phase 1). BLOCKS all user stories.
- **User Stories (Phase 3+)**: All depend on Foundational (Phase 2).
  - US1 (P1) is the primary target.
  - US2 and US3 can proceed after US1 implementation starts.
- **Polish (Phase 6)**: Depends on all user stories.

### Parallel Opportunities

- T002, T003 (Setup)
- T005, T006, T007, T008 (Foundational)
- T009 (US1 Implementation) can start in parallel with training loop structure (T010).
- T015 (US2) and T013 (US1 Tests) are independent.
- Documentation and linting (T020, T021) can happen in parallel at the end.

---

## Parallel Example: Foundational Phase

```bash
# Implement data loader and metrics tracking in parallel
Task: "Implement interleaved Dataset loader (Source + Target) in src/data/datasets.py"
Task: "Setup MLflow metric logging schema in src/utils/metrics.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Setup and Foundational phases.
2. Implement core DANN model and training loop (User Story 1).
3. **STOP and VALIDATE**: Verify that the model trains and domain loss is logged.

### Incremental Delivery

1. **Baseline**: US1 completed (DANN works).
2. **Configuration**: US2 added (Configurable via .env).
3. **Insight**: US3 added (Visual validation of invariance).

---

## Notes

- [P] tasks = different files, no dependencies.
- [Story] label maps task to specific user story for traceability.
- Follow TD-MLD: Ensure tests for GRL and Data Loaders pass before complex training begins.
