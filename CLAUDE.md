# Semantic-ID Generation Project Context

This document serves as a context guide for AI agents and developers working on this project. It outlines the architecture, workflow, developer commands, and style guidelines of the Semantic-ID Generation system.

## Project Overview

**Goal:** Generate "Semantic IDs" for items in a recommender system. These IDs are tuples of discrete integers that hierarchically represent the item's content/embedding, used for "Generative Retrieval".

**Core Architecture:**
- **RQ-VAE:** A Variational Autoencoder that uses Residual Quantization (multiple codebooks) to encode items into a sequence of discrete IDs. Supports **STE** and **Gumbel Softmax** quantization, with **L2** and **Cosine** distance metrics.
- **FSQ:** Finite Scalar Quantization, an alternative to VQ that uses predefined discrete boundary mapping instead of learned codebooks, enabling gradient bypass without auxiliary losses.
- **RQ-KMeans:** Lightweight alternative: residual K-Means clustering without neural training. No gradient, directly fits on embeddings.
- **TIGER:** T5-based Seq2Seq model for generative recommendation using SIDs.
- **SASRec:** Self-Attentive Sequential Recommendation baseline (decoder-only Transformer on raw item IDs).

---

## Environment Setup

Before running any commands or tests, ensure you are using the correct Conda environment defined in `environment.yml`:
```bash
conda activate semantic-id-generation
```
Alternatively, prefix your commands with `conda run -n semantic-id-generation `.

## Developer Commands

### 1. Training & ID Generation Commands (`main.py`)
All main workflows are orchestrated via `main.py`. Any config parameter can be overridden at the end of the command using dot-list format (e.g., `model.latent_dimension=128`).

* **Train RQ-VAE (Item Encoder):**
  ```bash
  python main.py train-rqvae --config config/<config_file>.yaml [override.key=value ...]
  ```
* **Train RQ-KMeans (Alternative SID method):**
  ```bash
  python main.py train-rqkmeans --config config/<config_file>.yaml [override.key=value ...]
  ```
* **Generate Semantic IDs (from RQ-VAE):**
  ```bash
  python main.py generate-ids --config config/<config_file>.yaml --model_path models/<model_id>.pt [--output_path <path>] [--batch_size <size>]
  ```
* **Train Seq2Seq (TIGER):**
  ```bash
  python main.py train-seq2seq --config config/<config_file>.yaml --semids outputs/semids.pt [--resume <ckpt>] [--warmup_steps N] [override.key=value ...]
  ```
* **Test Seq2Seq (TIGER):**
  ```bash
  python main.py test-seq2seq --config config/<config_file>.yaml --semids outputs/semids.pt --model_path models/<tiger_id>.pt [override.key=value ...]
  ```
* **Train SASRec Baseline:**
  ```bash
  python main.py train-sasrec --config config/<config_file>.yaml [--resume <ckpt>] [override.key=value ...]
  ```
* **Test SASRec Baseline:**
  ```bash
  python main.py test-sasrec --config config/<config_file>.yaml --model_path models/<sasrec_id>.pt [override.key=value ...]
  ```

### 2. Other Executable Scripts
* **Hyperparameter Tuning (RQ-VAE):**
  ```bash
  python hyperparameter.py --config config/<config_file>.yaml [--search_type random/grid] [--num_trials N]
  ```
* **Evaluate Naive Baselines:**
  ```bash
  python test_naive_baselines.py --config config/<config_file>.yaml --semids outputs/semids.pt
  ```

### 3. Test Commands
Tests are managed using `pytest`. Configuration options are defined in `pyproject.toml`.
* **Run all tests:** `pytest`
* **Run fast tests (exclude slow tests):** `pytest -m "not slow"`
* **Run unit tests:** `pytest tests/unit/`
* **Run integration tests:** `pytest tests/integration/`
* **Run a specific test file:** `pytest tests/unit/data/test_augmentation.py`

---

## Code Style & Guidelines

* **Formatting:** Follow standard PEP 8 guidelines. Use consistent 4-space indentation.
* **Naming Conventions:**
  * Files, variables, and functions: `snake_case` (e.g., `train_seq2seq.py`, `learning_rate`).
  * Classes: `CamelCase` (e.g., `RQ_VAE`, `TigerSeq2Seq`).
* **Imports Order:**
  1. Standard library imports (e.g., `os`, `logging`, `json`, `argparse`).
  2. Third-party library imports (e.g., `torch`, `omegaconf`, `wandb`, `tqdm`).
  3. Local module imports (e.g., `from modules.rqvae import RQ_VAE`).
* **Logging & Error Handling:** Use Python's built-in `logging` module (`logger.info`, `logger.error`) rather than `print()` statements for diagnostic messages.
* **Resuming Runs:** Resuming is highly configurable. When resuming training, verify if configuration parameters like `resume_optimizer` or `early_stopping` need to be modified in the overrides or configuration files.

---

## Directory Structure

- **Root Scripts:** `main.py`, `train_rq_vae.py`, `train_rqkmeans.py`, `train_seq2seq.py`, `test_seq2seq.py`, `train_sasrec.py`, `test_sasrec.py`, `generate_semids.py`, `test_naive_baselines.py`, `hyperparameter.py`.
- **`modules/`**: Package-based model implementations.
  - **`rqvae/`**: Core RQ-VAE SID generation.
    - `model.py`: Main `RQ_VAE` class.
    - `networks.py`: Encoder & Decoder (MLP architectures).
    - `quantization.py`: RVQ logic and `QuantizeLoss`.
    - `scheduler.py`: Temperature annealing strategies for Gumbel-Softmax.
  - **`rqkmeans/`**: RQ-KMeans (non-neural SID baseline).
    - `model.py`: `RQKMeans` class.
    - `kmeans.py`: `BatchKMeans` implementation.
  - **`recommender/`**: Generative Recommendation logic.
    - `seq2seq.py`: `TigerSeq2Seq` (T5-based TIGER).
  - **`sasrec/`**: Sequential recommendation baseline.
    - `model.py`: `SASRec` decoder-only Transformer.
- **`schemas/`**: Typed output schemas (`rq_vae.py`, `quantization.py`).
- **`notebooks/`**: Marimo interactive analysis notebooks (`sid_analysis.py`, `analyze_sid_quality.py`, `check_utilization.py`, etc.).
- **`experiments/`**: Experimental seq2seq variants (`seq2seq_v2.py`, `seq2seq_v3_grid.py`).
- **`plotting/`**: Plotting utilities (`plot.py`).
- **`data/`**: Data loading and preprocessing (`amazon_data.py`, `ml1m.py`, `loader.py`, `factory.py`, `schemas.py`, `sequence.py`, `preprocessing.py`).
- **`utils/`**: Helper scripts (`wandb.py`, `metrics.py`, `model_id_generation.py`, `sid_evaluation.py`).
- **`tests/`**: Unit and integration tests (`tests/unit/`, `tests/integration/`).
- **`config/`**: YAML configuration files.
- **`models/`**: Trained model checkpoints (`.pt`).
- **`outputs/`**: Generated Semantic IDs (`semids.pt`).

---

## Key Technical Details

- **Input:** High-dimensional vectors (e.g., T5 text embeddings).
- **Latent Space:** Hierarchical discrete codes (e.g., `[45, 12, 255]`).
- **Loss:** Reconstruction loss uses `mean` reduction with L2-normalization on both input and reconstruction for semantic consistency on the unit hypersphere.
- **Decoder:** No `Sigmoid` output activation to support full dynamic range.
- **SID Evaluation:** `utils/sid_evaluation.py` provides `compute_utilisation` (per-layer codebook coverage and perplexity).
- **Schemas:** `schemas/rq_vae.py` defines `RqVaeOutput` and `RqVaeComputedLosses` NamedTuples with per-layer diagnostics (layer coverages, entropies, residual norms).

## TIGER / Seq2Seq Architecture

The `modules/recommender/seq2seq.py` file implements the **TIGER** architecture using a **T5 backbone**.
- **Generative Retrieval:** Predicts next item SIDs autoregressively based on user history.
- **Unified Vocabulary:** Manages offsets to map User IDs, Semantic IDs, and Collision Tokens into a single embedding space.
- **Continuous Generation:** No EOS token — flexible Recommendation@K via item-based generation.

## SASRec Baseline

The `modules/sasrec/model.py` file implements **SASRec** as a non-generative retrieval baseline.
- Decoder-only Transformer on raw item ID sequences (pre-norm, causal self-attention).
- Shared item embedding for prediction (BCE loss with per-position negative sampling).
