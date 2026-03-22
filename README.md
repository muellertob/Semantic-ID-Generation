# Semantic-ID Generation using RQ-VAE

Semantic ID Generation based on RQ-VAE Architecture for Recommendation purposes, implementing the pipeline described in "Recommender Systems with Generative Retrieval" (TIGER) by Rajput et al. and "Better Generalization with Semantic IDs" by Singh et al.

This project is a fork of [justinhangoebl/Semantic-ID-Generation](https://github.com/justinhangoebl/Semantic-ID-Generation), which itself was inspired by [EdoardoBotta/RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender). This fork extends Justin's implementation with a full TIGER-style generative retrieval pipeline (seq2seq training and evaluation).

Items in the corpus are mapped to tuples of Semantic IDs by training an RQ-VAE, see figure below.

![proposed Architecture for RQ-VAE from Tiger](./assets/RQ-VAE-Architecture-Rajput-TIGER.jpg)

## Pipeline Overview

The system follows a three-stage pipeline:

1. **Train RQ-VAE** — Learn discrete hierarchical Semantic IDs from item embeddings.
2. **Generate Semantic IDs** — Encode all items into SID tuples using the trained RQ-VAE.
3. **Train / Test TIGER** — Train a T5-based seq2seq model to predict the next item's SID from user history.

## Features

- **Unified CLI** (`main.py`) for all pipeline stages
- **Multiple Quantization Methods**: STE and Gumbel Softmax
- **Distance Metrics**: L2 and Cosine Similarity (cosine preferred for text embeddings)
- **Temperature Annealing**: Automatic scheduling for Gumbel Softmax training
- **TIGER Generative Retrieval**: T5-based seq2seq model with unified vocabulary for User IDs and SIDs
- **Flexible Configuration**: OmegaConf YAML-based configuration

## Installation

```bash
git clone <repository-url>
cd Semantic-ID-Generation
conda env create -f environment.yml
conda activate semantic-id-generation
```

## Usage

All pipeline stages are controlled via `main.py`.

### 1. Train RQ-VAE

```bash
python main.py train-rqvae --config config/config_amazon_tiger.yaml
```

Trains the RQ-VAE to reconstruct item embeddings and learn discrete codebooks. The trained model is saved to `models/<model_id>.pt`.

### 2. Generate Semantic IDs

```bash
python main.py generate-ids \
  --config config/config_amazon_tiger.yaml \
  --model_path models/<model_id>.pt \
  --output_path outputs/semids.pt
```

Encodes all items into SID tuples and writes them to `outputs/semids.pt`.

### 3. Train TIGER (Seq2Seq Generative Retrieval)

```bash
python main.py train-seq2seq \
  --config config/config_amazon_tiger.yaml \
  --semids outputs/semids.pt
```

Optional flags:
- `--resume models/<checkpoint>.pt` — resume training from a checkpoint
- `--warmup_steps N` — override warmup steps from config

### 4. Test TIGER

```bash
python main.py test-seq2seq \
  --config config/config_amazon_tiger.yaml \
  --semids outputs/semids.pt \
  --model_path models/<tiger_id>.pt
```

## Configuration

Key parameters in the YAML config files:

```yaml
model:
  quantization_method: "ste"   # "ste" or "gumbel_softmax"
  distance_method: "cosine"    # "cosine" or "l2"
  temperature: 2.0             # Initial temperature (Gumbel Softmax only)
  min_temperature: 0.05
  temperature_decay: 0.9995

train:
  temperature_annealing: true
  temperature_update_frequency: 1
```

## Quantization Methods

### Straight-Through Estimation (STE)

The traditional method using discrete quantization with gradient bypass:

- Fast and memory efficient
- Well-established in VQ-VAE literature
- May suffer from gradient mismatch issues

### Gumbel Softmax (Recommended for Joint Training)

A differentiable alternative that provides better gradient flow:

- Continuous relaxation of discrete sampling
- Better gradient flow for end-to-end training
- Temperature annealing for gradual transition from soft to hard assignments
- Ideal for joint training with language models

## TIGER Architecture

`modules/recommender/seq2seq.py` implements TIGER using a T5 backbone:

- **Generative Retrieval**: autoregressively predicts the next item's SID sequence from user interaction history
- **Unified Vocabulary**: maps User IDs, Semantic IDs, and Collision Tokens into a single embedding space via offsets
- **Recommendation@K**: beam search over item-level SID sequences without an EOS token

## Directory Structure

```
.
├── main.py                    # Unified CLI entry point
├── train_rq_vae.py            # RQ-VAE training logic
├── train_seq2seq.py           # TIGER training logic
├── test_seq2seq.py            # TIGER evaluation
├── generate_semids.py         # SID generation from trained RQ-VAE
├── modules/
│   ├── rqvae/                 # RQ-VAE model, encoder/decoder, quantization
│   └── recommender/           # TIGER seq2seq model
├── data/                      # Data loading and preprocessing
├── config/                    # YAML configuration files
├── models/                    # Saved model checkpoints
└── outputs/                   # Generated Semantic IDs
```

## References

**Recommender Systems with Generative Retrieval** (Rajput et al., NeurIPS 2023). [[arXiv]](https://arxiv.org/abs/2305.05065)

**Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations** (Singh et al., arXiv 2023). [[arXiv]](https://arxiv.org/abs/2306.08121)

**Generative Recommendation with Semantic IDs: A Practitioner's Handbook** (Ju et al., CIKM 2025). [[DOI]](https://doi.org/10.1145/3746252.3761612) [[GitHub]](https://github.com/snap-research/GRID)

**Semantic-ID-Generation** (Hangoebl, base repository this fork extends). [[GitHub]](https://github.com/justinhangoebl/Semantic-ID-Generation)

**RQ-VAE-Recommender** (Botta, inspirational repository). [[GitHub]](https://github.com/EdoardoBotta/RQ-VAE-Recommender)
