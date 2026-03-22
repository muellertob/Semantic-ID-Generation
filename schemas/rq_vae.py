from typing import NamedTuple
from torch import Tensor

class RqVaeOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    quantize_loss: Tensor


class RqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor
    layer_coverages: Tensor       # (n_layers,) fraction of codebook used per layer
    layer_entropies: Tensor       # (n_layers,) assignment entropy per layer
    first_residual_norm: Tensor   # scalar: ||residual after layer 0|| / ||encoder output||
    last_residual_norm: Tensor    # scalar: ||residual after last layer|| / ||encoder output||
    first_centroids_norm: Tensor  # scalar: mean norm of first-layer codebook vectors
    last_centroids_norm: Tensor   # scalar: mean norm of last-layer codebook vectors