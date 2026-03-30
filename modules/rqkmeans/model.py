import torch
import torch.nn.functional as F

from .kmeans import BatchKMeans


class RQKMeans:
    """
    Residual quantization with batch K-Means.

    Fits n_layers of BatchKMeans on successive residuals and returns
    a semantic ID matrix of shape (n_items, n_layers).

    Args:
        n_layers: Number of residual quantization layers.
        n_clusters: Codebook size per layer.
        n_iters: K-Means iterations per layer.
        normalize_residuals: If True, L2-normalize the residual before each fit.
        seed: Random seed (same seed used for all layers for determinism).
    """

    def __init__(self, n_layers: int, n_clusters: int, n_iters: int,
                 normalize_residuals: bool, seed: int):
        self.n_layers = n_layers
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.normalize_residuals = normalize_residuals
        self.seed = seed
        self.kmeans_layers: list[BatchKMeans] = [
            self._make_kmeans_layer() for _ in range(n_layers)
        ]

    def _make_kmeans_layer(self) -> BatchKMeans:
        return BatchKMeans(
            n_clusters=self.n_clusters,
            n_iters=self.n_iters,
            seed=self.seed,
        )

    def fit_and_generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fit all layers and return semantic IDs.

        Args:
            x: Item embedding tensor of shape (n_items, d).

        Returns:
            sem_ids: LongTensor of shape (n_items, n_layers).
        """
        n_items = x.shape[0]
        residual = x.float().clone()
        sem_ids = torch.zeros(n_items, self.n_layers, dtype=torch.long)

        for layer_idx, km in enumerate(self.kmeans_layers):
            if self.normalize_residuals:
                residual = F.normalize(residual, p=2, dim=-1)
            km.fit(residual)
            ids, quantized = km.assign(residual)
            residual = residual - quantized
            sem_ids[:, layer_idx] = ids

        return sem_ids

    def save(self, path: str) -> None:
        """
        Save centroids and constructor args to a .pt file.

        Saved dict keys:
            centroids: list of Tensor(n_clusters, d), one per layer
            n_layers, n_clusters, n_iters, normalize_residuals
        """
        torch.save({
            "centroids": [km.centroids for km in self.kmeans_layers],
            "n_layers": self.n_layers,
            "n_clusters": self.n_clusters,
            "n_iters": self.n_iters,
            "normalize_residuals": self.normalize_residuals,
        }, path)

    @classmethod
    def load(cls, path: str) -> "RQKMeans":
        """
        Reconstruct a fitted RQKMeans from a saved .pt file.

        Args:
            path: Path to the file written by save().

        Returns:
            Fully reconstructed RQKMeans instance with fitted centroids.
        """
        data = torch.load(path, weights_only=False)
        model = cls(
            n_layers=data["n_layers"],
            n_clusters=data["n_clusters"],
            n_iters=data["n_iters"],
            normalize_residuals=data["normalize_residuals"],
            seed=0,  # seed not needed post-load (centroids already set)
        )
        for km, centroids in zip(model.kmeans_layers, data["centroids"]):
            km.centroids = centroids
        return model
