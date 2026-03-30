"""
BatchKMeans: single K-Means layer operating on in-memory tensors.

K-Means++ initialization + deterministic seeding + empty-cluster preservation.
"""
import torch
import torch.nn.functional as F


class BatchKMeans:
    """
    Single K-Means layer using batch (full-data) updates.

    Args:
        n_clusters: Number of centroids.
        n_iters: Number of EM iterations.
        seed: Random seed for reproducibility.
    """

    def __init__(self, n_clusters: int, n_iters: int = 100, seed: int = 42):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.seed = seed
        self.centroids: torch.Tensor | None = None

    def _kmeans_plus_plus_init(self, x: torch.Tensor) -> torch.Tensor:
        """
        K-Means++ centroid initialization.

        Returns:
            centroids: Tensor of shape (n_clusters, d)
        """
        n, d = x.shape
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        # first centroid: uniform random
        idx = torch.randint(0, n, (1,), generator=generator).item()
        centroids = [x[idx]]

        for _ in range(1, self.n_clusters):
            # stack current centroids: (k, d)
            c = torch.stack(centroids, dim=0)
            # squared distances from each point to its nearest centroid
            dists = torch.cdist(x, c, p=2)              # (n, k)
            min_sq_dists = dists.min(dim=1).values ** 2 # (n,)
            total = min_sq_dists.sum().item()
            if total <= 0:
                # all points coincide with existing centroids —> fall back to uniform
                next_idx = torch.randint(0, n, (1,), generator=generator).item()
            else:
                probs = min_sq_dists / total
                next_idx = torch.multinomial(probs, 1, generator=generator).item()
            centroids.append(x[next_idx])

        return torch.stack(centroids, dim=0)   # (n_clusters, d)

    def fit(self, x: torch.Tensor) -> None:
        """
        Fit centroids on x using batch K-Means.

        Args:
            x: Tensor of shape (n, d)
        """
        x = x.float()
        self.centroids = self._kmeans_plus_plus_init(x)

        for _ in range(self.n_iters):
            # assignment step (euclidean distance)
            dists = torch.cdist(x, self.centroids, p=2) # (n, n_clusters)
            cluster_ids = dists.argmin(dim=1)           # (n,)

            # update step
            new_centroids = self.centroids.clone()
            for k in range(self.n_clusters):
                # select points assigned to cluster k
                mask = cluster_ids == k
                if mask.any():
                    # update centroid to mean of assigned points
                    new_centroids[k] = x[mask].mean(dim=0)
                # else: keep old centroid (empty-cluster preservation)

            self.centroids = new_centroids

    def assign(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assign each point in x to its nearest centroid.

        Args:
            x: Tensor of shape (n, d)

        Returns:
            cluster_ids:          LongTensor of shape (n,)
            quantized_embeddings: FloatTensor of shape (n, d)
        """
        if self.centroids is None:
            raise RuntimeError("BatchKMeans has not been fitted yet. Call fit() first.")
        x = x.float()
        dists = torch.cdist(x, self.centroids, p=2) # (n, n_clusters)
        cluster_ids = dists.argmin(dim=1)           # (n,)
        quantized = self.centroids[cluster_ids]     # (n, d)
        return cluster_ids, quantized
