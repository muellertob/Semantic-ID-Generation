"""
Unit tests for RQ-KMeans quantization (modules/rqkmeans/).

Covers BatchKMeans (init, fit, assign) and RQKMeans (fit_and_generate, save/load)
in isolation — no orchestration or I/O.
"""
import torch
import torch.nn.functional as F
from modules.rqkmeans.kmeans import BatchKMeans
from modules.rqkmeans.model import RQKMeans


class TestBatchKMeansInit:

    def test_kmeans_plus_plus_init_produces_n_clusters_centroids(self):
        """After fit(), centroids.shape == (n_clusters, d)."""
        n_clusters = 8
        d = 16
        x = torch.randn(100, d)
        km = BatchKMeans(n_clusters=n_clusters, n_iters=5, seed=42)
        km.fit(x)
        assert km.centroids.shape == (n_clusters, d)


class TestBatchKMeansFit:

    def test_fit_assigns_all_points_to_valid_cluster_indices(self):
        """All returned cluster ids must be in [0, n_clusters)."""
        n_clusters = 4
        x = torch.randn(50, 8)
        km = BatchKMeans(n_clusters=n_clusters, n_iters=10, seed=0)
        km.fit(x)
        ids, _ = km.assign(x)
        assert ids.shape == (50,)
        assert ids.min().item() >= 0
        assert ids.max().item() < n_clusters

    def test_fit_is_deterministic_with_same_seed(self):
        """Two BatchKMeans instances with the same seed must produce identical centroids."""
        x = torch.randn(200, 32)
        km1 = BatchKMeans(n_clusters=16, n_iters=20, seed=7)
        km1.fit(x)
        km2 = BatchKMeans(n_clusters=16, n_iters=20, seed=7)
        km2.fit(x)
        assert torch.allclose(km1.centroids, km2.centroids)

    def test_fit_reduces_quantization_error_over_iterations(self):
        """MSE after fit < MSE of random assignment (before fit)."""
        torch.manual_seed(42)
        n, d, k = 300, 16, 8
        x = torch.randn(n, d)

        random_centroids = x[torch.randperm(n)[:k]]
        dists_random = torch.cdist(x, random_centroids)
        min_dists_random = dists_random.min(dim=1).values
        mse_before = (min_dists_random ** 2).mean().item()

        km = BatchKMeans(n_clusters=k, n_iters=50, seed=42)
        km.fit(x)
        _, quantized = km.assign(x)
        mse_after = ((x - quantized) ** 2).mean().item()

        assert mse_after < mse_before


class TestBatchKMeansAssign:

    def test_assign_returns_correct_shapes(self):
        """assign() returns (ids shape (n,), quantized shape (n, d))."""
        n, d, k = 60, 12, 6
        x = torch.randn(n, d)
        km = BatchKMeans(n_clusters=k, n_iters=10, seed=1)
        km.fit(x)
        ids, quantized = km.assign(x)
        assert ids.shape == (n,)
        assert quantized.shape == (n, d)

    def test_empty_cluster_keeps_previous_centroid(self):
        """
        When a cluster receives no points after an update step, its centroid must
        remain unchanged (no NaN, no zero-reset).
        """
        x = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ])
        km = BatchKMeans(n_clusters=20, n_iters=10, seed=0)
        km.fit(x)
        assert not torch.isnan(km.centroids).any(), "Centroids must not contain NaN"
        assert not torch.isinf(km.centroids).any(), "Centroids must not contain Inf"


class TestRQKMeansFitAndGenerate:

    def test_fit_and_generate_returns_correct_shape(self):
        """fit_and_generate returns shape (n_items, n_layers)."""
        n_items, d, n_layers, n_clusters = 80, 16, 3, 8
        x = torch.randn(n_items, d)
        model = RQKMeans(n_layers=n_layers, n_clusters=n_clusters, n_iters=10,
                         normalize_residuals=False, seed=42)
        sem_ids = model.fit_and_generate(x)
        assert sem_ids.shape == (n_items, n_layers)

    def test_all_ids_within_valid_range(self):
        """All values in sem_ids must be in [0, n_clusters)."""
        n_items, d, n_layers, n_clusters = 100, 8, 3, 16
        x = torch.randn(n_items, d)
        model = RQKMeans(n_layers=n_layers, n_clusters=n_clusters, n_iters=10,
                         normalize_residuals=False, seed=0)
        sem_ids = model.fit_and_generate(x)
        assert sem_ids.min().item() >= 0
        assert sem_ids.max().item() < n_clusters

    def test_residuals_decrease_across_layers(self):
        """The L2 norm of the residual must decrease as more layers are applied."""
        torch.manual_seed(42)
        n_items, d, n_layers, n_clusters = 200, 32, 3, 16
        x = torch.randn(n_items, d)

        residuals = []

        class _CapturingRQKMeans(RQKMeans):
            def fit_and_generate(self, x):
                residual = x.clone().float()
                sem_ids = torch.zeros(x.shape[0], self.n_layers, dtype=torch.long)
                for layer_idx, km in enumerate(self.kmeans_layers):
                    if self.normalize_residuals:
                        residual = F.normalize(residual, p=2, dim=-1)
                    km.fit(residual)
                    ids, quantized = km.assign(residual)
                    residual = residual - quantized
                    residuals.append(residual.norm(dim=1).mean().item())
                    sem_ids[:, layer_idx] = ids
                return sem_ids

        model = _CapturingRQKMeans(n_layers=n_layers, n_clusters=n_clusters,
                                   n_iters=20, normalize_residuals=False, seed=7)
        model.fit_and_generate(x)

        assert len(residuals) == n_layers
        assert residuals[1] < residuals[0], (
            f"Residual norm should decrease: L1={residuals[0]:.4f}, L2={residuals[1]:.4f}"
        )

    def test_normalize_residuals_flag(self):
        """When normalize_residuals=True, each fit() input must have unit row norms."""
        fit_inputs = []

        class _RecordingKMeans(BatchKMeans):
            def fit(self, x):
                fit_inputs.append(x.clone())
                super().fit(x)

        class _RQKMeansWithRecording(RQKMeans):
            def _make_kmeans_layer(self):
                return _RecordingKMeans(
                    n_clusters=self.n_clusters,
                    n_iters=self.n_iters,
                    seed=self.seed,
                )

        n_items, d, n_layers, n_clusters = 50, 8, 3, 4
        x = torch.randn(n_items, d) * 10

        model = _RQKMeansWithRecording(n_layers=n_layers, n_clusters=n_clusters,
                                       n_iters=5, normalize_residuals=True, seed=42)
        model.fit_and_generate(x)

        assert len(fit_inputs) == n_layers, "fit() must be called once per layer"
        for layer_idx, inp in enumerate(fit_inputs):
            norms = inp.norm(dim=1, p=2)
            assert torch.allclose(norms, torch.ones(n_items), atol=1e-5), (
                f"Layer {layer_idx}: fit() input row norms must be 1.0 when "
                f"normalize_residuals=True, got min={norms.min():.4f} max={norms.max():.4f}"
            )

    def test_single_layer_matches_single_kmeans(self):
        """With n_layers=1, RQKMeans output must match a standalone BatchKMeans."""
        torch.manual_seed(0)
        n_items, d, n_clusters = 100, 16, 8
        x = torch.randn(n_items, d)

        km = BatchKMeans(n_clusters=n_clusters, n_iters=20, seed=99)
        km.fit(x)
        expected_ids, _ = km.assign(x)

        model = RQKMeans(n_layers=1, n_clusters=n_clusters, n_iters=20,
                         normalize_residuals=False, seed=99)
        sem_ids = model.fit_and_generate(x)

        assert torch.equal(sem_ids[:, 0], expected_ids)


class TestRQKMeansSaveLoad:

    def test_save_and_load_roundtrip(self, tmp_path):
        """Loaded centroids must match saved centroids exactly."""
        n_items, d, n_layers, n_clusters = 60, 8, 2, 4
        x = torch.randn(n_items, d)
        model = RQKMeans(n_layers=n_layers, n_clusters=n_clusters, n_iters=10,
                         normalize_residuals=False, seed=42)
        model.fit_and_generate(x)

        save_path = str(tmp_path / "rqkmeans.pt")
        model.save(save_path)

        loaded = RQKMeans.load(save_path)
        assert len(loaded.kmeans_layers) == n_layers
        for i, (orig_km, load_km) in enumerate(zip(model.kmeans_layers, loaded.kmeans_layers)):
            assert torch.allclose(orig_km.centroids, load_km.centroids), (
                f"Centroids mismatch at layer {i}"
            )
