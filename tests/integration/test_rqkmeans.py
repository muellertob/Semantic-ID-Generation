"""
Integration tests for the RQ-KMeans training pipeline (train_rqkmeans.py).

Covers orchestration behaviour: output file structure, collision resolution,
call ordering, and output shape. Unit tests for BatchKMeans and RQKMeans
are in tests/unit/quantization/test_rqkmeans.py.
"""
import torch
from unittest.mock import MagicMock, patch # patch replaces a name inside a module with a mock for the duration of the with block 
from omegaconf import OmegaConf


class TestOrchestrationOutputKeys:

    def test_output_file_has_correct_keys(self, tmp_path):
        """The saved .pt file must contain 'semantic_ids', 'config', and 'num_items'."""
        import train_rqkmeans

        config = OmegaConf.create({
            "data": {"dataset": "amazon", "category": "beauty", "normalize_data": True},
            "model": {"n_layers": 2, "codebook_clusters": 4, "n_iters": 5,
                      "normalize_residuals": False},
            "general": {"seed": 0, "save_model": False, "model_path": "",
                        "output_path": str(tmp_path / "semids.pt"),
                        "plot_dir": None},
        })

        n_items, d = 20, 8
        fake_embeddings = torch.randn(n_items, d)

        with patch("train_rqkmeans.load_data", return_value=fake_embeddings), \
             patch("train_rqkmeans.evaluate_semids"), \
             patch("train_rqkmeans.resolve_collisions",
                   side_effect=lambda s, **kw: torch.cat(
                       [s, torch.zeros(s.shape[0], 1, dtype=torch.long)], dim=1)):
            train_rqkmeans.run_training.__wrapped__(config) if hasattr(
                train_rqkmeans.run_training, "__wrapped__"
            ) else train_rqkmeans.run_training(
                config_path=None, _config_override=config
            )

        saved = torch.load(str(tmp_path / "semids.pt"), weights_only=False)
        assert "semantic_ids" in saved, "Key 'semantic_ids' missing from output"
        assert "config" in saved, "Key 'config' missing from output"
        assert "num_items" in saved, "Key 'num_items' missing from output"


class TestOrchestrationNoCollisionDuplicates:

    def test_collision_resolution_applied(self, tmp_path):
        """The final semantic_ids tensor must have no duplicate tuples."""
        import train_rqkmeans

        # 70 items, 8^2=64 possible tuples → pigeonhole guarantees collisions exist.
        n_items, d, n_layers, n_clusters = 70, 8, 2, 8
        torch.manual_seed(42)
        fake_embeddings = torch.randn(n_items, d)

        config = OmegaConf.create({
            "data": {"dataset": "amazon", "category": "beauty", "normalize_data": True},
            "model": {"n_layers": n_layers, "codebook_clusters": n_clusters,
                      "n_iters": 20, "normalize_residuals": False},
            "general": {"seed": 42, "save_model": False, "model_path": "",
                        "output_path": str(tmp_path / "semids.pt"),
                        "plot_dir": None},
        })

        with patch("train_rqkmeans.load_data", return_value=fake_embeddings), \
             patch("train_rqkmeans.evaluate_semids"):
            train_rqkmeans.run_training(config_path=None, _config_override=config)

        saved = torch.load(str(tmp_path / "semids.pt"), weights_only=False)
        sem_ids = saved["semantic_ids"]
        unique_rows = torch.unique(sem_ids, dim=0)
        assert unique_rows.shape[0] == n_items, (
            f"Expected {n_items} unique rows after collision resolution, "
            f"got {unique_rows.shape[0]}"
        )

    def test_collision_tokens_within_codebook_range(self, tmp_path):
        """Collision tokens (last column) must all be in [0, codebook_size)."""
        import train_rqkmeans

        n_items, d, n_layers, n_clusters = 70, 8, 2, 8
        torch.manual_seed(7)
        fake_embeddings = torch.randn(n_items, d)

        config = OmegaConf.create({
            "data": {"dataset": "amazon", "category": "beauty", "normalize_data": True},
            "model": {"n_layers": n_layers, "codebook_clusters": n_clusters,
                      "n_iters": 20, "normalize_residuals": False},
            "general": {"seed": 7, "save_model": False, "model_path": "",
                        "output_path": str(tmp_path / "semids.pt"),
                        "plot_dir": None},
        })

        with patch("train_rqkmeans.load_data", return_value=fake_embeddings), \
             patch("train_rqkmeans.evaluate_semids"):
            train_rqkmeans.run_training(config_path=None, _config_override=config)

        saved = torch.load(str(tmp_path / "semids.pt"), weights_only=False)
        collision_col = saved["semantic_ids"][:, -1]
        assert collision_col.min().item() >= 0
        assert collision_col.max().item() < n_clusters, (
            f"Collision token {collision_col.max().item()} exceeds codebook size {n_clusters}"
        )


class TestOrchestrationEvaluateBeforeResolve:

    def test_evaluate_called_before_collision_resolve(self, tmp_path):
        """evaluate_semids() must be called BEFORE resolve_collisions()."""
        import train_rqkmeans

        n_items, d = 20, 8
        torch.manual_seed(0)
        fake_embeddings = torch.randn(n_items, d)

        config = OmegaConf.create({
            "data": {"dataset": "amazon", "category": "beauty", "normalize_data": True},
            "model": {"n_layers": 2, "codebook_clusters": 4, "n_iters": 5,
                      "normalize_residuals": False},
            "general": {"seed": 0, "save_model": False, "model_path": "",
                        "output_path": str(tmp_path / "semids.pt"),
                        "plot_dir": None},
        })

        # both mocks are children of parent_mock so their calls appear in a shared
        # timeline (parent_mock.mock_calls), making call order verifiable.
        # side_effect on resolve mimics the shape contract (appends collision column)
        # so run_training doesn't crash after the mock returns.
        parent_mock = MagicMock()
        parent_mock.evaluate = MagicMock(return_value={})
        parent_mock.resolve = MagicMock(
            side_effect=lambda s, **kw: torch.cat(
                [s, torch.zeros(s.shape[0], 1, dtype=torch.long)], dim=1
            )
        )

        with patch("train_rqkmeans.load_data", return_value=fake_embeddings), \
             patch("train_rqkmeans.evaluate_semids", parent_mock.evaluate), \
             patch("train_rqkmeans.resolve_collisions", parent_mock.resolve):
            train_rqkmeans.run_training(config_path=None, _config_override=config)

        call_names = [c[0] for c in parent_mock.mock_calls]
        eval_idx = call_names.index("evaluate")
        resolve_idx = call_names.index("resolve")
        assert eval_idx < resolve_idx, (
            f"evaluate_semids (#{eval_idx}) must be called before "
            f"resolve_collisions (#{resolve_idx})"
        )


class TestOrchestrationOutputShape:

    def test_output_shape_matches_expected(self, tmp_path):
        """semantic_ids.shape must be (N, n_layers + 1) after collision token is appended."""
        import train_rqkmeans

        n_items, d, n_layers, n_clusters = 25, 8, 3, 8
        torch.manual_seed(1)
        fake_embeddings = torch.randn(n_items, d)

        config = OmegaConf.create({
            "data": {"dataset": "amazon", "category": "beauty", "normalize_data": True},
            "model": {"n_layers": n_layers, "codebook_clusters": n_clusters,
                      "n_iters": 20, "normalize_residuals": False},
            "general": {"seed": 1, "save_model": False, "model_path": "",
                        "output_path": str(tmp_path / "semids.pt"),
                        "plot_dir": None},
        })

        with patch("train_rqkmeans.load_data", return_value=fake_embeddings), \
             patch("train_rqkmeans.evaluate_semids"):
            train_rqkmeans.run_training(config_path=None, _config_override=config)

        saved = torch.load(str(tmp_path / "semids.pt"), weights_only=False)
        sem_ids = saved["semantic_ids"]
        assert sem_ids.shape == (n_items, n_layers + 1), (
            f"Expected shape ({n_items}, {n_layers + 1}), got {tuple(sem_ids.shape)}"
        )


class TestOrchestrationSaveModel:

    def test_model_file_created_when_save_model_true(self, tmp_path):
        """When save_model=True, the fitted RQKMeans model file must be written to model_path."""
        import train_rqkmeans

        n_items, d, n_layers, n_clusters = 20, 8, 2, 8
        torch.manual_seed(3)
        fake_embeddings = torch.randn(n_items, d)
        model_path = str(tmp_path / "rqkmeans.pt")

        config = OmegaConf.create({
            "data": {"dataset": "amazon", "category": "beauty", "normalize_data": True},
            "model": {"n_layers": n_layers, "codebook_clusters": n_clusters,
                      "n_iters": 10, "normalize_residuals": False},
            "general": {"seed": 3, "save_model": True, "model_path": model_path,
                        "output_path": str(tmp_path / "semids.pt"),
                        "plot_dir": None},
        })

        with patch("train_rqkmeans.load_data", return_value=fake_embeddings), \
             patch("train_rqkmeans.evaluate_semids"):
            train_rqkmeans.run_training(config_path=None, _config_override=config)

        import os
        assert os.path.exists(model_path), "Model file was not created despite save_model=True"
