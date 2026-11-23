"""
Integration tests for the training orchestration script (train_seq2seq.py).

Verifies:
  - Initial training run saves a checkpoint when recall improves
  - Resuming from checkpoint restores optimizer state
  - warmup_steps_override produces a decaying LR after warmup
  - Config overrides are respected
  - eval_delay_epochs skips retrieval metrics calculation during warmup phase
  - the seq2seq.checkpoint_metric block ({metric, k}) selects the metric for checkpointing / early stopping
    (default recall at k=5) and resuming restores the best value of that metric
  - the optional on_validation callback gets one record per validation and can stop training

Fixtures (seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences)
are defined in tests/integration/conftest.py.
"""
import math
import pytest
import torch
import os
from unittest.mock import patch
from omegaconf import OmegaConf


FAKE_RECALL = {5: 1.0, 10: 1.0}
FAKE_NDCG   = {5: 0.5,  10: 0.3}
FAKE_HIER   = {}


def _report(recall=FAKE_RECALL, ndcg=FAKE_NDCG, hier=FAKE_HIER):
    """compute_metrics returns the full MetricAccumulator report."""
    return {"recall": recall, "ndcg": ndcg, "hierarchical": hier, "total_samples": 2}


FAKE_REPORT = _report()


def _run_and_save_checkpoint(tmp_path, config_path, semids_path):
    """
    Run phase-1 training and write the checkpoint to tmp_path/best.pt.

    Assumes compute_metrics, load_sequences, and TigerSeq2Seq are
    already patched by the calling test's decorators — no re-patching here.
    Only save_checkpoint is patched to redirect the file to our controlled path.
    """
    from train_seq2seq import run_training

    # redirect to tmp_path to avoid writing to the real models/ directory.
    checkpoint_path = str(tmp_path / "best.pt")

    with patch("train_seq2seq.save_checkpoint") as mock_save:
        def _write(model, opt, sched, scaler, epoch, val, _path, metric_name="recall@5"):
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict() if sched else None,
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                metric_name: val,
            }, checkpoint_path)
        mock_save.side_effect = _write
        run_training(config_path, semids_path)

    return checkpoint_path


class TestCheckpointSaved:

    @patch("train_seq2seq.compute_metrics",
           return_value=FAKE_REPORT)
    @patch("train_seq2seq.load_sequences")
    @patch("train_seq2seq.TigerSeq2Seq")
    def test_checkpoint_created_on_recall_improvement(
            self, mock_model_class, mock_load_data, _,
            tmp_path, seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
        """Initial training saves a checkpoint when recall@5 improves from 0."""
        from train_seq2seq import run_training

        mock_model_class.return_value = mock_seq2seq_model
        mock_load_data.return_value = (mock_amazon_sequences, 10, 20)
        _, config_path = seq2seq_config

        ckpt_path = str(tmp_path / "best.pt")

        with patch("train_seq2seq.save_checkpoint") as mock_save:
            def _write(model, opt, sched, scaler, epoch, val, _path, metric_name="recall@5"):
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict() if sched else None,
                    "scaler_state_dict": scaler.state_dict() if scaler else None,
                    metric_name: val,
                }, ckpt_path)
            mock_save.side_effect = _write
            run_training(config_path, semids_path)

        assert mock_save.called, "save_checkpoint must be called when recall improves"

        # verify correct epoch, recall value, and metric_name were passed.
        _, _, _, _, epoch_saved, recall_saved, _, metric_name_saved = mock_save.call_args.args
        assert epoch_saved == 1, (
            f"Expected checkpoint at epoch 1 (last of 2-epoch run), got {epoch_saved}"
        )
        assert recall_saved == FAKE_RECALL[5], (
            f"Expected recall={FAKE_RECALL[5]}, got {recall_saved}"
        )
        assert metric_name_saved == "recall@5"

    @patch("train_seq2seq.compute_metrics",
           return_value=FAKE_REPORT)
    @patch("train_seq2seq.load_sequences")
    @patch("train_seq2seq.TigerSeq2Seq")
    def test_no_checkpoint_when_save_model_false(
            self, mock_model_class, mock_load_data, _,
            seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences, tmp_path):
        """When save_model=False, save_checkpoint must NOT be called."""
        from train_seq2seq import run_training

        mock_model_class.return_value = mock_seq2seq_model
        mock_load_data.return_value = (mock_amazon_sequences, 10, 20)

        config, config_path = seq2seq_config
        config["general"]["save_model"] = False
        from omegaconf import OmegaConf
        OmegaConf.save(OmegaConf.create(config), config_path)

        with patch("train_seq2seq.save_checkpoint") as mock_save:
            run_training(config_path, semids_path)
            assert not mock_save.called, "save_checkpoint must NOT be called when save_model=False"


class TestResumeTraining:

    @patch("train_seq2seq.compute_metrics",
           return_value=FAKE_REPORT)
    @patch("train_seq2seq.load_sequences")
    @patch("train_seq2seq.TigerSeq2Seq")
    def test_resume_loads_optimizer_state(
            self, mock_model_class, mock_load_data, _,
            tmp_path, seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
        """Resuming from a checkpoint must call model.load_state_dict with the saved weights."""
        from train_seq2seq import run_training

        mock_model_class.return_value = mock_seq2seq_model
        mock_load_data.return_value = (mock_amazon_sequences, 10, 20)
        _, config_path = seq2seq_config

        checkpoint_path = _run_and_save_checkpoint(tmp_path, config_path, semids_path)
        assert os.path.exists(checkpoint_path), "Precondition: checkpoint must exist"

        run_training(config_path, semids_path, resume_path=checkpoint_path)
        mock_seq2seq_model.load_state_dict.assert_called_once()

        # verify the exact state dict from the checkpoint was restored.
        restored_state = mock_seq2seq_model.load_state_dict.call_args.args[0]
        expected_state = mock_seq2seq_model.state_dict.return_value
        assert set(restored_state.keys()) == set(expected_state.keys()), (
            "load_state_dict keys must match the saved checkpoint"
        )
        for k in expected_state:
            # move tensors to CPU for comparison
            assert torch.equal(restored_state[k].cpu(), expected_state[k].cpu()), (
                f"State dict mismatch at key '{k}'"
            )

    @patch("train_seq2seq.load_sequences")
    @patch("train_seq2seq.TigerSeq2Seq")
    def test_resume_restores_best_recall(
            self, mock_model_class, mock_load_data,
            tmp_path, seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
        """best_recall_at_5 is restored from checkpoint; no new save when recall doesn't improve."""
        from train_seq2seq import run_training

        mock_model_class.return_value = mock_seq2seq_model
        mock_load_data.return_value = (mock_amazon_sequences, 10, 20)
        _, config_path = seq2seq_config

        # first run with high recall (1.0) so a checkpoint is written.
        with patch("train_seq2seq.compute_metrics",
                   return_value=FAKE_REPORT):
            checkpoint_path = _run_and_save_checkpoint(tmp_path, config_path, semids_path)

        assert os.path.exists(checkpoint_path), "Precondition: checkpoint must exist"

        # resume with low recall (0.0 < saved best 1.0) → no new checkpoint.
        with patch("train_seq2seq.compute_metrics",
                   return_value=_report(recall={5: 0.0, 10: 0.0})), \
             patch("train_seq2seq.save_checkpoint") as mock_save:
            run_training(config_path, semids_path, resume_path=checkpoint_path)
            assert not mock_save.called, (
                "save_checkpoint must not be called when resumed recall does not improve"
            )


class TestWarmupOverride:

    @patch("train_seq2seq.compute_metrics",
           return_value=FAKE_REPORT)
    @patch("train_seq2seq.load_sequences")
    @patch("train_seq2seq.TigerSeq2Seq")
    def test_warmup_override_produces_decaying_lr(
            self, mock_model_class, mock_load_data, _,
            tmp_path, seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
        """
        With warmup_steps_override equal to steps_per_epoch and num_epochs=2:
          - steps_per_epoch = ceil(train_samples / batch_size)
          - total_steps = 2 * steps_per_epoch
          - warmup_steps = steps_per_epoch, decay_steps = steps_per_epoch
          - resume fast-forwards steps_per_epoch steps (end of warmup) → LR enters cosine decay
          - 1 more epoch completes cosine decay → LR ≈ eta_min (1e-5)
        LR must be < initial learning_rate (0.01).
        """
        from train_seq2seq import run_training

        mock_model_class.return_value = mock_seq2seq_model
        mock_load_data.return_value = (mock_amazon_sequences, 10, 20)
        config, config_path = seq2seq_config

        # derive steps_per_epoch from the fixture data so the warmup math stays
        # consistent even if batch_size or dataset size changes.
        train_samples = len(mock_amazon_sequences["train"]["userId"])  # 10
        batch_size = config["seq2seq"]["batch_size"]                   # 2
        steps_per_epoch = math.ceil(train_samples / batch_size)        # 5
        warmup_steps_override = steps_per_epoch

        checkpoint_path = _run_and_save_checkpoint(tmp_path, config_path, semids_path)
        assert os.path.exists(checkpoint_path), "Precondition: phase-1 checkpoint must exist"

        # resume with warmup_steps_override and capture the LambdaLR
        captured_schedulers = []
        import torch.optim as optim
        original_lambda = optim.lr_scheduler.LambdaLR

        def capture_lambda(*args, **kwargs):
            sched = original_lambda(*args, **kwargs)
            captured_schedulers.append(sched)
            return sched

        with patch("train_seq2seq.optim.lr_scheduler.LambdaLR",
                   side_effect=capture_lambda):
            run_training(config_path, semids_path,
                         resume_path=checkpoint_path,
                         warmup_steps_override=warmup_steps_override)

        assert len(captured_schedulers) == 1, "LambdaLR must be created exactly once"
        last_lr = captured_schedulers[0].get_last_lr()[0]
        assert last_lr < config["seq2seq"]["learning_rate"], (
            f"LR ({last_lr:.6f}) must be below initial LR "
            f"({config['seq2seq']['learning_rate']}) after warmup completes"
        )


class TestConfigOverrides:

    @patch("train_seq2seq.compute_metrics",
           return_value=FAKE_REPORT)
    @patch("train_seq2seq.load_sequences")
    @patch("train_seq2seq.TigerSeq2Seq")
    def test_overrides_respected(
            self, mock_model_class, mock_load_data, _,
            seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
        """Verify dot-list overrides successfully update configuration parameters (e.g. learning rate)."""
        from train_seq2seq import run_training
        import torch.optim as optim

        mock_model_class.return_value = mock_seq2seq_model
        mock_load_data.return_value = (mock_amazon_sequences, 10, 20)
        _, config_path = seq2seq_config

        captured_lr = None
        original_adam = optim.Adam

        def capture_lr(*args, **kwargs):
            nonlocal captured_lr
            captured_lr = kwargs.get('lr')
            return original_adam(*args, **kwargs)

        overrides = ["seq2seq.learning_rate=0.1234", "seq2seq.early_stopping=False"]
        with patch("train_seq2seq.optim.Adam", side_effect=capture_lr):
            run_training(config_path, semids_path, overrides=overrides)

        assert captured_lr == 0.1234


class TestEvalDelayEpochs:

    @patch("train_seq2seq.load_sequences")
    @patch("train_seq2seq.TigerSeq2Seq")
    def test_eval_delay_epochs_skips_metrics(
            self, mock_model_class, mock_load_data,
            seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
        """When num_epochs=2 and eval_delay_epochs=2, compute_metrics should not be called at all."""
        from train_seq2seq import run_training

        mock_model_class.return_value = mock_seq2seq_model
        mock_load_data.return_value = (mock_amazon_sequences, 10, 20)
        _, config_path = seq2seq_config

        overrides = ["seq2seq.num_epochs=2", "seq2seq.eval_delay_epochs=2", "seq2seq.eval_epoch_interval=1"]
        with patch("train_seq2seq.compute_metrics") as mock_compute:
            run_training(config_path, semids_path, overrides=overrides)
            assert not mock_compute.called, "compute_metrics should not be called when delay is equal to or greater than num_epochs"

    @patch("train_seq2seq.compute_metrics", return_value=FAKE_REPORT)
    @patch("train_seq2seq.load_sequences")
    @patch("train_seq2seq.TigerSeq2Seq")
    def test_eval_delay_epochs_allows_metrics_after_delay(
            self, mock_model_class, mock_load_data, mock_compute,
            seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
        """When num_epochs=2 and eval_delay_epochs=1 and eval_epoch_interval=1, compute_metrics should be called once (for epoch 2)."""
        from train_seq2seq import run_training

        mock_model_class.return_value = mock_seq2seq_model
        mock_load_data.return_value = (mock_amazon_sequences, 10, 20)
        _, config_path = seq2seq_config

        overrides = ["seq2seq.num_epochs=2", "seq2seq.eval_delay_epochs=1", "seq2seq.eval_epoch_interval=1"]
        run_training(config_path, semids_path, overrides=overrides)
        assert mock_compute.call_count == 1, "compute_metrics should be called exactly once (for epoch 2)"


def _write_resume_checkpoint(path, model_state_dict, metric_key, metric_value):
    """A resumable checkpoint that stores the metric under metric_key, as save_checkpoint does."""
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(1))])
    torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)  # adds 'initial_lr' to the param groups
    torch.save({
        "epoch": 0,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "scaler_state_dict": None,
        metric_key: metric_value,
    }, path)


class TestCheckpointMetric:

    def test_save_checkpoint_stores_metric_under_its_name(self, tmp_path):
        """save_checkpoint stores the metric only under its own key (e.g. 'ndcg@10')."""
        from train_seq2seq import save_checkpoint
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(model, optimizer, None, None, 3, 0.25, path, "ndcg@10")
        ckpt = torch.load(path, weights_only=False)
        assert ckpt["ndcg@10"] == 0.25
        assert ckpt["epoch"] == 3
        assert "metric_name" not in ckpt
        assert "metric_value" not in ckpt

    @pytest.mark.parametrize("block, expected", [
        (None, ("recall", 5)),
        ({"metric": "ndcg", "k": 10}, ("ndcg", 10)),
        ({"k": 10}, ("recall", 10)),
        ({"metric": "ndcg"}, ("ndcg", 5)),
    ])
    def test_get_checkpoint_metric(self, block, expected):
        from train_seq2seq import get_checkpoint_metric
        seq2seq = {} if block is None else {"checkpoint_metric": block}
        assert get_checkpoint_metric(OmegaConf.create({"seq2seq": seq2seq})) == expected

    @pytest.mark.parametrize("bad, message", [
        ({"metric": "mrr", "k": 5}, "metric must be"),
        ({"metric": "NDCG", "k": 10}, "metric must be"),
        ({"metric": "ndcg", "k": 20}, "k must be"),
        ({"metric": "ndcg", "k": "10"}, "k must be"),
        ({"metric": "recall", "k": True}, "k must be"),
        ("ndcg@10", "must be a block"),
    ])
    def test_get_checkpoint_metric_rejects_invalid(self, bad, message):
        from train_seq2seq import get_checkpoint_metric
        with pytest.raises(ValueError, match=message):
            get_checkpoint_metric(OmegaConf.create({"seq2seq": {"checkpoint_metric": bad}}))

    @patch("train_seq2seq.compute_metrics")
    @patch("train_seq2seq.load_sequences")
    @patch("train_seq2seq.TigerSeq2Seq")
    def test_unvalidated_k_fails_before_training(
            self, mock_model_class, mock_load_data, mock_compute,
            seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
        """checkpoint_metric k=20 is not a validation k: run_training raises at startup."""
        from train_seq2seq import run_training

        mock_model_class.return_value = mock_seq2seq_model
        mock_load_data.return_value = (mock_amazon_sequences, 10, 20)
        _, config_path = seq2seq_config

        with pytest.raises(ValueError, match="k must be"):
            run_training(config_path, semids_path, overrides=["seq2seq.checkpoint_metric.metric=ndcg", "seq2seq.checkpoint_metric.k=20"])
        assert not mock_compute.called

    @patch("train_seq2seq.compute_metrics", return_value=FAKE_REPORT)
    @patch("train_seq2seq.load_sequences")
    @patch("train_seq2seq.TigerSeq2Seq")
    def test_ndcg_metric_selects_checkpoint_value(
            self, mock_model_class, mock_load_data, _,
            tmp_path, seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
        """checkpoint_metric {metric: ndcg, k: 10} saves the NDCG@10 value under the name 'ndcg@10'."""
        from train_seq2seq import run_training

        mock_model_class.return_value = mock_seq2seq_model
        mock_load_data.return_value = (mock_amazon_sequences, 10, 20)
        _, config_path = seq2seq_config

        with patch("train_seq2seq.save_checkpoint") as mock_save:
            run_training(config_path, semids_path, overrides=["seq2seq.checkpoint_metric.metric=ndcg", "seq2seq.checkpoint_metric.k=10"])

        assert mock_save.called
        _, _, _, _, _, value_saved, _, metric_name_saved = mock_save.call_args.args
        assert metric_name_saved == "ndcg@10"
        assert value_saved == FAKE_NDCG[10]

    @patch("train_seq2seq.load_sequences")
    @patch("train_seq2seq.TigerSeq2Seq")
    def test_early_stopping_counts_chosen_metric(
            self, mock_model_class, mock_load_data,
            seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
        """Recall@5 keeps improving but NDCG@10 does not: with ndcg@10 training stops early."""
        from train_seq2seq import run_training

        mock_model_class.return_value = mock_seq2seq_model
        mock_load_data.return_value = (mock_amazon_sequences, 10, 20)
        _, config_path = seq2seq_config

        reports = [_report(recall={5: 0.1 * (i + 1), 10: 0.5}, ndcg={5: 0.1, 10: 0.2}) for i in range(4)]
        overrides = ["seq2seq.num_epochs=4", "seq2seq.eval_epoch_interval=1",
                     "seq2seq.early_stopping_patience=1", "seq2seq.checkpoint_metric.metric=ndcg", "seq2seq.checkpoint_metric.k=10"]
        with patch("train_seq2seq.compute_metrics", side_effect=reports) as mock_compute, \
             patch("train_seq2seq.save_checkpoint"):
            run_training(config_path, semids_path, overrides=overrides)
        # evaluation 1 improves (0 -> 0.2), evaluation 2 does not -> stop after 2 evaluations
        assert mock_compute.call_count == 2

        with patch("train_seq2seq.compute_metrics", side_effect=reports) as mock_compute, \
             patch("train_seq2seq.save_checkpoint"):
            run_training(config_path, semids_path,
                         overrides=overrides[:-2] + ["seq2seq.checkpoint_metric.metric=recall", "seq2seq.checkpoint_metric.k=5"])
        assert mock_compute.call_count == 4

    @patch("train_seq2seq.compute_metrics", return_value=FAKE_REPORT)
    @patch("train_seq2seq.load_sequences")
    @patch("train_seq2seq.TigerSeq2Seq")
    def test_resume_restores_best_recall(
            self, mock_model_class, mock_load_data, _,
            tmp_path, seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
        """A checkpoint with a 'recall@5' key restores the best recall@5."""
        from train_seq2seq import run_training

        mock_model_class.return_value = mock_seq2seq_model
        mock_load_data.return_value = (mock_amazon_sequences, 10, 20)
        _, config_path = seq2seq_config

        ckpt_path = str(tmp_path / "recall.pt")
        # 5.0 is above any recall the mocked evaluation returns
        _write_resume_checkpoint(ckpt_path, mock_seq2seq_model.state_dict.return_value, "recall@5", 5.0)

        with patch("train_seq2seq.save_checkpoint") as mock_save:
            run_training(config_path, semids_path, resume_path=ckpt_path)
            assert not mock_save.called, "best recall@5 from the checkpoint must be restored"

        # resuming with another metric: the stored recall@5 does not apply, so the best is reset to 0
        with patch("train_seq2seq.save_checkpoint") as mock_save:
            run_training(config_path, semids_path, resume_path=ckpt_path,
                         overrides=["seq2seq.checkpoint_metric.metric=ndcg", "seq2seq.checkpoint_metric.k=10"])
            assert mock_save.called

    @patch("train_seq2seq.compute_metrics", return_value=FAKE_REPORT)
    @patch("train_seq2seq.load_sequences")
    @patch("train_seq2seq.TigerSeq2Seq")
    def test_resume_restores_best_value_from_ndcg_key(
            self, mock_model_class, mock_load_data, _,
            tmp_path, seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
        """checkpoint_metric {metric: ndcg, k: 10} resumes from the checkpoint key 'ndcg@10' and restores its best value."""
        from train_seq2seq import run_training

        mock_model_class.return_value = mock_seq2seq_model
        mock_load_data.return_value = (mock_amazon_sequences, 10, 20)
        _, config_path = seq2seq_config

        ckpt_path = str(tmp_path / "ndcg.pt")
        # 5.0 is above any NDCG the mocked evaluation returns
        _write_resume_checkpoint(ckpt_path, mock_seq2seq_model.state_dict.return_value, "ndcg@10", 5.0)

        with patch("train_seq2seq.save_checkpoint") as mock_save:
            run_training(config_path, semids_path, resume_path=ckpt_path,
                         overrides=["seq2seq.checkpoint_metric.metric=ndcg", "seq2seq.checkpoint_metric.k=10"])
            assert not mock_save.called, "best ndcg@10 from the checkpoint must be restored"


class TestCheckpointName:

    @pytest.fixture
    def train_and_get_meta(self, tmp_path, seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
        """Runs a 2-epoch training with the given general.run_name (or none) and returns (best checkpoint path, meta)."""
        import json
        from train_seq2seq import run_training

        def run(run_name):
            config, config_path = seq2seq_config
            config["general"].update(recommender_model_dir=str(tmp_path / "rec"), meta_json=str(tmp_path / "meta.json"))
            if run_name:
                config["general"]["run_name"] = run_name
            OmegaConf.save(OmegaConf.create(config), config_path)
            # the Stage-1 type is read from the SIDs file
            semids = torch.load(semids_path, weights_only=False)
            semids["sid_type"] = "RQ-VAE"
            torch.save(semids, semids_path)

            with patch("train_seq2seq.compute_metrics", return_value=FAKE_REPORT), \
                    patch("train_seq2seq.load_sequences", return_value=(mock_amazon_sequences, 10, 20)), \
                    patch("train_seq2seq.TigerSeq2Seq", return_value=mock_seq2seq_model), \
                    patch("train_seq2seq.save_checkpoint") as mock_save:
                run_training(config_path, semids_path)
            with open(tmp_path / "meta.json") as f:
                return mock_save.call_args.args[6], json.load(f)
        return run

    def test_run_name_from_config_names_the_checkpoint_and_the_meta(self, train_and_get_meta):
        # two variants with the same sid_type must not write the same file
        save_path, meta = train_and_get_meta("RQ_VAE_cal_seed_42")
        assert (os.path.basename(save_path), meta["recommender_run_name"]) == (
            "RQ_VAE_cal_seed_42_best.pt", "RQ_VAE_cal_seed_42")

    def test_without_run_name_the_checkpoint_is_named_after_the_sid_type_and_seed(self, train_and_get_meta):
        save_path, meta = train_and_get_meta(None)
        assert (os.path.basename(save_path), meta["recommender_run_name"]) == (
            "RQ-VAE_seed_42_best.pt", "RQ-VAE_seed_42")


# ---- on_validation callback ------------------------------------------------------------------
# Every test below trains with ndcg@10 as checkpoint metric, one validation per epoch (unless stated),
# and these validation reports (recall@5 = recall@10 = 1.0 and ndcg@5 = 0.5 throughout):
#
# validation | epoch | global_step | ndcg@10 | best_ndcg@10 | save_checkpoint (epoch index, value)
#     1      |   1   |      5      |  0.30   |    0.30      | (0, 0.30)
#     2      |   2   |     10      |  0.20   |    0.30      | -
#     3      |   3   |     15      |  0.40   |    0.40      | (2, 0.40)
#     4      |   4   |     20      |  0.35   |    0.40      | -
#
# global_step: the fixture has 10 training rows at batch size 2, so one epoch is 5 optimizer steps.
VALIDATION_NDCG10 = [0.30, 0.20, 0.40, 0.35]


def _record(epoch, step, ndcg10, best):
    return {"epoch": epoch, "global_step": step, "recall@5": 1.0, "recall@10": 1.0,
            "ndcg@5": 0.5, "ndcg@10": ndcg10, "best_ndcg@10": best}


def _recording_callback(records, returns):
    """Appends each record to `records` and returns the next value of `returns`."""
    answers = iter(returns)

    def on_validation(record):
        records.append(record)
        return next(answers)
    return on_validation


@pytest.fixture
def train_with_callback(seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences):
    """Returns run(on_validation, epochs, interval, overrides) -> (save calls, compute_metrics calls, evaluate_loss calls)."""
    import train_seq2seq

    def run(on_validation, epochs=4, interval=1, overrides=()):
        _, config_path = seq2seq_config
        reports = [_report(ndcg={5: 0.5, 10: v}) for v in VALIDATION_NDCG10]
        all_overrides = [f"seq2seq.num_epochs={epochs}", f"seq2seq.eval_epoch_interval={interval}",
                         "seq2seq.eval_delay_epochs=0", "seq2seq.early_stopping=False",
                         "seq2seq.checkpoint_metric.metric=ndcg", "seq2seq.checkpoint_metric.k=10", *overrides]
        with patch("train_seq2seq.compute_metrics", side_effect=reports) as mock_compute, \
                patch("train_seq2seq.evaluate_loss", wraps=train_seq2seq.evaluate_loss) as mock_eval_loss, \
                patch("train_seq2seq.load_sequences", return_value=(mock_amazon_sequences, 10, 20)), \
                patch("train_seq2seq.TigerSeq2Seq", return_value=mock_seq2seq_model), \
                patch("train_seq2seq.save_checkpoint") as mock_save:
            train_seq2seq.run_training(config_path, semids_path, overrides=all_overrides, on_validation=on_validation)
        saves = [(c.args[4], c.args[5]) for c in mock_save.call_args_list]
        return saves, mock_compute.call_count, mock_eval_loss.call_count
    return run


class TestValidationCallback:

    def test_without_callback_training_is_unchanged(self, train_with_callback):
        saves, n_compute, _ = train_with_callback(None)
        assert (saves, n_compute) == ([(0, 0.30), (2, 0.40)], 4)

    def test_callback_gets_one_exact_record_per_validation(self, train_with_callback):
        records = []
        train_with_callback(_recording_callback(records, [False] * 4))
        assert records == [_record(1, 5, 0.30, 0.30), _record(2, 10, 0.20, 0.30),
                           _record(3, 15, 0.40, 0.40), _record(4, 20, 0.35, 0.40)]

    def test_records_survive_a_json_round_trip(self, train_with_callback):
        import json
        records = []
        train_with_callback(_recording_callback(records, [False] * 4))
        # the worker writes each record to a JSON lines file; tensors or numpy floats would not survive
        assert [json.loads(json.dumps(r)) for r in records] == records

    def test_callback_runs_only_at_validation_epochs_and_the_last_epoch(self, train_with_callback):
        # interval 2 over 5 epochs: validations after epochs 2 and 4, and after the last epoch 5
        records = []
        train_with_callback(_recording_callback(records, [False] * 3), epochs=5, interval=2)
        assert [(r["epoch"], r["global_step"]) for r in records] == [(2, 10), (4, 20), (5, 25)]

    def test_callback_runs_after_the_checkpoint_decision(self, train_with_callback):
        import train_seq2seq
        saves_seen = []

        def on_validation(record):
            # save_checkpoint is the patched mock while training runs
            saves_seen.append(train_seq2seq.save_checkpoint.call_count)
            return False
        train_with_callback(on_validation)
        # validations 1 and 3 save a new best before the callback sees them
        assert saves_seen == [1, 1, 2, 2]

    def test_true_stops_training_right_after_that_validation(self, train_with_callback):
        _, n_compute, n_eval_loss = train_with_callback(_recording_callback([], [False, True]))
        # no third epoch: neither its loss evaluation nor its validation runs
        assert (n_compute, n_eval_loss) == (2, 2)

    def test_stop_keeps_the_best_checkpoint_of_that_validation(self, train_with_callback):
        # the stop comes at validation 3, which is also a new best (0.40)
        saves, _, _ = train_with_callback(_recording_callback([], [False, False, True]))
        assert saves == [(0, 0.30), (2, 0.40)]

    def test_train_meta_is_written_after_a_stop(self, train_with_callback, tmp_path, semids_path):
        import json
        # a sid_type in the SIDs file makes general.run_name name the run (see TestCheckpointName)
        semids = torch.load(semids_path, weights_only=False)
        semids["sid_type"] = "RQ-VAE"
        torch.save(semids, semids_path)
        rec_dir, meta_path = tmp_path / "rec", tmp_path / "meta.json"
        train_with_callback(_recording_callback([], [False, True]),
                            overrides=[f"general.recommender_model_dir={rec_dir}", f"general.meta_json={meta_path}",
                                       "general.run_name=trial"])
        with open(meta_path) as f:
            assert json.load(f) == {"recommender_run_name": "trial",
                                    "best_model_path": os.path.join(str(rec_dir), "trial_best.pt")}

    def test_callback_must_return_a_bool(self, train_with_callback):
        # a forgotten return (None) must not count as "continue": the error comes at the first validation
        records = []
        with pytest.raises(TypeError, match="on_validation must return a bool, got NoneType"):
            train_with_callback(_recording_callback(records, [None]))
        assert records == [_record(1, 5, 0.30, 0.30)]

    def test_exception_in_callback_propagates(self, train_with_callback):
        def on_validation(record):
            raise RuntimeError("reporter failed")
        with pytest.raises(RuntimeError, match="reporter failed"):
            train_with_callback(on_validation)
