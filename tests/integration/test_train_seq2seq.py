"""
Integration tests for the training orchestration script (train_seq2seq.py).

Verifies:
  - Initial training run saves a checkpoint when recall improves
  - Resuming from checkpoint restores optimizer state
  - warmup_steps_override produces a decaying LR after warmup
  - Config overrides are respected
  - eval_delay_epochs skips retrieval metrics calculation during warmup phase

Fixtures (seq2seq_config, semids_path, mock_seq2seq_model, mock_amazon_sequences)
are defined in tests/integration/conftest.py.
"""
import math
import torch
import os
from unittest.mock import patch


FAKE_RECALL = {5: 1.0, 10: 1.0}
FAKE_NDCG   = {5: 0.5,  10: 0.3}
FAKE_HIER   = {}


def _run_and_save_checkpoint(tmp_path, config_path, semids_path):
    """
    Run phase-1 training and write the checkpoint to tmp_path/best.pt.

    Assumes compute_metrics, load_amazon_sequences, and TigerSeq2Seq are
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
                "recall@5": val,
            }, checkpoint_path)
        mock_save.side_effect = _write
        run_training(config_path, semids_path)

    return checkpoint_path


class TestCheckpointSaved:

    @patch("train_seq2seq.compute_metrics",
           return_value=(FAKE_RECALL, FAKE_NDCG, FAKE_HIER))
    @patch("train_seq2seq.load_amazon_sequences")
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
                    "recall@5": val,
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
           return_value=(FAKE_RECALL, FAKE_NDCG, FAKE_HIER))
    @patch("train_seq2seq.load_amazon_sequences")
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
           return_value=(FAKE_RECALL, FAKE_NDCG, FAKE_HIER))
    @patch("train_seq2seq.load_amazon_sequences")
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

    @patch("train_seq2seq.load_amazon_sequences")
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
                   return_value=(FAKE_RECALL, FAKE_NDCG, FAKE_HIER)):
            checkpoint_path = _run_and_save_checkpoint(tmp_path, config_path, semids_path)

        assert os.path.exists(checkpoint_path), "Precondition: checkpoint must exist"

        # resume with low recall (0.0 < saved best 1.0) → no new checkpoint.
        with patch("train_seq2seq.compute_metrics",
                   return_value=({5: 0.0, 10: 0.0}, FAKE_NDCG, FAKE_HIER)), \
             patch("train_seq2seq.save_checkpoint") as mock_save:
            run_training(config_path, semids_path, resume_path=checkpoint_path)
            assert not mock_save.called, (
                "save_checkpoint must not be called when resumed recall does not improve"
            )


class TestWarmupOverride:

    @patch("train_seq2seq.compute_metrics",
           return_value=(FAKE_RECALL, FAKE_NDCG, FAKE_HIER))
    @patch("train_seq2seq.load_amazon_sequences")
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
           return_value=(FAKE_RECALL, FAKE_NDCG, FAKE_HIER))
    @patch("train_seq2seq.load_amazon_sequences")
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

    @patch("train_seq2seq.load_amazon_sequences")
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

    @patch("train_seq2seq.compute_metrics", return_value=(FAKE_RECALL, FAKE_NDCG, FAKE_HIER))
    @patch("train_seq2seq.load_amazon_sequences")
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
