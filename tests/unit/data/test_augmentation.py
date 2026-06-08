import torch
import pytest
from data.sequence import collate_fn_with_augmentation

def test_augmentation_creates_correct_slices():
    # create fake batch
    # batch size = 2
    # user 0: history = [A, B, C], target = D (codebook_layers = 2)
    # user 1: history = [E, F, G, H], target = I
    codebook_layers = 2
    
    A, B, C, D = [torch.tensor([i, i]) for i in range(1, 5)]
    E, F, G, H, I = [torch.tensor([i, i]) for i in range(5, 10)]
    
    batch = [
        {
            "history_tuples": torch.stack([A, B, C]),
            "target_tuples": D.unsqueeze(0),
            "user_id": 0
        },
        {
            "history_tuples": torch.stack([E, F, G, H]),
            "target_tuples": I.unsqueeze(0),
            "user_id": 1
        }
    ]
    
    # batch size is 2, so it will return exactly 2 augmented sequences
    augmented_batch = collate_fn_with_augmentation(batch)
    
    # should return exactly len(batch)
    assert augmented_batch["history_tuples"].shape[0] == 2
    assert augmented_batch["target_tuples"].shape[0] == 2
    assert augmented_batch["user_ids"].shape[0] == 2
    
    # since it randomly samples 2 out of 9 possible sub-sequences, we can just verify
    # that the outputs have the correct shapes and types.
    all_targets = augmented_batch["target_tuples"].squeeze(1) # [2, 2]
    assert all_targets.shape == (2, codebook_layers)
    
    # verify that the returned sequences are valid sub-sequences by checking that
    # the target tuples are among the possible targets (B, C, D from user 0; F, G, H, I from user 1).
    valid_targets = [2, 3, 4, 6, 7, 8, 9]
    for target in all_targets:
        assert target[0].item() in valid_targets
        assert target[1].item() in valid_targets

def test_augmentation_enforces_max_len():
    
    # create a long history of 10 items
    items = [torch.tensor([i, i]) for i in range(1, 12)]
    
    batch = [
        {
            "history_tuples": torch.stack(items[:-1]),
            "target_tuples": items[-1].unsqueeze(0),
            "user_id": 0
        }
    ]
    
    # we enforce max_len = 3
    # sub-sequences could originally be up to 10 items long
    augmented_batch = collate_fn_with_augmentation(batch, max_len=3)
    
    # verify that the padded history max length is exactly 3
    assert augmented_batch["history_tuples"].shape[1] <= 3

def test_augmentation_max_len_mixed_lengths():
    from data.sequence import collate_fn
    codebook_layers = 2
    
    # seq A length = 10
    A_hist = torch.ones(10, codebook_layers, dtype=torch.long)
    A_targ = torch.ones(1, codebook_layers, dtype=torch.long) * 2
    
    # seq B length = 2
    B_hist = torch.ones(2, codebook_layers, dtype=torch.long) * 3
    B_targ = torch.ones(1, codebook_layers, dtype=torch.long) * 4
    
    batch = [
        {
            "history_tuples": A_hist,
            "target_tuples": A_targ,
            "user_id": 0
        },
        {
            "history_tuples": B_hist,
            "target_tuples": B_targ,
            "user_id": 1
        }
    ]
    
    # using collate_fn directly to test the same trimming logic
    res = collate_fn(batch, max_len=3)
    
    # assert exact shapes
    assert res["history_tuples"].shape == (2, 3, codebook_layers)
    
    # assert exact content for A (trimmed to last 3 elements, all of which are 1s)
    expected_A = torch.ones(3, codebook_layers, dtype=torch.long)
    assert torch.equal(res["history_tuples"][0], expected_A), "Sequence A was not trimmed/padded correctly!"
    
    # assert exact content for B (kept as length 2, then padded to 3 with -1 at the end)
    expected_B = torch.tensor([[3, 3],
                               [3, 3],
                               [-1, -1]], dtype=torch.long)
    assert torch.equal(res["history_tuples"][1], expected_B), "Sequence B was not trimmed/padded correctly!"
    
    # check augmentation
    torch.manual_seed(42)
    res_aug = collate_fn_with_augmentation(batch, max_len=3)
    
    # assert exact shapes
    assert res_aug["history_tuples"].shape == (2, 3, codebook_layers)
    
    # assert exact user IDs (with seed 42, both should be user 0)
    assert torch.equal(res_aug["user_ids"], torch.tensor([0, 0], dtype=torch.long))
    
    # first augmented sequence should be length 3, containing all 1s
    expected_aug_0 = torch.ones(3, codebook_layers, dtype=torch.long)
    assert torch.equal(res_aug["history_tuples"][0], expected_aug_0), "First augmented sequence history is incorrect!"
    
    # second augmented sequence should be length 2 (1s), padded to 3 with -1 at the end
    expected_aug_1 = torch.tensor([[1, 1],
                                   [1, 1],
                                   [-1, -1]], dtype=torch.long)
    assert torch.equal(res_aug["history_tuples"][1], expected_aug_1), "Second augmented sequence history is incorrect!"

def test_augmentation_fallback_on_short_sequences():
    codebook_layers = 2
    # create batch with history length 0 (only target)
    A_hist = torch.empty(0, codebook_layers, dtype=torch.long)
    A_targ = torch.ones(1, codebook_layers, dtype=torch.long) * 2
    
    batch = [
        {
            "history_tuples": A_hist,
            "target_tuples": A_targ,
            "user_id": 0
        }
    ]
    
    augmented_batch = collate_fn_with_augmentation(batch)
    
    assert augmented_batch["history_tuples"].shape[1] == 0
    assert augmented_batch["target_tuples"].shape[0] == 1

def test_augmentation_mixed_empty_and_valid_sequences():
    codebook_layers = 2
    # create batch with one valid history and one empty history
    A_hist = torch.ones(3, codebook_layers, dtype=torch.long)
    A_targ = torch.ones(1, codebook_layers, dtype=torch.long) * 2
    
    B_hist = torch.empty(0, codebook_layers, dtype=torch.long)
    B_targ = torch.ones(1, codebook_layers, dtype=torch.long) * 4
    
    batch = [
        {
            "history_tuples": A_hist,
            "target_tuples": A_targ,
            "user_id": 0
        },
        {
            "history_tuples": B_hist,
            "target_tuples": B_targ,
            "user_id": 1
        }
    ]
    
    # user 0 has N=4, possible sub-sequences: 4*3/2 = 6
    # user 1 has N=1, possible sub-sequences: 0
    # total sub-sequences = 6. max batch size = 2.
    # it should sample 2 sequences, both must come from User 0.
    
    augmented_batch = collate_fn_with_augmentation(batch)
    
    assert augmented_batch["history_tuples"].shape[0] == 2
    assert augmented_batch["target_tuples"].shape[0] == 2
    
    # all sampled sub-sequences should belong to user_id = 0
    assert (augmented_batch["user_ids"] == 0).all()


