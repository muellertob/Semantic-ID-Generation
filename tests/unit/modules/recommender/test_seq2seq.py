"""
Unit tests for modules/recommender/seq2seq.py: the value-range check of set_codebooks, the
decoder cache of beam_search and the user tokens (switched off with user_tokens=0).
"""
from unittest.mock import Mock, patch

import pytest
import torch

from modules.recommender import TigerSeq2Seq


# Vocabulary of the fixture model (codebook_layers=2, codebook_size=4), token = code + offset:
#
#   column      codes   offset   tokens
#   level 1     0..3    0        0..3
#   level 2     0..3    4        4..7
#   collision   0..3    8        8..11
#   pad                          12
@pytest.fixture
def model():
    return TigerSeq2Seq(codebook_layers=2, codebook_size=4, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2)


def test_set_codebooks_accepts_the_largest_code_of_every_column(model):
    model.set_codebooks(torch.tensor([[3, 3, 3], [0, 0, 0]]))

    # 3 is the largest code of each column: tokens 3+0, 3+4, 3+8
    assert model.codebooks.tolist() == [[3, 7, 11], [0, 4, 8]]


def test_set_codebooks_rejects_a_level_code_equal_to_codebook_size(model):
    # level-1 code 4 would get token 4, which is the token of level-2 code 0
    with pytest.raises(ValueError, match=r"column\(s\) \[0\].*max per column \[4, 0, 0\]"):
        model.set_codebooks(torch.tensor([[4, 0, 0]]))


def test_set_codebooks_rejects_a_collision_code_equal_to_codebook_size(model):
    # collision code 4 would get token 12, which is the padding token
    with pytest.raises(ValueError, match=r"column\(s\) \[2\]"):
        model.set_codebooks(torch.tensor([[0, 0, 4]]))


def test_set_codebooks_rejects_a_negative_code(model):
    # -1 marks padding in the history input, so it cannot be a code of an item
    with pytest.raises(ValueError, match=r"column\(s\) \[1\].*min per column \[0, -1, 0\]"):
        model.set_codebooks(torch.tensor([[0, -1, 0]]))


# Six items (codebook_layers=3, codebook_size=4; last column = collision code) whose prefixes
# branch at every level, so the beams change their parent beam between the decoding steps:
#
#   item   level 1   level 2   level 3   collision
#   0      0         0         0         0
#   1      0         0         1         0
#   2      0         1         2         0
#   3      1         0         3         0
#   4      1         2         0         0
#   5      1         2         0         1
BRANCHING_ITEMS = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 2, 0], [1, 0, 3, 0], [1, 2, 0, 0], [1, 2, 0, 1]]


def test_beam_search_with_cache_ranks_items_by_their_teacher_forced_log_probability():
    torch.manual_seed(0)
    model = TigerSeq2Seq(codebook_layers=3, codebook_size=4, d_model=16, d_kv=8, d_ff=32, num_layers=2, num_heads=2).eval()
    items = torch.tensor(BRANCHING_ITEMS)
    model.set_codebooks(items)
    # three users, so that a skipped cache reordering fails reliably: with one user the
    # ranking stayed right by chance for some seeds
    history = items[torch.tensor([[2, 4], [0, 3], [5, 1]])]

    # beam size = number of items: no prefix is dropped, so the beam search is exact
    predictions = model.beam_search(history, beam_size=6)

    # the same ranking without a cache: one teacher-forced forward pass per user and item
    expected = []
    with torch.no_grad():
        for user_history in history:
            logits = model(user_history.repeat(6, 1, 1), items.unsqueeze(1))["logits"][:, :-1]
            token_log_probs = torch.log_softmax(logits, dim=-1).gather(-1, model.codebooks.unsqueeze(-1)).squeeze(-1)
            expected.append(model.codebooks[token_log_probs.sum(dim=-1).argsort(descending=True)].tolist())
    assert predictions.tolist() == expected


def test_beam_search_raises_if_the_decoder_cache_cannot_be_reordered():
    model = TigerSeq2Seq(codebook_layers=3, codebook_size=4, d_model=16, d_kv=8, d_ff=32, num_layers=2, num_heads=2).eval()
    items = torch.tensor(BRANCHING_ITEMS)
    model.set_codebooks(items)
    # a decoder output whose cache is a plain tuple, which has no method to reorder it
    decoder_output = Mock(last_hidden_state=torch.zeros(2, 1, 16), past_key_values=())

    with patch.object(model.backbone.decoder, "forward", return_value=decoder_output):
        with pytest.raises(RuntimeError, match="Cannot reorder the decoder cache of type tuple"):
            model.beam_search(items[[2, 4]].unsqueeze(0), beam_size=2)


# Encoder input of the fixture model (codebook_layers=2, codebook_size=4) for one history item
# [1, 0, 0] and the raw user id 7. Item tokens = code + offset (0, 4, 8). Output vocabulary:
# 2*4 level + 4 collision + 3 special (pad, bos, eos) = 15 tokens, so user tokens start at 15.
#
#   user_tokens   user token          encoder input
#   0 (off)       none                [1, 4, 8]
#   3             15 + (7 mod 3)=16   [16, 1, 4, 8]
def tiny_model(**kwargs):
    return TigerSeq2Seq(codebook_layers=2, codebook_size=4, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2, **kwargs)


ONE_ITEM_HISTORY = torch.tensor([[[1, 0, 0]]])


def test_user_tokens_off_leaves_only_the_history_in_the_encoder_input():
    encoder_input = tiny_model(user_tokens=0).process_input_tuples(ONE_ITEM_HISTORY, user_ids=torch.tensor([7]))

    assert encoder_input.tolist() == [[1, 4, 8]]


def test_user_tokens_off_removes_the_user_rows_from_the_vocabulary():
    assert tiny_model(user_tokens=0).backbone.shared.num_embeddings == 15


def test_user_tokens_on_hash_the_raw_user_id_modulo_the_number_of_slots():
    encoder_input = tiny_model(user_tokens=3).process_input_tuples(ONE_ITEM_HISTORY, user_ids=torch.tensor([7]))

    assert encoder_input.tolist() == [[16, 1, 4, 8]]


def test_forward_with_user_ids_adds_no_user_token_to_the_encoder_input_when_off():
    # the training and test scripts always pass user_ids (they also feed the per-user metrics)
    history = torch.zeros((2, 3, 3), dtype=torch.long)

    out = tiny_model(user_tokens=0)(history, user_ids=torch.arange(2))

    # 3 items * 3 tokens, no user token in front
    assert out["encoder_attention_mask"].tolist() == [[1] * 9] * 2


def test_encoder_output_does_not_depend_on_the_user_id_with_user_tokens_off():
    model = tiny_model(user_tokens=0).eval()
    history = torch.tensor([[[1, 0, 0], [2, 3, 1]], [[3, 3, 0], [0, 1, 1]]])

    with torch.no_grad():
        with_ids = model(history, user_ids=torch.tensor([0, 1]))["encoder_last_hidden_state"]
        without_ids = model(history)["encoder_last_hidden_state"]

    assert torch.equal(with_ids, without_ids)


def test_beam_search_ignores_the_user_id_with_user_tokens_off():
    torch.manual_seed(0)
    model = tiny_model(user_tokens=0).eval()
    model.set_codebooks(torch.tensor([[1, 2, 0], [1, 3, 0], [2, 0, 0]]))
    history = torch.tensor([[[1, 2, 0]], [[2, 0, 0]]])

    with_ids = model.beam_search(history, user_ids=torch.tensor([0, 1]), beam_size=3)
    without_ids = model.beam_search(history, beam_size=3)

    assert with_ids.tolist() == without_ids.tolist()


def test_negative_number_of_user_tokens_is_rejected():
    with pytest.raises(ValueError, match="user_tokens must be 0"):
        tiny_model(user_tokens=-1)


def test_user_tokens_are_off_by_default():
    # a config without the key must not switch user tokens on
    assert tiny_model().user_vocab_size == 0
