"""
Unit tests for modules/recommender/seq2seq.py: the value-range check of set_codebooks.
"""
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
