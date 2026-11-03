"""
Unit tests for data/loader.py: normalize_items.

Fixture: three items, the first two are training items.
  item   x       training?
  0      [1, 0]  yes
  1      [3, 4]  yes
  2      [2, 2]  no
Training mean [2, 2] (so item 2 sits exactly on it), population std [1, 2].
"""
import pytest
import torch
import torch.nn.functional as F

from data.loader import normalize_items

X = torch.tensor([[1., 0.], [3., 4.], [2., 2.]])
TRAIN_MASK = torch.tensor([True, True, False])
SQRT2 = 2 ** 0.5


def test_center_l2_centers_with_the_training_mean_then_normalizes_rows():
    # centered rows [-1,-2], [1,2], [0,0] (= [-1,-2]/sqrt5, [1,2]/sqrt5, 0 after L2); the zero row stays 0
    # under F.normalize. Compared with F.normalize of the hand-centered rows, so the check is exact.
    centered = torch.tensor([[-1., -2.], [1., 2.], [0., 0.]])
    assert torch.equal(normalize_items(X, TRAIN_MASK, "center_l2"), F.normalize(centered, p=2, dim=1))


def test_std_l2_divides_by_the_training_std_then_normalizes_rows():
    # centered [-1,-2], [1,2], [0,0]; divided by std [1,2]: [-1,-1], [1,1], [0,0]; unit rows follow
    expected = torch.tensor([[-1 / SQRT2, -1 / SQRT2], [1 / SQRT2, 1 / SQRT2], [0., 0.]])
    # the divisor is sqrt(var + 1e-5) (as BatchNorm), not std: it shifts the direction by < 1e-5, hence atol 1e-4
    assert torch.allclose(normalize_items(X, TRAIN_MASK, "std_l2"), expected, atol=1e-4, rtol=0)


def test_center_subtracts_the_training_mean_and_keeps_the_length():
    expected = torch.tensor([[-1., -2.], [1., 2.], [0., 0.]])
    assert torch.equal(normalize_items(X, TRAIN_MASK, "center"), expected)


def test_none_returns_the_input_unchanged():
    assert torch.equal(normalize_items(X, TRAIN_MASK, "none"), X)


@pytest.mark.parametrize("mode", ["center_l2", "std_l2"])
def test_statistics_come_from_training_items_only(mode):
    # a far-away non-training item must not move the mean or std: the training rows keep their values
    changed = X.clone()
    changed[2] = torch.tensor([100., 100.])
    assert torch.equal(normalize_items(changed, TRAIN_MASK, mode)[:2], normalize_items(X, TRAIN_MASK, mode)[:2])


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="input_normalization must be one of"):
        normalize_items(X, TRAIN_MASK, "foo")
