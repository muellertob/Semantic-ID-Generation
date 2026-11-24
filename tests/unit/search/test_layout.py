"""Tests for search/layout.py."""
from pathlib import Path

from search.layout import trial_folder


def test_a_trial_folder_is_named_after_the_four_digit_trial_number():
    assert trial_folder("/search/toys/A-rqvae", 7) == Path("/search/toys/A-rqvae/trial_0007")
