"""
Shared fixtures for integration tests.
"""
import json
import pytest

@pytest.fixture
def tiny_amazon_dataset(tmp_path):
    """
    A minimal AmazonReviews instance backed by synthetic files.

    Dataset layout:
      - 20 items (1-based IDs 1..20)
      - 10 users, each with 8 distinct items  → all pass the 5-core filter (len >= 5)
      - 1 user with exactly 5 items (uid=11)  → kept (boundary: == 5)
      - 1 user with exactly 4 items (uid=12)  → filtered (boundary: < 5)
      - 1 user with 3 items (uid=99)          → filtered

    PyG's InMemoryDataset.__init__ (download / process / load) is bypassed via a
    local subclass that overrides __init__ and the raw_dir property. No real
    dataset or processed .pt file is needed.
    """
    from data.amazon_data import AmazonReviews

    split = "beauty"
    split_dir = tmp_path / "raw" / split
    split_dir.mkdir(parents=True)

    # datamaps.json — 20 items
    item2id = {f"asin{i}": i for i in range(1, 21)}
    with open(split_dir / "datamaps.json", "w") as f:
        json.dump({"item2id": item2id}, f)

    # sequential_data.txt — "user_id item1 item2 ..." (1-based item IDs)
    lines = []
    for uid in range(1, 11):
        items = [(uid * 7 + i) % 20 + 1 for i in range(8)]
        lines.append(f"{uid} " + " ".join(map(str, items)))
    lines.append("11 18 19 20 1 2")
    lines.append("12 5 6 7 8")
    lines.append("99 1 2 3")
    with open(split_dir / "sequential_data.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    # PyG defines raw_dir as a @property on the class — instance assignment is silently ignored.
    # subclass override redirects it to our tmp_path without triggering download/process.
    class _TinyDataset(AmazonReviews):
        def __init__(self, raw_dir_path, split):
            self._raw_dir_path = raw_dir_path
            self.split = split

        @property
        def raw_dir(self):
            return self._raw_dir_path

    return _TinyDataset(str(tmp_path / "raw"), split)
