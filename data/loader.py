import hashlib
import os

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from data.amazon_data import AmazonReviews
from data.amazon_interactions import read_amazon_interactions
from data.onion_features import read_onion_features
from data.onion_interactions import read_onion_interactions
from data.split_cache import check_meta, load_or_build_split, save_atomic

AMAZON_ROOT = "dataset/amazon"
AMAZON_PROCESSED_DIR = f"{AMAZON_ROOT}/processed"

ONION_ROOT = "dataset/music4all_onion"
ONION_PROCESSED_DIR = f"{ONION_ROOT}/processed"
ONION_EVENTS = "userid_trackid_timestamp.tsv.bz2"
# The thesis subset of Music4All-Onion: the listening events of August and September 2019
# (both bounds inclusive, as stored), the 5-core on distinct (user, track) pairs, and the
# MAEST audio embeddings as item features.
ONION_SUBSET = {"start": "2019-08-01 00:00:00", "end": "2019-09-30 23:59:59", "k": 5,
                "features": "id_maest.tsv.bz2"}
# increase when the reader rules or the stored layout change
ONION_FORMAT_VERSION = 1


def _load_amazon_data(category):
    """The processed item data of a category (HeteroData); built on first use."""
    path = f"{AMAZON_PROCESSED_DIR}/data_{category}.pt"
    if not os.path.exists(path):
        AmazonReviews(AMAZON_ROOT, split=category)
    data, _, _ = torch.load(path, weights_only=False)
    return data


def load_amazon_split(category, split_config):
    """
    The Stage-2 split of an Amazon category under the given SplitConfig (data/protocols.py),
    built from the raw P5 files on first use and cached (data/split_cache.py).
    """
    data = _load_amazon_data(category)
    id_permutation = data['item']['id_permutation']
    return load_or_build_split(
        name=category,
        split_config=split_config,
        processed_dir=AMAZON_PROCESSED_DIR,
        read_sequences=lambda: read_amazon_interactions(
            os.path.join(AMAZON_ROOT, "raw", category), id_permutation,
            with_times=split_config.protocol == 'gts'),
        id_permutation=id_permutation,
        num_items=data['item']['x'].shape[0],
    )


INPUT_NORMALIZATIONS = ("none", "center", "center_l2", "std_l2")


def normalize_items(x, train_mask, mode):
    """
    Normalize item embeddings with statistics of the training items only (no leakage from
    test or cold items).

    mode: 'none' (unchanged), 'center' (subtract the training mean), 'center_l2' (center, then
    L2-normalize rows), 'std_l2' (center, divide each dimension by sqrt(var + 1e-5) of the
    training items as BatchNorm does, then L2-normalize rows).
    """
    if mode not in INPUT_NORMALIZATIONS:
        raise ValueError(f"input_normalization must be one of {INPUT_NORMALIZATIONS}, got {mode!r}")
    if mode == "none":
        return x
    train = x[train_mask]
    x = x - train.mean(dim=0)
    if mode == "std_l2":
        x = x / torch.sqrt(train.var(dim=0, unbiased=False) + 1e-5)
    if mode in ("center_l2", "std_l2"):
        x = F.normalize(x, p=2, dim=1)
    return x


def load_amazon(category, split_config, input_normalization, split='train'):
    """
    Load the item embeddings of an Amazon category.

    protocol 'loo': 'train' / 'test' are the 95% / 5% item split (is_train).
    protocol 'gts': 'train' are the stage-1 (warm) items of the global temporal
    split, 'test' the remaining items. 'all' returns all items.
    input_normalization: see normalize_items; its statistics come from the 'train' items.
    """
    data = _load_amazon_data(category)

    if split_config.protocol == 'gts':
        train_mask = load_amazon_split(category, split_config)['stage1_items']
    else:
        train_mask = data['item']['is_train'] == True

    return select_items(data['item']['x'], train_mask, input_normalization, split)


def select_items(x, train_mask, input_normalization, split):
    """
    The item embeddings of one item split, normalized with statistics of the training items
    (normalize_items).

    split: 'train' (the items of train_mask), 'test' (the others) or 'all' (every item, in
    item id order).
    """
    x = normalize_items(x, train_mask, input_normalization)
    if split == 'all':
        return x
    if split == 'train':
        return x[train_mask]
    if split == 'test':
        return x[~train_mask]
    raise ValueError(f"Invalid split: {split}. Must be 'train', 'test', or 'all'.")


def split_sequences(split):
    """
    (sequences, num_users, num_items) of a cached split: its rows per split name, the
    number of distinct users with at least one row, the number of items.
    """
    sequences = split['history']
    num_users = torch.cat([rows['userId'] for rows in sequences.values()]).unique().numel()
    return sequences, num_users, split['meta']['num_items']


def load_amazon_sequences(category, split_config):
    """
    Load the user history rows of an Amazon category under the given SplitConfig.

    protocol 'loo': per-user leave-one-out splits 'train', 'eval', 'test'.
    protocol 'gts': global temporal split with the splits 'train', 'eval', 'test', 'test_cold'.
    Row layout and rules: data/protocols.py. The histories are complete; the length of the
    model input is set by the collate function (data/sequence.py).

    Returns:
        tuple: (sequences, num_users, num_items)
            sequences: Dict of split name to rows (userId, itemId, itemId_fut, ...)
            num_users: Number of distinct users with at least one row
            num_items: Total number of items
    """
    return split_sequences(load_amazon_split(category, split_config))


def _onion_tag(subset):
    """<start date>_<end date>_k<k>: the part of the file names that fixes the events."""
    return f"{subset['start'][:10]}_{subset['end'][:10]}_k{subset['k']}"


def onion_data_path(subset):
    """data_onion_<start date>_<end date>_k<k>_<feature>.pt in the Onion processed folder."""
    feature = subset['features'].removeprefix("id_").removesuffix(".tsv.bz2")
    return f"{ONION_PROCESSED_DIR}/data_onion_{_onion_tag(subset)}_{feature}.pt"


def _load_onion_data(subset):
    """
    The data file of a Music4All-Onion subset: track ids, sequences (data/onion_interactions.py),
    item features in track id order (data/onion_features.py), the md5 of the sequences and
    a meta. Built from the raw files on first use. A file whose meta differs from the
    current format version or subset raises a ValueError.
    """
    path = onion_data_path(subset)
    meta = {"format_version": ONION_FORMAT_VERSION, **subset}
    if os.path.exists(path):
        data = torch.load(path, weights_only=False)
        check_meta(data, meta, path)
        return data

    raw = f"{ONION_ROOT}/raw"
    sequences, track_ids = read_onion_interactions(
        f"{raw}/{ONION_EVENTS}", subset['start'], subset['end'], subset['k'])
    data = {
        "track_ids": track_ids,
        "sequences": sequences,
        "x": read_onion_features(f"{raw}/{subset['features']}", track_ids),
        "sequences_md5": hashlib.md5(repr(sequences).encode()).hexdigest(),
        "meta": meta,
    }
    save_atomic(data, path)
    return data


def _load_onion_data_and_split(split_config, subset):
    """
    The data file and the cached split of a Music4All-Onion subset. The split file name
    carries the md5 of the sequences, so changed sequences never reuse an old split: their
    item count and item ids, the only inputs the split meta checks, can stay the same.
    """
    if split_config.protocol != 'gts':
        raise ValueError(f"Music4All-Onion supports only the global temporal split ('gts'), "
                         f"got {split_config.protocol!r}")
    data = _load_onion_data(subset)
    num_items = len(data['track_ids'])
    split = load_or_build_split(
        name=f"onion_{_onion_tag(subset)}_{data['sequences_md5'][:8]}",
        split_config=split_config,
        processed_dir=ONION_PROCESSED_DIR,
        read_sequences=lambda: data['sequences'],
        # the model item id of a track is its position in track_ids, without a permutation
        id_permutation=torch.arange(num_items),
        num_items=num_items,
    )
    return data, split


def load_onion_split(split_config, subset=ONION_SUBSET):
    """The Stage-2 split of a Music4All-Onion subset, built on first use and cached."""
    return _load_onion_data_and_split(split_config, subset)[1]


def load_onion(split_config, input_normalization, split='train', subset=ONION_SUBSET):
    """
    The item features of a Music4All-Onion subset: 'train' are the Stage-1 (warm) items of
    the global temporal split, 'test' the others, 'all' every item.
    input_normalization: see normalize_items; its statistics come from the 'train' items.
    """
    data, gts_split = _load_onion_data_and_split(split_config, subset)
    return select_items(data['x'], gts_split['stage1_items'], input_normalization, split)


def load_onion_sequences(split_config, subset=ONION_SUBSET):
    """
    The user history rows of a Music4All-Onion subset under the global temporal split;
    same return value as load_amazon_sequences.
    """
    return split_sequences(load_onion_split(split_config, subset))

def load_movie_lens(category='1M', dimension="user", train=True, raw=True):
    # Build the file path
    sub_folder = "raw" if raw else "processed"
    path = fr"dataset/ml-{category}/{sub_folder}/ml-{category}.{dimension}"
    
    if not raw and os.path.exists(path):
        return torch.load(path, weights_only=False)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please ensure the dataset is downloaded and placed correctly.")

    # Load the dataset
    data = pd.read_csv(path, sep='\t', index_col=0)
    print(data.columns)

    # Load pretrained Sentence-T5 model
    model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    # Build textual inputs based on the dimension
    if dimension == "user":
        texts = data.apply(
            lambda row: f"""User is a {row['age:token']}-year-old \
{'male' if row['gender:token'] == 'M' else 'female'} \
{row['occupation:token']} \
living in zip code {row['zip_code:token']}.""",
            axis=1
        ).tolist()
    elif dimension == "item":
        texts = data.apply(
            lambda row: f"""The movie {row['movie_title:token_seq']} was released in {row['release_year:token']} \
and has mostly regarded these genres: {row['genre:token_seq']}.""",
            axis=1
        ).tolist()
    elif dimension == "relation":
        texts = data.apply(lambda row: f"{row['relation_nl:token']}", axis=1).tolist()
    elif dimension == "entity":
        raise NotImplementedError(f"{dimension}-based embeddings not supported yet.")
    else:
        raise ValueError("Invalid dimension. Choose from 'user', 'item', 'relation', or 'entity'.")

    # Generate embeddings
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    torch.save(embeddings, fr"dataset/ml-{category}/processed/ml-{category}.{dimension}")

    torch.cuda.empty_cache()
    return embeddings
    