import torch
from torch.utils.data import Dataset


def get_max_history_len(config):
    """
    Length of the model-input history window: the number of most recent history items the
    recommender sees (config.seq2seq.max_history_len, 20 by default).
    """
    return config.seq2seq.get('max_history_len', 20)


class SemanticIDSequenceDataset(Dataset):
    """
    Dataset for TIGER Seq2Seq training.
    Maps the raw item IDs of a user's full history to Semantic ID tuples. The length of the
    model input is set by the collate function.
    """
    def __init__(self, history_data, semantic_ids, mode='train'):
        """
        Args:
            history_data (dict): Split name to rows in the row layout of data/protocols.py
                (userId, itemId, itemId_fut, optional is_last)
            semantic_ids (torch.Tensor): Tensor of shape [num_items, codebook_layers]
            mode (str): 'train', 'eval', 'test' or 'test_cold'
        """
        self.semantic_ids = semantic_ids
        self.mode = mode
        
        # load data from the specific split
        self.user_ids = history_data[mode]["userId"]
        self.item_seqs = history_data[mode]["itemId"]
        self.target_items = history_data[mode]["itemId_fut"]
        
        # evaluation-only fields (eval and test modes)
        self.with_eval_fields = (mode != 'train')
        self.is_last = history_data[mode].get("is_last", None)
        
        self.codebook_layers = semantic_ids.shape[1]

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        """
        Retrieves and maps a user sequence to Semantic ID tuples.
        Returns:
            history_tuples (torch.Tensor): [T, codebook_layers], the full history
            target_tuple (torch.Tensor): [codebook_layers]
            user_id (int): User ID
            seen_tuples (torch.Tensor): [T, codebook_layers] (eval/test modes only)
            is_last (bool): last target of the user (eval/test modes only)
        """
        history = torch.as_tensor(self.item_seqs[idx], dtype=torch.long)
        history_tuples = self.semantic_ids[history]
        target_tuple = self.semantic_ids[int(self.target_items[idx])]
        
        item = {
            "history_tuples": history_tuples,
            "target_tuples": target_tuple.unsqueeze(0), # unsqueeze as decoder expects sequence
            "user_id": self.user_ids[idx]
        }
        
        if self.with_eval_fields:
            # keep the full history for metric calculation, as collate_fn cuts history_tuples
            item["seen_tuples"] = history_tuples
            item["is_last"] = bool(self.is_last[idx]) if self.is_last is not None else True
        
        return item


def collate_fn(batch, max_len=None):
    """
    Custom collate function to pad sequences in the batch.
    Cuts every history to its last max_len items (the model input window) if max_len is given.
    Returns:
        dict with keys:
            'history_tuples': Padded tensor of shape [B, T, codebook_layers]
            'target_tuples': Tensor of shape [B, 1, codebook_layers]
            'user_ids': Tensor of shape [B]
        and, if the items carry them (eval/test datasets):
            'seen_tuples': Tensor of shape [B, S_max, codebook_layers], padded with -1
            'is_last': BoolTensor of shape [B]
    """
    history_tuples = [item['history_tuples'] for item in batch]
    target_tuples = [item['target_tuples'] for item in batch]
    user_ids = [item['user_id'] for item in batch]
    
    if max_len is not None:
        history_tuples = [h[-max_len:] for h in history_tuples]
        
    # pad history sequences
    # expects input [T, codebook_layers], pads with -1
    padded_history = torch.nn.utils.rnn.pad_sequence(
        history_tuples, 
        batch_first=True, 
        padding_value=-1
    )
        
    # stack target tuples and user IDs
    stacked_targets = torch.stack(target_tuples, dim=0)
    stacked_users = torch.tensor(user_ids, dtype=torch.long)
    
    out = {
        "history_tuples": padded_history,
        "target_tuples": stacked_targets,
        "user_ids": stacked_users
    }
    
    # evaluation-only fields (not cut to max_len)
    if 'seen_tuples' in batch[0]:
        out["seen_tuples"] = torch.nn.utils.rnn.pad_sequence(
            [item['seen_tuples'] for item in batch],
            batch_first=True,
            padding_value=-1
        )
    if 'is_last' in batch[0]:
        out["is_last"] = torch.tensor([item['is_last'] for item in batch], dtype=torch.bool)
    
    return out


def collate_fn_with_augmentation(batch, max_len=None):
    """
    Custom collate function that performs data augmentation by sampling
    contiguous sub-sequences from user histories.
    """
    max_batch_size = len(batch)
    full_seqs = []
    user_ids = []
    full_lengths = []
    for item in batch:
        hist = item["history_tuples"]
        targ = item["target_tuples"]
        user_id = item["user_id"]
        
        # concatenate history and target
        full_seq = torch.cat([hist, targ], dim=0) # shape [T + 1, codebook_layers]
        full_seqs.append(full_seq)
        user_ids.append(user_id)
        full_lengths.append(full_seq.shape[0])

    # calculate sub-sequence counts for each sequence
    counts = [N * (N - 1) // 2 for N in full_lengths]
    total_subseqs = sum(counts)
    
    if total_subseqs == 0:
        # fallback: if no augmentation is possible, use normal collate
        return collate_fn(batch, max_len=max_len)

    # sample indices for sub-sequences to include in the batch
    if total_subseqs > max_batch_size:
        select_seqs = torch.randint(low=0, high=total_subseqs, size=(max_batch_size,))
    else:
        select_seqs = torch.arange(total_subseqs)

    # MAP INDICES TO SLICES
    # generate cumulative array of boundaries
    cum_counts = [0]
    for c in counts:
        cum_counts.append(cum_counts[-1] + c)
    cum_counts_t = torch.tensor(cum_counts)

    # find which sequence each selected index belongs to
    # searchsorted returns the index of the first element in cum_counts_t that is greater than select_seqs
    # we subtract 1 to get the correct sequence index
    row_indices = torch.searchsorted(cum_counts_t, select_seqs, right=True) - 1
    
    new_history_tuples = []
    new_target_tuples = []
    new_user_ids = []

    for idx_in_selection, row_idx in enumerate(row_indices):
        # get the sequence index and their original history length
        row_idx = row_idx.item()
        N = full_lengths[row_idx]
        # get the local index within the selected sequence
        global_idx = select_seqs[idx_in_selection].item()
        local_idx = global_idx - cum_counts[row_idx]
        
        # number of valid sub-sequences that can start at index 'start' is (N - start - 1)
        # this block size is subtracted from local_idx until it fits inside the current block
        start = 0
        while local_idx >= (N - start - 1):
            local_idx -= (N - start - 1)
            start += 1
        
        # shortest valid sequence requires 1 history item + 1 target item -> start + 2
        # local_idx is the remainder which determines how many additional items to include in the history beyond the minimum
        end = start + 2 + local_idx
        
        full_seq = full_seqs[row_idx]
        sliced_hist = full_seq[start : end - 1]
        sliced_targ = full_seq[end - 1 : end]
        
        new_history_tuples.append(sliced_hist)
        new_target_tuples.append(sliced_targ)
        new_user_ids.append(user_ids[row_idx])

    # trim to max_len if specified
    if max_len is not None:
        new_history_tuples = [h[-max_len:] if h.size(0) > max_len else h for h in new_history_tuples]

    padded_history = torch.nn.utils.rnn.pad_sequence(
        new_history_tuples, 
        batch_first=True, 
        padding_value=-1
    )
        
    stacked_targets = torch.stack(new_target_tuples, dim=0)
    stacked_users = torch.tensor(new_user_ids, dtype=torch.long)
    
    return {
        "history_tuples": padded_history,
        "target_tuples": stacked_targets,
        "user_ids": stacked_users
    }


class SASRecDataset(Dataset):
    """
    Dataset for SASRec benchmark.
    Returns left-padded sequences of 1-based item IDs (0 = padding).
    """

    def __init__(self, history_data, num_items, max_len=50, mode='train'):
        """
        Args:
            history_data: dict with splits, each containing userId, itemId (the full history), itemId_fut
            num_items: total number of items in the dataset
            max_len: length of the model input (the last max_len items of the history)
            mode: 'train', 'eval', or 'test'
        """
        self.num_items = num_items
        self.max_len = max_len
        self.mode = mode

        self.user_ids = history_data[mode]['userId']
        self.item_seqs = history_data[mode]['itemId']
        self.target_items = history_data[mode]['itemId_fut']

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        raw_seq = torch.as_tensor(self.item_seqs[idx], dtype=torch.long)
        target = int(self.target_items[idx])

        # the last max_len items of the history, as 1-based IDs
        items_1based = raw_seq[-self.max_len:] + 1
        seq_len = len(items_1based)

        # left-pad to max_len with 0
        item_seq = torch.zeros(self.max_len, dtype=torch.long)
        if seq_len > 0:
            item_seq[-seq_len:] = items_1based

        return {
            'item_seq': item_seq,
            'target_item': target + 1,  # 1-based
            'seq_len': seq_len,
        }


def sasrec_collate_fn(batch):
    """Collate function for SASRecDataset. Stacks fixed-length sequences."""
    return {
        'item_seq': torch.stack([b['item_seq'] for b in batch]),
        'target_item': torch.tensor([b['target_item'] for b in batch], dtype=torch.long),
        'seq_len': torch.tensor([b['seq_len'] for b in batch], dtype=torch.long),
    }
