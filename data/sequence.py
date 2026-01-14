import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class SemanticIDSequenceDataset(Dataset):
    """
    Dataset for TIGER Seq2Seq training.
    Maps raw item IDs in user history to Semantic ID tuples.
    """
    def __init__(self, history_data, semantic_ids, max_len=20, mode='train'):
        """
        Args:
            history_data (dict): Dictionary containing sequence data (from AmazonReviews)
            semantic_ids (torch.Tensor): Tensor of shape [num_items, codebook_layers]
            max_len (int): Maximum sequence length
            mode (str): 'train', 'eval', or 'test'
        """
        self.semantic_ids = semantic_ids
        self.max_len = max_len
        self.mode = mode
        
        # load data from the specific split
        self.user_ids = history_data[mode]["userId"]
        self.item_seqs = history_data[mode]["itemId"]
        self.target_items = history_data[mode]["itemId_fut"]
        
        self.num_items = semantic_ids.shape[0]
        self.codebook_layers = semantic_ids.shape[1]

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        """
        Retrieves and maps a user sequence to Semantic ID tuples.
        Returns:
            history_tuples (torch.Tensor): [T, codebook_layers]
            target_tuple (torch.Tensor): [codebook_layers]
            user_id (int): User ID
        """
        # get raw data
        user_id = self.user_ids[idx]
        raw_seq = self.item_seqs[idx]
        target_item = self.target_items[idx]
        
        # ensure raw_seq is a tensor
        if isinstance(raw_seq, list):
            raw_seq = torch.tensor(raw_seq, dtype=torch.long)
            
        # ensure target_item is a scalar integer
        if isinstance(target_item, torch.Tensor):
            target_item = target_item.item()
        
        # PROCESS SEQUENCE
        # last max_len items if the sequence is longer than max_len
        if len(raw_seq) > self.max_len:
            raw_seq = raw_seq[-self.max_len:]
        
        # create a mask to ignore padding (-1) or invalid IDs
        valid_mask = (raw_seq >= 0) & (raw_seq < self.num_items)
        
        # initialize tensor with padding (-1)
        seq_tuples = torch.full((len(raw_seq), self.codebook_layers), -1, dtype=torch.long)
        
        if valid_mask.any():
            # select all valid IDs
            valid_indices = raw_seq[valid_mask].long()
            # map to semantic IDs, invalid IDs remain as -1
            seq_tuples[valid_mask] = self.semantic_ids[valid_indices]

        # PROCESS TARGET ITEM
        # target is a single item ID
        if 0 <= target_item < self.num_items:
            target_tuple = self.semantic_ids[target_item]
        else:
            # set to zeros if invalid
            target_tuple = torch.zeros(self.codebook_layers, dtype=torch.long)
        
        return {
            "history_tuples": seq_tuples,
            "target_tuples": target_tuple.unsqueeze(0), # unsqueeze as decoder expects sequence
            "user_id": user_id
        }

def collate_fn(batch):
    """
    Custom collate function to pad sequences in the batch.
    Returns:
        dict with keys:
            'history_tuples': Padded tensor of shape [B, T, codebook_layers]
            'target_tuples': Tensor of shape [B, 1, codebook_layers]
            'user_ids': Tensor of shape [B]
    """
    history_tuples = [item['history_tuples'] for item in batch]
    target_tuples = [item['target_tuples'] for item in batch]
    user_ids = [item['user_id'] for item in batch]
    
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
    
    return {
        "history_tuples": padded_history,
        "target_tuples": stacked_targets,
        "user_ids": stacked_users
    }
