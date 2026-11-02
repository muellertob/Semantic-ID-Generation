import gzip
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import torch

from data.preprocessing import encode_text_embeddings
from torch_geometric.data import download_google_url
from torch_geometric.data import extract_zip
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from typing import Callable
from typing import List
from typing import Optional


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


class AmazonReviews(InMemoryDataset):
    gdrive_id = "1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G"
    gdrive_filename = "P5_data.zip"

    def __init__(
        self,
        root: str,
        split: str = 'beauty',  # 'beauty', 'sports', 'toys'
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.split = split
        super(AmazonReviews, self).__init__(
            root, transform, pre_transform, force_reload
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.split]

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.split}.pt'

    def download(self) -> None:
        path = download_google_url(self.gdrive_id, self.root, self.gdrive_filename)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'data')
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def _remap_ids(self, x):
        """
        Maps original 1-based P5 IDs to new 0-based IDs.
        If self.id_permutation is set, it uses a random mapping to avoid leakage.
        """
        original_idx = int(x) - 1
        if hasattr(self, 'id_permutation'):
            return self.id_permutation[original_idx].item()
        return original_idx

    def process(self) -> None:
        data = HeteroData()

        with open(os.path.join(self.raw_dir, self.split, "datamaps.json"), 'r') as f:
            data_maps = json.load(f)

        # generate a deterministic random permutation to avoid ID leakage.
        num_items = len(data_maps["item2id"])
        gen = torch.Generator()
        gen.manual_seed(42)
        self.id_permutation = torch.randperm(num_items, generator=gen)

        # save the permutation to the data object for future reference/inference
        data["item"].id_permutation = self.id_permutation

        # compute item features
        asin2id = pd.DataFrame([{"asin": k, "id": self._remap_ids(int(v))} for k, v in data_maps["item2id"].items()])
        item_data = (
            pd.DataFrame([
                meta for meta in
                parse(path=os.path.join(self.raw_dir, self.split, "meta.json.gz"))
            ])
            .merge(asin2id, on="asin")
            .sort_values(by="id")
            .fillna({"brand": "Unknown"})
        )

        sentences = item_data.apply(
            lambda row:
                "Title: " +
                str(row["title"]) + "; " +
                "Brand: " +
                str(row["brand"]) + "; " +
                "Categories: " +
                str(row["categories"][0]) + "; " +
                "Price: " +
                str(row["price"]) + "; ",
            axis=1
        )

        item_emb = encode_text_embeddings(sentences)

        data['item'].x = item_emb
        data['item'].text = np.array(sentences)

        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05

        self.save([data], self.processed_paths[0])

if __name__ == "__main__":
    AmazonReviews("dataset/amazon", split="beauty")
