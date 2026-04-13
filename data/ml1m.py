import numpy as np
import pandas as pd
import polars as pl
import torch
from typing import List

from data.preprocessing import encode_text_embeddings, sequence_df_to_tensor_dict
from data.schemas import FUT_SUFFIX
from torch_geometric.data import HeteroData
from torch_geometric.datasets import MovieLens1M


class RawMovieLens1M(MovieLens1M):
    MOVIE_HEADERS = ["movieId", "title", "genres"]
    USER_HEADERS = ["userId", "gender", "age", "occupation", "zipCode"]
    RATING_HEADERS = ['userId', 'movieId', 'rating', 'timestamp']

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        force_reload=False,
        split=None
    ) -> None:
        super(RawMovieLens1M, self).__init__(
            root, transform, pre_transform, force_reload
        )

    def _load_ratings(self):
        return pd.read_csv(
            self.raw_paths[2],
            sep='::',
            header=None,
            names=self.RATING_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )

    @staticmethod
    def _remove_low_occurrence(source_df, target_df, index_col):
        if isinstance(index_col, str):
            index_col = [index_col]
        out = target_df.copy()
        for col in index_col:
            count = source_df.groupby(col).agg(ratingCnt=("rating", "count"))
            high_occ = count[count["ratingCnt"] >= 5]
            out = out.merge(high_occ, on=col).drop(columns=["ratingCnt"])
        return out

    @staticmethod
    def _ordered_train_test_split(df, on, train_split=0.8):
        threshold = df.select(pl.quantile(on, train_split)).item()
        return df.with_columns(is_train=pl.col(on) <= threshold)

    @staticmethod
    def _rolling_window(group, features, window_size=200, stride=1):
        assert group["userId"].nunique() == 1, "Found data for too many users"

        if len(group) < window_size:
            window_size = len(group)
            stride = 1
        n_windows = (len(group)+1-window_size)//stride
        feats = group[features].to_numpy().T
        windows = np.lib.stride_tricks.as_strided(
            feats,
            shape=(len(features), n_windows, window_size),
            strides=(feats.strides[0], 8*stride, 8*1)
        )
        feat_seqs = np.split(windows, len(features), axis=0)
        rolling_df = pd.DataFrame({
            name: pd.Series(
                np.split(feat_seqs[i].squeeze(0), n_windows, 0)
            ).map(torch.tensor) for i, name in enumerate(features)
        })
        return rolling_df

    @staticmethod
    def _generate_user_history(
        ratings_df,
        features: List[str] = ["movieId", "rating"],
        window_size: int = 200,
        stride: int = 1,
        train_split: float = 0.8,
    ) -> torch.Tensor:

        if isinstance(ratings_df, pd.DataFrame):
            ratings_df = pl.from_pandas(ratings_df)

        grouped_by_user = (ratings_df
            .sort("userId", "timestamp")
            .group_by_dynamic(
                index_column=pl.int_range(pl.len()),
                every=f"{stride}i",
                period=f"{window_size}i",
                by="userId")
            .agg(
                *(pl.col(feat) for feat in features),
                seq_len=pl.col(features[0]).len(),
                max_timestamp=pl.max("timestamp")
            )
        )

        max_seq_len = grouped_by_user.select(pl.col("seq_len").max()).item()
        split_grouped_by_user = RawMovieLens1M._ordered_train_test_split(grouped_by_user, "max_timestamp", 0.8)
        padded_history = (split_grouped_by_user
            .with_columns(pad_len=max_seq_len-pl.col("seq_len"))
            .filter(pl.col("is_train").or_(pl.col("seq_len") > 1))
            .select(
                pl.col("userId"),
                pl.col("max_timestamp"),
                pl.col("is_train"),
                *(pl.when(pl.col("is_train"))
                    .then(
                        pl.col(feat).list.concat(
                            pl.lit(-1, dtype=pl.Int64).repeat_by(pl.col("pad_len"))
                        ).list.to_array(max_seq_len)
                    ).otherwise(
                        pl.col(feat).list.slice(0, pl.col("seq_len")-1).list.concat(
                            pl.lit(-1, dtype=pl.Int64).repeat_by(pl.col("pad_len")+1)
                        ).list.to_array(max_seq_len)
                    )
                    for feat in features
                ),
                *(pl.when(pl.col("is_train"))
                    .then(
                        pl.lit(-1, dtype=pl.Int64)
                    )
                    .otherwise(
                        pl.col(feat).list.get(-1)
                    ).alias(feat + FUT_SUFFIX)
                    for feat in features
                )
            )
        )

        out = {}
        out["train"] = sequence_df_to_tensor_dict(
            padded_history.filter(pl.col("is_train")),
            features
        )
        out["eval"] = sequence_df_to_tensor_dict(
            padded_history.filter(pl.col("is_train").not_()),
            features
        )

        return out

    def process(self, max_seq_len=None) -> None:
        data = HeteroData()
        ratings_df = self._load_ratings()

        # Process movie data:
        full_df = pd.read_csv(
            self.raw_paths[0],
            sep='::',
            header=None,
            index_col='movieId',
            names=self.MOVIE_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )
        df = self._remove_low_occurrence(ratings_df, full_df, "movieId")
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        genres = torch.from_numpy(df["genres"].str.get_dummies('|').values).to(torch.float)

        titles_text = df["title"].apply(lambda s: s.split("(")[0].strip()).tolist()
        titles_emb = encode_text_embeddings(titles_text)

        x = torch.cat([titles_emb, genres], axis=1)

        data['item'].x = x
        # Process user data:
        full_df = pd.read_csv(
            self.raw_paths[1],
            sep='::',
            header=None,
            index_col='userId',
            names=self.USER_HEADERS,
            dtype='str',
            encoding='ISO-8859-1',
            engine='python',
        )
        df = self._remove_low_occurrence(ratings_df, full_df, "userId")
        user_mapping = {idx: i for i, idx in enumerate(df.index)}

        age = df['age'].str.get_dummies().values.argmax(axis=1)[:, None]
        age = torch.from_numpy(age).to(torch.float)

        gender = df['gender'].str.get_dummies().values[:, 0][:, None]
        gender = torch.from_numpy(gender).to(torch.float)

        occupation = df['occupation'].str.get_dummies().values.argmax(axis=1)[:, None]
        occupation = torch.from_numpy(occupation).to(torch.float)

        data['user'].x = torch.cat([age, gender, occupation], dim=-1)

        self.int_user_data = df
        # Process rating data:
        df = self._remove_low_occurrence(
            ratings_df,
            ratings_df,
            ["userId", "movieId"]
        )
        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = torch.tensor([src, dst])
        data['user', 'rates', 'item'].edge_index = edge_index

        rating = torch.from_numpy(df['rating'].values).to(torch.long)
        data['user', 'rates', 'item'].rating = rating

        time = torch.from_numpy(df['timestamp'].values)
        data['user', 'rates', 'item'].time = time

        data['item', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        data['item', 'rated_by', 'user'].rating = rating
        data['item', 'rated_by', 'user'].time = time

        df["itemId"] = df["movieId"].apply(lambda x: movie_mapping[x])

        data["user", "rated", "item"].history = self._generate_user_history(
            df,
            features=["itemId", "rating"],
            window_size=max_seq_len if max_seq_len is not None else 1
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
