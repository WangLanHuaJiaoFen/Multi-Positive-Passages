import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding

from .arguments import DataArguments
from .trainer import TevatronTrainer

import logging

logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizer,
        trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer
        self.j = 0

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.prepare_for_model(
            text_encoding,
            truncation="only_first",
            max_length=(
                self.data_args.q_max_len if is_query else self.data_args.p_max_len
            ),
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group["query"]
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages: List[BatchEncoding] = []
        group_positives = group["positives"]
        group_negatives = group["negatives"]

        # Select the positive passages for this query according to the
        # configured multi-positive sampling mode.
        if self.data_args.positive_type == "single":
            pos_candidates = [group_positives[0]]
        elif self.data_args.positive_type in {"random_one", "multiple", "pair_wise"}:
            group_size = self.data_args.train_n_passages
            
            if self.data_args.max_positive_num > 0:
                if len(group_positives) >= self.data_args.max_positive_num:
                    pos_candidates = group_positives[: self.data_args.max_positive_num]
                else:
                    pos_candidates = group_positives
                if self.data_args.positive_type == "random_one":
                    idx = (_hashed_seed + epoch) % len(pos_candidates)
                    pos_candidates = [pos_candidates[idx]]
                if self.j < 20:
                    self.j += 1
                    print(f"pos num: {len(pos_candidates)}")
            elif self.data_args.max_positive_num == -1:
                if len(group_positives) >= group_size:
                    pos_candidates = group_positives[: group_size - 1]
                else:
                    pos_candidates = group_positives
        else:
            raise ValueError(
                f"Unsupported positive_type: {self.data_args.positive_type}"
            )

        # Keep positives at the beginning of the group so the loss can recover
        # positive positions from the per-query positive count.
        positive_cnt = len(pos_candidates)
        for pos in pos_candidates:
            pos_example = self.create_one_example(pos)
            pos_example["is_positive"] = 1
            encoded_passages.append(pos_example)

        # Fill the remaining slots with negatives; sample with replacement when
        # the example does not contain enough negatives.
        negative_size = max(0, self.data_args.train_n_passages - positive_cnt)

        if negative_size <= 0 or self.data_args.train_n_passages == 1:
            negs = []
        elif not group_negatives:
            negs = []
        elif len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = (epoch * negative_size) % len(group_negatives)
            negs = list(group_negatives)
            random.Random(_hashed_seed).shuffle(negs)
            negs = (negs * 2)[_offset : _offset + negative_size]

        for neg_psg in negs:
            neg_example = self.create_one_example(neg_psg)
            neg_example["is_positive"] = 0
            encoded_passages.append(neg_example)

        return encoded_query, encoded_passages


class EncodeDataset(Dataset):
    input_keys = ["text_id", "text"]

    def __init__(
        self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128
    ):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = self.tok.prepare_for_model(
            text,
            max_length=self.max_len,
            truncation="only_first",
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that converts from List[Tuple[encode_qry, encode_psg]] to a batch of queries and passages.
    """

    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])

        # Record the number of positives for each query so multi-positive losses
        # can rebuild query-local positive masks after passages are flattened.
        positive_counts = []
        if isinstance(dd[0], list):
            flat_dd = []
            for passages in dd:
                pos_cnt = 0
                for passage in passages:
                    pos_cnt += int(passage.pop("is_positive", 0))
                    flat_dd.append(passage)
                positive_counts.append(pos_cnt)
            dd = flat_dd
        else:
            for passage in dd:
                positive_counts.append(int(passage.pop("is_positive", 0)))

        q_collated = self.tokenizer.pad(
            qq,
            padding="max_length",
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding="max_length",
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        d_collated["positive_counts"] = torch.tensor(positive_counts, dtype=torch.long)

        return q_collated, d_collated


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        return text_ids, collated_features
