from datasets import load_dataset, load_from_disk
from transformers import PreTrainedTokenizer
from .preprocessor import TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor
from ..arguments import DataArguments

DEFAULT_PROCESSORS = [TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor]
# Dataset-specific processor registry. Entries map each dataset name to the
# train/query/corpus preprocessors expected by the downstream datasets.
PROCESSOR_INFO = {
    "Tevatron/wikipedia-nq": DEFAULT_PROCESSORS,
    "Tevatron/wikipedia-trivia": DEFAULT_PROCESSORS,
    "Tevatron/wikipedia-curated": DEFAULT_PROCESSORS,
    "Tevatron/wikipedia-wq": DEFAULT_PROCESSORS,
    "Tevatron/wikipedia-squad": DEFAULT_PROCESSORS,
    "Tevatron/scifact": DEFAULT_PROCESSORS,
    "Tevatron___msmarco-passage": DEFAULT_PROCESSORS,
    "nq-annotation": DEFAULT_PROCESSORS,
    "msmarco_annotation": DEFAULT_PROCESSORS,
    "msmarco_fusion": DEFAULT_PROCESSORS,
    "nq_corpus": DEFAULT_PROCESSORS,
    "nq-test": DEFAULT_PROCESSORS,
    "dl19": DEFAULT_PROCESSORS,
    "dl20": DEFAULT_PROCESSORS,
    "json": [None, None, None],
}


class HFTrainDataset:
    def __init__(
        self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str
    ):
        data_files = data_args.train_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        # Load the training split. Several local experimental datasets are
        # stored with `save_to_disk`, while public Tevatron datasets are loaded
        # through the HuggingFace datasets API.
        # self.dataset = load_dataset(data_args.dataset_name,
        #                             data_args.dataset_language,
        #                             data_files=data_files, cache_dir=cache_dir, use_auth_token=True)[data_args.dataset_split]
        if "nq-annotation" in data_args.dataset_name:
            self.dataset = load_from_disk(
                dataset_path=data_args.dataset_name,
            )["train"]
        if "msmarco_annotation" in data_args.dataset_name:
            self.dataset = load_from_disk(
                dataset_path=data_args.dataset_name,
            )["train"]
        if "msmarco_fusion" in data_args.dataset_name:
            self.dataset = load_from_disk(
                dataset_path=data_args.dataset_name,
            )["train"]
        if "Tevatron___msmarco-passage" in data_args.dataset_name:
            self.dataset = load_dataset(data_args.dataset_name)["train"]
        # Pick the registered preprocessor for this dataset, falling back to the
        # default Tevatron train/query/corpus processors when no override exists.
        self.preprocessor = (
            PROCESSOR_INFO[data_args.dataset_name][0]
            if data_args.dataset_name in PROCESSOR_INFO
            else DEFAULT_PROCESSORS[0]
        )
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(
            self.tokenizer,
            data_args.passage_field_separator,
            data_args.passage_field_separator,
        )

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            # Tokenize each example into the compact fields consumed by
            # TrainDataset: query token ids, positive passages, and negatives.
            self.dataset = self.dataset.map(
                self.preprocessor(
                    self.tokenizer, self.q_max_len, self.p_max_len, self.separator
                ),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset


class HFQueryDataset:
    def __init__(
        self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str
    ):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        # self.dataset = load_dataset(
        #     data_args.dataset_name,
        #     data_args.dataset_language,
        #     data_files=data_files,
        #     cache_dir=cache_dir,
        #     use_auth_token=True,
        # )[data_args.dataset_split]
        if "Tevatron___msmarco-passage" in data_args.dataset_name:
            self.dataset = load_dataset(
                data_args.dataset_name,
                cache_dir=cache_dir
            )["validation"]
        if "dl19" in data_args.dataset_name:
            self.dataset = load_from_disk(
                dataset_path=data_args.dataset_name,
            )["train"]
        if "dl20" in data_args.dataset_name:
            self.dataset = load_from_disk(
                dataset_path=data_args.dataset_name,
            )["train"]
        self.preprocessor = (
            PROCESSOR_INFO[data_args.dataset_name][1]
            if data_args.dataset_name in PROCESSOR_INFO
            else DEFAULT_PROCESSORS[1]
        )
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.proc_num = data_args.dataset_proc_num

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
            )
        return self.dataset


class HFCorpusDataset:
    def __init__(
        self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str
    ):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        # self.dataset = load_dataset(
        #     data_args.dataset_name,
        #     data_args.dataset_language,
        #     data_files=data_files,
        #     cache_dir=cache_dir,
        #     use_auth_token=True,
        # )[data_args.dataset_split]
        if "nq_corpus" in data_args.dataset_name:
            self.dataset = load_dataset(
                data_args.dataset_name,
                cache_dir=cache_dir
            )["train"]
        if "msmarco-passage" in data_args.dataset_name:
            self.dataset = load_dataset(
                data_args.dataset_name,
                cache_dir=cache_dir
            )["train"]
        script_prefix = data_args.dataset_name
        if script_prefix.endswith("-corpus"):
            script_prefix = script_prefix[:-7]
        self.preprocessor = (
            PROCESSOR_INFO[script_prefix][2]
            if script_prefix in PROCESSOR_INFO
            else DEFAULT_PROCESSORS[2]
        )
        self.tokenizer = tokenizer
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(
            self.tokenizer,
            data_args.passage_field_separator,
            data_args.passage_field_separator,
        )

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
            )
        return self.dataset
