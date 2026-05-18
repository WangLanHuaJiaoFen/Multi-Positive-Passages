import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PreTrainedModel, AutoModel
from transformers.file_utils import ModelOutput

from tevatron.arguments import (
    ModelArguments,
    TevatronTrainingArguments as TrainingArguments,
    DataArguments,
)

import logging

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):  # Structured return value from forward.
    q_reps: Optional[Tensor] = None  # Query representations.
    p_reps: Optional[Tensor] = None  # Passage representations.
    loss: Optional[Tensor] = None  # Training loss.
    scores: Optional[Tensor] = None  # Pairwise similarity scores.


class EncoderPooler(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderPooler, self).__init__()
        self._config = {}

    def forward(self, q_reps, p_reps):
        raise NotImplementedError("EncoderPooler is an abstract class")

    def load(self, model_dir: str):
        pooler_path = os.path.join(model_dir, "pooler.pt")
        if pooler_path is not None:
            if os.path.exists(pooler_path):
                logger.info(f"Loading Pooler from {pooler_path}")
                state_dict = torch.load(pooler_path, map_location="cpu")
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, "pooler.pt"))
        with open(os.path.join(save_path, "pooler_config.json"), "w") as f:
            json.dump(self._config, f)


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        data_args: DataArguments,
        train_args: TrainingArguments,
        lm_q: PreTrainedModel,  # Query encoder.
        lm_p: PreTrainedModel,  # Passage encoder.
        pooler: nn.Module = None,  # Optional projection/pooling layer.
        untie_encoder: bool = False,  # Whether query and passage encoders are separate.
        negatives_x_device: bool = False,  # Whether negatives are shared across devices.
    ):
        super().__init__()
        self.data_args = data_args
        self.train_args = train_args
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")  # NLL + log_softmax
        self.negatives_x_device = negatives_x_device
        self.untie_encoder = untie_encoder

        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError(
                    "Distributed training has not been initialized for representation all gather."
                )
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(
        self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None
    ):
        positive_counts = None
        if passage is not None and hasattr(passage, "pop"):
            # The collator attaches one positive count per query; remove it from
            # the model input before passing tensors to the transformer.
            positive_counts = passage.pop("positive_counts", None)

        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        # for training
        if self.training:
            if self.negatives_x_device:
                # Gather representations from all ranks so each query can use
                # other devices' passages as additional negatives.
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            # Build retrieval logits: each row contains scores from one query to
            # all candidate passages visible in the current batch.
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            if (
                self.data_args.positive_type == "single"
                and self.train_args.loss_type == "infonce"
            ) or (
                self.data_args.positive_type == "random_one"
                and self.train_args.loss_type == "random_one_lh"
            ):
                # Single-positive training path: each query's target is the
                # first passage in its local group, located at [0, k, 2k, ...].
                target = torch.arange(
                    scores.size(0), device=scores.device, dtype=torch.long
                )
                target = target * (p_reps.size(0) // q_reps.size(0))

                # Standard in-batch contrastive cross entropy.
                loss = self.compute_loss(scores, target)
                if self.negatives_x_device:
                    loss = loss * self.world_size  # counter average weight reduction
            elif (
                self.data_args.positive_type == "multiple"
                and self.train_args.loss_type == "joint_lh"
            ):
                # Number of passages contributed by each query before in-batch
                # negatives are mixed in the score matrix.
                passages_per_query = p_reps.size(0) // q_reps.size(0)

                # Multi-positive likelihood needs the collator-provided positive
                # count to identify each query's positive block.
                if positive_counts is None:
                    raise ValueError(
                        "joint_lh loss requires positive_counts provided by the dataloader."
                    )

                # Move counts beside scores and keep them as integer indices.
                positive_counts_tensor = positive_counts.to(
                    device=scores.device, dtype=torch.long
                )
                # Keep scalar inputs compatible with vectorized batch code.
                if positive_counts_tensor.dim() == 0:
                    positive_counts_tensor = positive_counts_tensor.view(1)
                positive_counts_tensor = positive_counts_tensor.contiguous()

                # Gather counts when representations were gathered across ranks.
                if self.negatives_x_device:
                    positive_counts_tensor = self._dist_gather_tensor(
                        positive_counts_tensor.unsqueeze(-1)
                    ).squeeze(-1)

                # After any gather, counts must still align one-to-one with rows.
                if positive_counts_tensor.size(0) != scores.size(0):
                    raise ValueError(
                        f"positive_counts size {positive_counts_tensor.size(0)} does not match "
                        f"batch size {scores.size(0)} after gathering."
                    )
                # Clamp invalid counts to the local group width.
                positive_counts_tensor = positive_counts_tensor.clamp(
                    min=0, max=passages_per_query
                )

                # Denominator over all visible passages for each query.
                lse = torch.logsumexp(scores, dim=1)

                # Local passage positions inside each query group.
                group_positions = torch.arange(passages_per_query, device=scores.device)
                # Starting offset for each query's own passage block.
                start_offsets = (
                    torch.arange(scores.size(0), device=scores.device)
                    * passages_per_query
                )
                # Indices that recover each query's own passage block from scores.
                gather_indices = start_offsets.unsqueeze(1) + group_positions.unsqueeze(
                    0
                )

                # Scores for passages originally paired with each query.
                block_scores = scores.gather(1, gather_indices)

                # Positives are stored first in every query-local passage block.
                positive_mask = group_positions.unsqueeze(
                    0
                ) < positive_counts_tensor.unsqueeze(1)
                # Sum only positive scores for the numerator term.
                positive_scores = block_scores * positive_mask.to(scores.dtype)
                positive_sum = positive_scores.sum(dim=1)

                # Joint likelihood: m * logsumexp(all passages) - sum(pos scores).
                positive_counts_float = positive_counts_tensor.to(scores.dtype)
                loss_per_query = positive_counts_float * lse - positive_sum
                # normalizaion
                if self.train_args.normalize_loss_by_positive:
                    loss_per_query = loss_per_query / positive_counts_float.clamp_min(
                        1.0
                    )

                # Ignore malformed examples that contain no positive passages.
                valid_mask = positive_counts_tensor > 0
                if valid_mask.any():
                    loss = loss_per_query[valid_mask].mean()
                else:
                    loss = scores.new_tensor(0.0)

                # Match the scaling convention used by distributed contrastive training.
                if self.negatives_x_device:
                    loss = loss * self.world_size
            elif (
                self.data_args.positive_type == "multiple"
                and self.train_args.loss_type == "sum_mar"
            ):
                # Number of passages contributed by each query before in-batch
                # negatives are mixed in the score matrix.
                passages_per_query = p_reps.size(0) // q_reps.size(0)

                # Sum-MAR needs positive counts to build the positive mask.
                if positive_counts is None:
                    raise ValueError(
                        "sum_mar loss requires positive_counts provided by the dataloader."
                    )

                # Move counts beside scores and keep them as integer indices.
                positive_counts_tensor = positive_counts.to(
                    device=scores.device, dtype=torch.long
                )
                # Keep scalar inputs compatible with vectorized batch code.
                if positive_counts_tensor.dim() == 0:
                    positive_counts_tensor = positive_counts_tensor.view(1)
                positive_counts_tensor = positive_counts_tensor.contiguous()

                # Gather counts when representations were gathered across ranks.
                if self.negatives_x_device:
                    positive_counts_tensor = self._dist_gather_tensor(
                        positive_counts_tensor.unsqueeze(-1)
                    ).squeeze(-1)

                # After any gather, counts must still align one-to-one with rows.
                if positive_counts_tensor.size(0) != scores.size(0):
                    raise ValueError(
                        f"positive_counts size {positive_counts_tensor.size(0)} does not match "
                        f"batch size {scores.size(0)} after gathering."
                    )
                # Clamp invalid counts to the local group width.
                positive_counts_tensor = positive_counts_tensor.clamp(
                    min=0, max=passages_per_query
                )

                # Denominator over all visible passages for each query.
                log_denominator = torch.logsumexp(scores, dim=1)

                # Local passage positions inside each query group.
                group_positions = torch.arange(passages_per_query, device=scores.device)
                # Starting offset for each query's own passage block.
                start_offsets = (
                    torch.arange(scores.size(0), device=scores.device)
                    * passages_per_query
                )
                # Indices that recover each query's own passage block from scores.
                gather_indices = start_offsets.unsqueeze(1) + group_positions.unsqueeze(
                    0
                )

                # Scores for passages originally paired with each query.
                block_scores = scores.gather(1, gather_indices)
                # Positives are stored first in every query-local passage block.
                positive_mask = group_positions.unsqueeze(
                    0
                ) < positive_counts_tensor.unsqueeze(1)

                # Use the dtype minimum as a finite stand-in for negative infinity.
                neg_inf = torch.finfo(block_scores.dtype).min
                # Mask non-positive positions out of the positive logsumexp.
                masked_positive_scores = torch.where(
                    positive_mask,
                    block_scores,
                    torch.full_like(block_scores, neg_inf),
                )
                # Numerator over all positive passages for each query.
                log_positive_sum = torch.logsumexp(masked_positive_scores, dim=1)

                # Negative log probability mass assigned to the positive set.
                loss_per_query = -(log_positive_sum - log_denominator)

                # Ignore malformed examples that contain no positive passages.
                valid_mask = positive_counts_tensor > 0
                if valid_mask.any():
                    loss = loss_per_query[valid_mask].mean()
                else:
                    loss = scores.new_tensor(0.0)

                # Match the scaling convention used by distributed contrastive training.
                if self.negatives_x_device:
                    loss = loss * self.world_size
            elif (
                self.data_args.positive_type == "pair_wise"
                and self.train_args.loss_type == "lsep"
            ):
                # Number of passages originally paired with each query.
                passages_per_query = p_reps.size(0) // q_reps.size(0)

                # Pair-wise LSEP needs the per-query positive count from the collator.
                if positive_counts is None:
                    raise ValueError(
                        "lsep loss requires positive_counts provided by the dataloader."
                    )

                # Move counts beside scores and keep them as a batch-length vector.
                positive_counts_tensor = positive_counts.to(
                    device=scores.device, dtype=torch.long
                )
                # Keep scalar inputs compatible with vectorized batch code.
                if positive_counts_tensor.dim() == 0:
                    positive_counts_tensor = positive_counts_tensor.view(1)
                positive_counts_tensor = positive_counts_tensor.contiguous()

                # Gather counts when representations were gathered across ranks.
                if self.negatives_x_device:
                    positive_counts_tensor = self._dist_gather_tensor(
                        positive_counts_tensor.unsqueeze(-1)
                    ).squeeze(-1)

                # After any gather, counts must still align one-to-one with rows.
                if positive_counts_tensor.size(0) != scores.size(0):
                    raise ValueError(
                        f"positive_counts size {positive_counts_tensor.size(0)} does not match "
                        f"batch size {scores.size(0)} after gathering."
                    )
                # Clamp invalid counts to the local group width.
                positive_counts_tensor = positive_counts_tensor.clamp(
                    min=0, max=passages_per_query
                )

                # Build local passage positions so each query's own passage block
                # can be recovered from the flattened score matrix.
                group_positions = torch.arange(
                    passages_per_query, device=scores.device
                )
                # Starting offset for each query's own passage block.
                start_offsets = (
                    torch.arange(scores.size(0), device=scores.device)
                    * passages_per_query
                )
                # gather_indices has shape (batch_size, passages_per_query).
                gather_indices = start_offsets.unsqueeze(1) + group_positions.unsqueeze(0)

                # Scores for passages originally paired with each query.
                block_scores = scores.gather(1, gather_indices)
                # Positives are stored first in every query-local passage block.
                positive_mask = group_positions.unsqueeze(0) < positive_counts_tensor.unsqueeze(1)

                # Mark positive positions in the full score matrix so in-batch
                # positives are excluded from the negative set.
                positive_indicator = torch.zeros_like(scores)
                positive_indicator.scatter_(
                    1, gather_indices, positive_mask.to(scores.dtype)
                )
                positive_indicator = positive_indicator.bool()
                # Everything not marked positive is treated as a negative.
                negative_indicator = ~positive_indicator

                # Use the dtype minimum as a finite stand-in for negative infinity.
                neg_inf = torch.finfo(scores.dtype).min

                # Mask positives out of the negative aggregation.
                neg_scores = torch.where(
                    negative_indicator,
                    scores,
                    torch.full_like(scores, neg_inf),
                )

                # The negative side can use either the hardest negative or all
                # negatives via logsumexp.
                if self.train_args.pair_wise_negative_selector == "max":
                    # Hardest-negative approximation.
                    log_negative_sum = neg_scores.max(dim=1).values
                elif self.train_args.pair_wise_negative_selector == "all":
                    # log(sum_n exp(score_n)) over all negatives.
                    log_negative_sum = torch.logsumexp(neg_scores, dim=1)
                else:
                    raise ValueError(
                        f"Unsupported pair_wise_negative_selector {self.train_args.pair_wise_negative_selector}"
                    )

                # Select how positives contribute to the pair-wise term.
                pos_selector = self.train_args.pair_wise_positive_selector
                if pos_selector == "all":
                    # log(sum_p exp(-score_p)) over all positives.
                    positive_scores_neg = torch.where(
                        positive_indicator,
                        -scores,
                        torch.full_like(scores, neg_inf),
                    )
                    log_positive_neg_sum = torch.logsumexp(positive_scores_neg, dim=1)
                elif pos_selector == "min":
                    # Use the lowest-scoring positive, which emphasizes the
                    # weakest positive passage for this query.
                    pos_scores = torch.where(
                        positive_indicator,
                        scores,
                        torch.full_like(scores, float("inf")),
                    )
                    pos_selected = pos_scores.min(dim=1).values
                    log_positive_neg_sum = -pos_selected
                elif pos_selector == "max":
                    # Use the highest-scoring positive as the representative
                    # positive for this query.
                    pos_scores = torch.where(
                        positive_indicator,
                        scores,
                        torch.full_like(scores, neg_inf),
                    )
                    pos_selected = pos_scores.max(dim=1).values
                    log_positive_neg_sum = -pos_selected
                else:
                    raise ValueError(f"Unsupported pair_wise_positive_selector {pos_selector}")

                # log_pairs corresponds to log(sum_{n,p} exp(score_n - score_p))
                # under the selected positive/negative aggregation.
                log_pairs = log_negative_sum + log_positive_neg_sum

                # Softplus gives the smooth LSEP loss and keeps the computation
                # numerically stable for large score gaps.
                loss_per_query = F.softplus(log_pairs)
                # Optional normalization prevents queries with many positives
                # from dominating the batch loss.
                if self.train_args.normalize_loss_by_positive:
                    loss_per_query = loss_per_query / positive_counts_tensor.to(scores.dtype).clamp_min(1.0)

                # Drop rows with no positives or invalid intermediate values.
                valid_mask = (
                    positive_counts_tensor > 0
                ) & torch.isfinite(log_negative_sum) & torch.isfinite(log_positive_neg_sum)
                if valid_mask.any():
                    loss = loss_per_query[valid_mask].mean()
                else:
                    loss = scores.new_tensor(0.0)

                # Match the scaling convention used by distributed contrastive training.
                if self.negatives_x_device:
                    loss = loss * self.world_size

            elif (
                self.data_args.positive_type == "multiple"
                and self.train_args.loss_type == "lsep"
            ):
                passages_per_query = p_reps.size(0) // q_reps.size(0)

                if positive_counts is None:
                    raise ValueError(
                        "lsep loss requires positive_counts provided by the dataloader."
                    )

                positive_counts_tensor = positive_counts.to(
                    device=scores.device, dtype=torch.long
                )
                if positive_counts_tensor.dim() == 0:
                    positive_counts_tensor = positive_counts_tensor.view(1)
                positive_counts_tensor = positive_counts_tensor.contiguous()

                if self.negatives_x_device:
                    positive_counts_tensor = self._dist_gather_tensor(
                        positive_counts_tensor.unsqueeze(-1)
                    ).squeeze(-1)

                if positive_counts_tensor.size(0) != scores.size(0):
                    raise ValueError(
                        f"positive_counts size {positive_counts_tensor.size(0)} does not match "
                        f"batch size {scores.size(0)} after gathering."
                    )
                positive_counts_tensor = positive_counts_tensor.clamp(
                    min=0, max=passages_per_query
                )

                # Local passage positions inside each query group.
                group_positions = torch.arange(passages_per_query, device=scores.device)
                # Starting offset for each query's own passage block.
                start_offsets = (
                    torch.arange(scores.size(0), device=scores.device)
                    * passages_per_query
                )
                # Each row indexes the passages originally paired with that query.
                gather_indices = start_offsets.unsqueeze(1) + group_positions.unsqueeze(
                    0
                )

                # Positives are stored first in every query-local passage block.
                positive_mask = group_positions.unsqueeze(
                    0
                ) < positive_counts_tensor.unsqueeze(1)

                # Mark positive positions in the full score matrix so in-batch
                # positives are excluded from the negative set.
                positive_indicator = torch.zeros_like(scores)
                positive_indicator.scatter_(
                    1, gather_indices, positive_mask.to(scores.dtype)
                )
                positive_indicator = positive_indicator.bool()
                # Everything not marked positive is treated as a negative.
                negative_indicator = ~positive_indicator

                # Use the dtype minimum as a finite stand-in for negative infinity
                # when masking values out of logsumexp.
                neg_inf = torch.finfo(scores.dtype).min

                # Aggregate only negative scores.
                negative_scores = torch.where(
                    negative_indicator,
                    scores,
                    torch.full_like(scores, neg_inf),
                )
                log_negative_sum = torch.logsumexp(negative_scores, dim=1)

                # Aggregate negative positive-scores for the pairwise
                # log(sum_{n,p} exp(score_n - score_p)) term.
                positive_scores_neg = torch.where(
                    positive_indicator,
                    -scores,
                    torch.full_like(scores, neg_inf),
                )
                log_positive_neg_sum = torch.logsumexp(positive_scores_neg, dim=1)

                # Combine positive and negative aggregations into the LSEP
                # pairwise log-sum term.
                log_pairs = log_negative_sum + log_positive_neg_sum

                # Softplus gives a non-negative, numerically stable LSEP loss.
                loss_per_query = F.softplus(log_pairs)

                # normalization
                if self.train_args.normalize_loss_by_positive:
                    loss_per_query = loss_per_query / positive_counts_tensor.to(
                        scores.dtype
                    ).clamp_min(1.0)

                # Drop rows with no positives or invalid intermediate values.
                valid_mask = (
                    (positive_counts_tensor > 0)
                    & torch.isfinite(log_negative_sum)
                    & torch.isfinite(log_positive_neg_sum)
                )
                if valid_mask.any():
                    # Average only valid query rows.
                    loss = loss_per_query[valid_mask].mean()
                else:
                    # Preserve dtype and device when the batch has no valid rows.
                    loss = scores.new_tensor(0.0)

                # Match the scaling convention used by distributed contrastive training.
                if self.negatives_x_device:
                    loss = loss * self.world_size
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @staticmethod
    def build_pooler(model_args):
        return None

    @staticmethod
    def load_pooler(weights, **config):
        return None

    def encode_passage(self, psg):
        raise NotImplementedError("EncoderModel is an abstract class")

    def encode_query(self, qry):
        raise NotImplementedError("EncoderModel is an abstract class")

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        data_args: DataArguments,
        **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            # Load separate query and passage encoder weights when the checkpoint
            # was saved with untied encoders.
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(
                    model_args.model_name_or_path, "query_model"
                )
                _psg_model_path = os.path.join(
                    model_args.model_name_or_path, "passage_model"
                )
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f"loading query model weight from {_qry_model_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f"loading passage model weight from {_psg_model_path}")
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
            # Tied encoders share the same transformer instance for query and passage.
            else:
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    model_args.model_name_or_path, **hf_kwargs
                )
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name_or_path, **hf_kwargs
            )
            # For remote pretrained models, clone the encoder only when untied
            # query and passage towers are requested.
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            data_args=data_args,
            train_args=train_args,
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            negatives_x_device=train_args.negatives_x_device,
            untie_encoder=model_args.untie_encoder,
        )
        return model

    @classmethod
    def load(
        cls,
        data_args,
        train_args,
        model_args,
        model_name_or_path,
        **hf_kwargs,
    ):
        # load local
        untie_encoder = True
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, "query_model")
            _psg_model_path = os.path.join(model_name_or_path, "passage_model")
            if os.path.exists(_qry_model_path):
                logger.info(f"found separate weight for query/passage encoders")
                logger.info(f"loading query model weight from {_qry_model_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f"loading passage model weight from {_psg_model_path}")
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
                untie_encoder = False
            else:
                logger.info(f"try loading tied weight")
                logger.info(f"loading model weight from {model_name_or_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    model_name_or_path, **hf_kwargs
                )
                lm_p = lm_q
        else:
            logger.info(f"try loading tied weight")
            logger.info(f"loading model weight from {model_name_or_path}")
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, "pooler.pt")
        pooler_config = os.path.join(model_name_or_path, "pooler_config.json")
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f"found pooler weight and configuration")
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        model = cls(
            data_args=data_args,
            train_args=train_args,
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            untie_encoder=untie_encoder,
        )
        return model

    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, "query_model"))
            os.makedirs(os.path.join(output_dir, "passage_model"))
            self.lm_q.save_pretrained(os.path.join(output_dir, "query_model"))
            self.lm_p.save_pretrained(os.path.join(output_dir, "passage_model"))
        else:
            self.lm_q.save_pretrained(output_dir)
        if self.pooler:
            self.pooler.save_pooler(output_dir)

    # def compute_loss(self, scores, target):
    #     return self.cross_entropy(scores, target)

    # def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
    #     if t is None:
    #         return None
    #     t = t.contiguous()

    #     all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
    #     dist.all_gather(all_tensors, t)

    #     all_tensors[self.process_rank] = t
    #     all_tensors = torch.cat(all_tensors, dim=0)

    #     return all_tensors
