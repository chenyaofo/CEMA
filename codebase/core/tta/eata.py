import copy
import functools
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import CloudEdgeTestTimeAdaptation
from codebase.criterion.tta import entropy


def prepare_model_for_tent_tta(model: nn.Module, update_all_parameters: bool = False):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # m.eval()
            m.requires_grad_(True)
            # # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.eval()
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    if update_all_parameters:
        for name, p in model.named_parameters():
            p.requires_grad = True
    return model


class EntropyCriterion:
    def __init__(self, ent_high_margin: float):
        self.ent_high_margin = ent_high_margin

        self.n_cnt_total = 0
        self.n_cnt_remain = 0
        self.n_cnt_filter = 0

        self.n_filter_samples = 0
        self.n_remain_samples = 0
        self.n_total_samples = 0

    def filter_out(self, logits: torch.Tensor):
        with torch.no_grad():
            batch_size, *_ = logits.shape
            ents = entropy(logits, reduction="none")
            remain_ids, *_ = torch.where(ents < self.ent_high_margin)

            self.n_cnt_total = batch_size
            self.n_cnt_remain = remain_ids.numel()
            self.n_cnt_filter = self.n_cnt_total - self.n_cnt_remain

            self.n_total_samples += self.n_cnt_total
            self.n_remain_samples += self.n_cnt_remain
            self.n_filter_samples += self.n_cnt_filter

        return remain_ids


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


class SimilarityCriterion:
    def __init__(self, history_model_probs: torch.Tensor = None, d_margin: float = None):
        self.history_model_probs = history_model_probs
        self.d_margin = d_margin

        self.n_cnt_total = 0
        self.n_cnt_remain = 0
        self.n_cnt_filter = 0

        self.n_filter_samples = 0
        self.n_remain_samples = 0
        self.n_total_samples = 0

    def filter_out(self, logits: torch.Tensor):
        with torch.no_grad():
            batch_size, *_ = logits.shape

            if self.history_model_probs is not None:
                cosine_similarities = F.cosine_similarity(self.history_model_probs.unsqueeze(dim=0), logits.softmax(1), dim=1)
                remain_ids, *_ = torch.where(torch.abs(cosine_similarities) < self.d_margin)
                self.history_model_probs = update_model_probs(self.history_model_probs, logits[remain_ids].softmax(1))
            else:
                self.history_model_probs = update_model_probs(self.history_model_probs, logits.softmax(1))
                remain_ids = torch.arange(batch_size, device=logits.device)

            if remain_ids is not None:
                self.n_cnt_total = batch_size
                self.n_cnt_remain = remain_ids.numel()
                self.n_cnt_filter = self.n_cnt_total - self.n_cnt_remain

                self.n_total_samples += self.n_cnt_total
                self.n_remain_samples += self.n_cnt_remain
                self.n_filter_samples += self.n_cnt_filter

            return remain_ids


class EATA(CloudEdgeTestTimeAdaptation):
    def __init__(
        self,
        model: nn.Module,
        ent_high_margin: float,
        d_margin: float,
        tta_lr: float,
        is_episodic: bool = False,
        update_all_parameters: bool = False,
    ) -> None:

        self.IS_FILTERED = True

        self.tta_lr = tta_lr

        assert isinstance(is_episodic, bool)
        self.is_episodic = is_episodic

        super(EATA, self).__init__(model)
        self.model.train()

        prepare_model_for_tent_tta(self.model, update_all_parameters=update_all_parameters)

        self.tta_optimizer = optim.SGD(
            params=[p for name, p in self.model.named_parameters() if p.requires_grad],
            lr=tta_lr,
            momentum=0.9
        )

        self.loss_fn = functools.partial(entropy, reduction="mean")

        self.ent_high_margin = ent_high_margin
        self.entropy_criterion = EntropyCriterion(ent_high_margin=self.ent_high_margin)

        self.d_margin = d_margin
        self.similarity_criterion = SimilarityCriterion(d_margin=self.d_margin)

        if self.is_episodic:
            self.ori_model_state = copy.deepcopy(self.model.state_dict())
            self.ori_optimizer_state = copy.deepcopy(self.tta_optimizer.state_dict())

        self.labels_collection = []

    def reset(self):
        if self.is_episodic:
            self.model.load_state_dict(self.ori_model_state)
            self.tta_optimizer.load_state_dict(self.ori_optimizer_state)
        else:
            raise ValueError("Function reset should not be called when is_episodic=False.")

    def infer(self, x: torch.Tensor, y: torch.Tensor = None):
        with torch.no_grad():
            outputs: torch.Tensor = self.model(x)

        ent_remain_ids = self.entropy_criterion.filter_out(outputs)
        filtered_outputs = outputs[ent_remain_ids]

        sim_remain_ids = self.similarity_criterion.filter_out(filtered_outputs)
        filtered_outputs = outputs[sim_remain_ids]

        if y is not None:
            self.labels_collection.append(y[ent_remain_ids][sim_remain_ids].cpu())

        if self.similarity_criterion.n_cnt_remain > 0:
            filtered_samples = x[ent_remain_ids][sim_remain_ids]
            filtered_outputs = self.model(filtered_samples)
            ent: torch.Tensor = entropy(filtered_outputs, reduction="none")
            coeff = 1 / (torch.exp(ent.clone().detach() - self.ent_high_margin))
            loss = ent.mul(coeff).mean()

            self.tta_optimizer.zero_grad()
            loss.backward()
            self.tta_optimizer.step()
        return outputs

    def __str__(self) -> str:
        return f"Tent with tta_lr {self.tta_lr}"

    def extra_log(self):
        n_total_samples = self.entropy_criterion.n_total_samples
        n_filter_samples = self.entropy_criterion.n_filter_samples + self.similarity_criterion.n_filter_samples
        n_remain_samples = n_total_samples -  n_filter_samples
        return f"entropy_criterion(n_remain_samples={self.entropy_criterion.n_remain_samples:06d}, " + \
            f"n_filter_samples={self.entropy_criterion.n_filter_samples:06d}, " + \
            f"n_total_samples={self.entropy_criterion.n_total_samples:06d}), " + \
            f"similarity_criterion(n_remain_samples={self.similarity_criterion.n_remain_samples:06d}, " + \
            f"n_filter_samples={self.similarity_criterion.n_filter_samples:06d}, " + \
            f"n_total_samples={self.similarity_criterion.n_total_samples:06d}), " + \
            f"total(n_remain_samples={n_remain_samples:06d}, " + \
            f"n_filter_samples={n_filter_samples:06d}, " + \
            f"n_total_samples={n_total_samples:06d}), "
