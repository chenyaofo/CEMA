import copy
import functools
from typing import Callable
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import CloudEdgeTestTimeAdaptation
from .tbr import replace_bn_forward_with, tbr_bn_forward_impl
from codebase.criterion.tta import entropy

_logger = logging.getLogger(__name__)

class AverageMetric(object):
    def __init__(self, warm_up_n: int):
        self.n = 0
        self._value = 0.

        self._init = False
        self._init_value = 0.

        self.warm_up_n = warm_up_n

    def reset(self) -> None:
        self.n = 0
        self._value = 0.

    def update(self, value) -> None:
        if torch.is_tensor(value):
            self.n += value.numel()
            self._value += value.sum().item()
        elif isinstance(value, (int, float)):
            self.n += 1
            self._value += value
        else:
            raise ValueError("The parameter 'value' should be int, float or pytorch scalar tensor, but found {}"
                             .format(type(value)))

        if self.n >= self.warm_up_n and not self._init:
            self._init = True
            self._init_value = self.value

    @property
    def is_warm_up(self):
        return self._init

    @property
    def warm_up_value(self):
        return self._init_value

    @property
    def value(self) -> float:
        if self.n == 0:
            return 0
        return self._value / self.n


def prepare_model_for_tent_tta(model: nn.Module, is_tbr=False):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # m.eval()
            m.requires_grad_(True)
            # # force use of batch stats in train and eval modes
            if not is_tbr:
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.eval()
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model


def set_bn_training_mode(model: nn.Module, mode: bool):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train(mode=mode)
    return model


class EntropyCriterion:
    def __init__(self, ent_high_margin: float, ent_low_margin: float,
                 warm_up_batch_size: int, ent_high_margin_coeff: float):
        self.ent_high_margin = ent_high_margin
        self.ent_low_margin = ent_low_margin

        if warm_up_batch_size is None:
            self.warm_up_batch_size = 10**12
        else:
            self.warm_up_batch_size = warm_up_batch_size
        self.ent_high_margin_coeff = ent_high_margin_coeff

        self.n_cnt_total = 0
        self.n_cnt_remain = 0
        self.n_cnt_filter = 0

        self.n_filter_samples = 0
        self.n_remain_samples = 0
        self.n_total_samples = 0

        self.avg_ent = AverageMetric(self.warm_up_batch_size)

    @property
    def cnt_ent_high_margin(self):
        if self.avg_ent.is_warm_up:
            return self.ent_high_margin * (self.avg_ent.value / self.avg_ent.warm_up_value) * self.ent_high_margin_coeff
        else:
            return self.ent_high_margin

    def filter_out(self, logits: torch.Tensor):
        with torch.no_grad():
            batch_size, *_ = logits.shape
            ents = entropy(logits, reduction="none")
            self.avg_ent.update(ents)
            if self.ent_high_margin is not None:
                remain_ids, *_ = torch.where(ents < self.cnt_ent_high_margin)

            if self.ent_low_margin is not None:
                remain_ent = ents[remain_ids]
                sec_remain_ids, *_ = torch.where(remain_ent > self.ent_low_margin)
                remain_ids = remain_ids[sec_remain_ids]

            self.n_cnt_total = batch_size
            if self.ent_low_margin is None and self.ent_high_margin is None:
                self.n_cnt_remain = batch_size
                self.n_cnt_filter = 0
            else:
                self.n_cnt_remain = remain_ids.numel()
                self.n_cnt_filter = self.n_cnt_total - self.n_cnt_remain

            self.n_total_samples += self.n_cnt_total
            self.n_remain_samples += self.n_cnt_remain
            self.n_filter_samples += self.n_cnt_filter

        if self.ent_low_margin is None and self.ent_high_margin is None:
            remain_ids = torch.arange(batch_size, device=logits.device)

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


class EATAPlus(CloudEdgeTestTimeAdaptation):
    def __init__(
        self,
        model: nn.Module,
        ent_low_margin: float,
        ent_high_margin: float,
        d_margin: float,
        warm_up_batch_size: int,
        ent_high_margin_coeff: float,
        tta_lr: float,
        is_episodic: bool = False,
        is_tbr: bool = False
    ) -> None:

        self.IS_FILTERED = True

        self.tta_lr = tta_lr

        assert isinstance(is_episodic, bool)
        self.is_episodic = is_episodic

        assert isinstance(is_tbr, bool)
        self.is_tbr = is_tbr

        super(EATAPlus, self).__init__(model)
        self.model.train()

        prepare_model_for_tent_tta(self.model, is_tbr=self.is_tbr)

        if self.is_tbr:
            replace_bn_forward_with(
                model=self.model,
                fn=tbr_bn_forward_impl
            )

        self.tta_optimizer = optim.SGD(
            params=[p for name, p in self.model.named_parameters() if p.requires_grad],
            lr=tta_lr,
            momentum=0.9
        )

        self.loss_fn = functools.partial(entropy, reduction="mean")

        self.ent_high_margin = ent_high_margin
        self.ent_low_margin = ent_low_margin
        self.entropy_criterion = EntropyCriterion(
            ent_low_margin=self.ent_low_margin,
            ent_high_margin=self.ent_high_margin,
            warm_up_batch_size=warm_up_batch_size,
            ent_high_margin_coeff=ent_high_margin_coeff
        )

        self.d_margin = d_margin
        self.similarity_criterion = SimilarityCriterion(d_margin=self.d_margin)

        if self.is_episodic:
            self.ori_model_state = copy.deepcopy(self.model.state_dict())
            self.ori_optimizer_state = copy.deepcopy(self.tta_optimizer.state_dict())

    def reset(self):
        if self.is_episodic:
            self.model.load_state_dict(self.ori_model_state)
            self.tta_optimizer.load_state_dict(self.ori_optimizer_state)
        else:
            raise ValueError("Function reset should not be called when is_episodic=False.")

    def infer(self, x: torch.Tensor):
        with torch.no_grad():
            set_bn_training_mode(self.model, mode=False)
            outputs: torch.Tensor = self.model(x)

        ent_remain_ids = self.entropy_criterion.filter_out(outputs)
        filtered_outputs = outputs[ent_remain_ids]

        sim_remain_ids = self.similarity_criterion.filter_out(filtered_outputs)
        filtered_outputs = outputs[sim_remain_ids]

        if self.similarity_criterion.n_cnt_remain > 0:
            filtered_samples = x[ent_remain_ids][sim_remain_ids]
            batch_size, *_=filtered_samples.shape
            _logger.info(f"batch_size={batch_size}")
            set_bn_training_mode(self.model, mode=True)
            filtered_outputs = self.model(filtered_samples)
            ent: torch.Tensor = entropy(filtered_outputs, reduction="none")
            if self.ent_high_margin is not None:
                coeff = 1 / (torch.exp(ent.clone().detach() - self.ent_high_margin))
            else:
                coeff = 1
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
        n_remain_samples = n_total_samples - n_filter_samples
        return f"entropy_criterion(n_remain_samples={self.entropy_criterion.n_remain_samples:06d}, " + \
            f"n_filter_samples={self.entropy_criterion.n_filter_samples:06d}, " + \
            f"n_total_samples={self.entropy_criterion.n_total_samples:06d}), " + \
            f"similarity_criterion(n_remain_samples={self.similarity_criterion.n_remain_samples:06d}, " + \
            f"n_filter_samples={self.similarity_criterion.n_filter_samples:06d}, " + \
            f"n_total_samples={self.similarity_criterion.n_total_samples:06d}), " + \
            f"total(n_remain_samples={n_remain_samples:06d}, " + \
            f"n_filter_samples={n_filter_samples:06d}, " + \
            f"n_total_samples={n_total_samples:06d}), " + \
            f"cnt_ent_high_margin={self.entropy_criterion.cnt_ent_high_margin}, " + \
            f"warm_up={self.entropy_criterion.avg_ent.is_warm_up}, " + \
            f"warm_up_value={self.entropy_criterion.avg_ent.warm_up_value}, " + \
            f"cnt_avg_ent_value={self.entropy_criterion.avg_ent.value}, "
