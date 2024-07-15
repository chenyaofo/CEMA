import math
import random
import copy
import functools
import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from codebase.models import MODEL
from codebase.criterion.distillation import TTADistillationLoss

from .cotta import get_tta_transforms
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


def prepare_model_for_tent_tta(model: nn.Module, is_tbr=False, update_all_parameters=False):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # m.eval()
            m.requires_grad_(True)
            # # force use of batch stats in train and eval modes
            # if not is_tbr:
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.eval()
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    if update_all_parameters:
        for p in model.parameters():
            p.requires_grad_(True)
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

        _logger.info(f"Init EntropyCriterion with ent_high_margin={ent_high_margin}, " +
                     f"ent_low_margin={ent_low_margin}, warm_up_batch_size={warm_up_batch_size}, ent_high_margin_coeff={ent_high_margin_coeff}")

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

            if self.d_margin is None:
                remain_ids = torch.arange(batch_size, device=logits.device)
            else:
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


class CEMA:
    def __init__(
        self,
        model: nn.Module,
        ent_low_margin: float,
        ent_high_margin: float,
        d_margin: float,
        warm_up_batch_size: int,
        ent_high_margin_coeff: float,
        teacher_model_name: str,
        batch_size: int,
        alpha: float,
        beta: float,
        gamma: float,
        temperature: float,
        replay_buffer_size: int,
        s_tta_lr: float,
        t_tta_lr: float,
        n_class: int,
        is_tbr: bool = False,
        is_teacher_outputs: bool = False,
        update_teacher_all_parameters: bool = False,
        update_student_all_parameters: bool = False,
        K: int = 1,
    ) -> None:
        assert isinstance(is_tbr, bool)
        self.is_tbr = is_tbr

        assert isinstance(is_teacher_outputs, bool)
        self.is_teacher_outputs = is_teacher_outputs

        self.replay_buffer_size = replay_buffer_size

        is_reciprocal_temperature = False
        mse_balanced = False
        _logger.info(f"CETTA with teacher={teacher_model_name}, temperature={temperature}, "
                     f"alpha={alpha}, beta={beta}, gamma={gamma}, s_tta_lr={s_tta_lr}, t_tta_lr={t_tta_lr}, "
                     f"is_reciprocal_temperature={is_reciprocal_temperature}, mse_balanced={mse_balanced}")

        self.edge_model = model
        self.foundation_model: nn.Module = \
            MODEL.build_from(dict(type_=teacher_model_name, pretrained=True))

        self.foundation_model.to(device=list(model.parameters())[0].device)

        if self.is_tbr:
            replace_bn_forward_with(
                model=self.edge_model,
                fn=tbr_bn_forward_impl
            )

            replace_bn_forward_with(
                model=self.foundation_model,
                fn=tbr_bn_forward_impl,
            )

        prepare_model_for_tent_tta(self.edge_model, is_tbr=self.is_tbr, 
                                   update_all_parameters=update_student_all_parameters)
        self.tta_e_optimizer = optim.SGD(
            params=[p for name, p in self.edge_model.named_parameters() if p.requires_grad],
            lr=s_tta_lr,
            momentum=0.9
        )

        prepare_model_for_tent_tta(self.foundation_model, is_tbr=self.is_tbr,
                                   update_all_parameters=update_teacher_all_parameters)
        self.tta_f_optimizer = optim.SGD(
            params=[p for name, p in self.foundation_model.named_parameters() if p.requires_grad],
            lr=t_tta_lr,
            momentum=0.9
        )

        # self.ent_high_margin = ent_high_margin * math.log(n_class)
        if ent_high_margin is None:
            self.ent_high_margin = None
        else:
            self.ent_high_margin = ent_high_margin * math.log(n_class)
        if ent_low_margin is None:
            self.ent_low_margin = None
        else:
            self.ent_low_margin = ent_low_margin * math.log(n_class)
        self.entropy_criterion = EntropyCriterion(
            ent_low_margin=self.ent_low_margin,
            ent_high_margin=self.ent_high_margin,
            warm_up_batch_size=warm_up_batch_size,
            ent_high_margin_coeff=ent_high_margin_coeff
        )

        self.f_entropy_criterion = EntropyCriterion(
            ent_low_margin=None,
            ent_high_margin=self.ent_high_margin,
            warm_up_batch_size=None,
            ent_high_margin_coeff=1
        )

        self.d_margin = d_margin
        self.similarity_criterion = SimilarityCriterion(d_margin=self.d_margin)

        self.distill_loss = TTADistillationLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            temperature=temperature,
            is_reciprocal_temperature=is_reciprocal_temperature,
            mse_balanced=mse_balanced,
            E_max=self.ent_high_margin
        )

        self.cloud_buffer = []
        self.history_buffer = []
        self.batch_size = batch_size

        self.transforms = get_tta_transforms()

        # self.edge_model_shadow = copy.deepcopy(self.edge_model)

        # replace_bn_forward_with(
        #     model=self.edge_model_shadow,
        #     fn=tbr_bn_forward_impl
        # )

        # prepare_model_for_tent_tta(self.edge_model_shadow, is_tbr=self.is_tbr, 
        #                            update_all_parameters=update_student_all_parameters)
        self.iters = 0

        self.K = K

    def infer(self, x: torch.Tensor):
        self.iters += 1
        # excuted in edge
        with torch.no_grad():
            _logger.info(f"iters={self.iters}")
            if self.iters < 10:
                set_bn_training_mode(self.edge_model, mode=False)
                edge_outputs: torch.Tensor = self.edge_model(x)
            else:
                if self.iters % self.K == 0 or self.iters == 10:
                    self.edge_model_shadow = copy.deepcopy(self.edge_model)
                    _logger.info(f"update edge model")

                set_bn_training_mode(self.edge_model_shadow, mode=False)
                edge_outputs: torch.Tensor = self.edge_model_shadow(x)
            ###
            ent_remain_ids = self.entropy_criterion.filter_out(edge_outputs)
            filtered_samples_outputs = edge_outputs[ent_remain_ids]

            sim_remain_ids = self.similarity_criterion.filter_out(filtered_samples_outputs)
            filtered_samples = x[ent_remain_ids][sim_remain_ids]
            ###

            ###
            # batch_size = edge_outputs.shape[0]
            # remain_batch_size = int(batch_size * 9889 / 50000)
            # remain_ids = torch.randperm(batch_size, device=edge_outputs.device)[:remain_batch_size]
            # filtered_samples = x[remain_ids]
            ###

            self.cloud_buffer += filtered_samples.unbind(dim=0)
            if self.replay_buffer_size != 0:
                b_samples = filtered_samples.unbind(dim=0)
                b_samples = [s.cpu() for s in b_samples]
                self.history_buffer += b_samples

        # excuted in cloud
        # uploaded_samples = filtered_samples
        if len(self.history_buffer) > self.replay_buffer_size:
            self.history_buffer = self.history_buffer[-self.replay_buffer_size:]

        is_excuted = False
        while len(self.cloud_buffer) >= self.batch_size:
            is_excuted=True
            uploaded_samples = self.cloud_buffer[:self.batch_size]
            uploaded_samples = torch.stack(uploaded_samples, dim=0)
            self.cloud_buffer = self.cloud_buffer[self.batch_size:]

            # step 1: Adaptation of the foundation model via entropy minimization
            # uploaded_samples = filtered_samples
            set_bn_training_mode(self.foundation_model, mode=True)
            uploaded_samples_outputs: torch.Tensor = self.foundation_model(uploaded_samples)
            # f_ent_remain_ids = self.f_entropy_criterion.filter_out(uploaded_samples_outputs)
            ent: torch.Tensor = entropy(uploaded_samples_outputs, reduction="none")
            if self.ent_high_margin is not None:
                coeff = 1 / (torch.exp(ent.clone().detach() - self.ent_high_margin))
            else:
                coeff = 1
            loss = ent.mul(coeff).mean()
            self.tta_f_optimizer.zero_grad()
            loss.backward()
            self.tta_f_optimizer.step()

            # step 2: Adaptation of the edge model via knowledge distillation
            set_bn_training_mode(self.foundation_model, mode=False)
            set_bn_training_mode(self.edge_model, mode=True)

            # for i in range(1):
            #     with torch.no_grad():
            #         f_logits: torch.Tensor = self.foundation_model(uploaded_samples)
            #     e_logits: torch.Tensor = self.edge_model(uploaded_samples)
            #     distill_loss: torch.Tensor = self.distill_loss(e_logits, f_logits.detach())
            #     # distill_loss: torch.Tensor = self.distill_loss(e_logits, uploaded_samples_outputs.detach())
            #     # ent: torch.Tensor = entropy(e_logits, reduction="none")
            #     # coeff = 1 / (torch.exp(ent.clone().detach() - self.ent_high_margin))
            #     # loss = ent.mul(coeff).mean()
            #     self.tta_e_optimizer.zero_grad()
            #     distill_loss.backward()
            #     # (0.1*loss+distill_loss).backward()
            #     self.tta_e_optimizer.step()
            if len(self.history_buffer) < 256 or self.replay_buffer_size == 0:
                for i in range(1):
                    with torch.no_grad():
                        f_logits: torch.Tensor = self.foundation_model(uploaded_samples)
                    e_logits: torch.Tensor = self.edge_model(uploaded_samples)
                    distill_loss: torch.Tensor = self.distill_loss(e_logits, f_logits.detach())
                    # distill_loss: torch.Tensor = self.distill_loss(e_logits, uploaded_samples_outputs.detach())
                    # ent: torch.Tensor = entropy(e_logits, reduction="none")
                    # coeff = 1 / (torch.exp(ent.clone().detach() - self.ent_high_margin))
                    # loss = ent.mul(coeff).mean()
                    self.tta_e_optimizer.zero_grad()
                    distill_loss.backward()
                    # (0.1*loss+distill_loss).backward()
                    self.tta_e_optimizer.step()
            else:
                for i in range(1):
                    # set_bn_training_mode(self.edge_model, mode=False)
                    random_history_samples = random.sample(self.history_buffer, k=self.batch_size*3)
                    random_history_samples = torch.stack(random_history_samples, dim=0).cuda()
                    concat_samples = torch.cat([uploaded_samples, random_history_samples], dim=0)
                    with torch.no_grad():
                        f_logits: torch.Tensor = self.foundation_model(concat_samples)
                    e_logits: torch.Tensor = self.edge_model(concat_samples)
                    distill_loss: torch.Tensor = self.distill_loss(e_logits, f_logits.detach())
                    # distill_loss: torch.Tensor = self.distill_loss(e_logits, uploaded_samples_outputs.detach())
                    # ent: torch.Tensor = entropy(e_logits, reduction="none")
                    # coeff = 1 / (torch.exp(ent.clone().detach() - self.ent_high_margin))
                    # loss = ent.mul(coeff).mean()
                    self.tta_e_optimizer.zero_grad()
                    distill_loss.backward()
                    # (0.1*loss+distill_loss).backward()
                    self.tta_e_optimizer.step()

            # step 3: distribute the updated parameters of the edge model
            # since we simulate with the shared edge model, pass

        if not is_excuted and len(self.history_buffer) > 256 and self.replay_buffer_size != 0:
            random_history_samples = random.sample(self.history_buffer, k=self.batch_size*4)
            random_history_samples = torch.stack(random_history_samples, dim=0).cuda()
            with torch.no_grad():
                f_logits: torch.Tensor = self.foundation_model(random_history_samples)
            e_logits: torch.Tensor = self.edge_model(random_history_samples)
            distill_loss: torch.Tensor = self.distill_loss(e_logits, f_logits.detach())
            self.tta_e_optimizer.zero_grad()
            distill_loss.backward()
            self.tta_e_optimizer.step()

        if self.is_teacher_outputs:
            set_bn_training_mode(self.foundation_model, mode=False)
            with torch.no_grad():
                teacher_outputs = self.foundation_model(x)
            return teacher_outputs
        else:
            return edge_outputs

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
            f"cnt_avg_ent_value={self.entropy_criterion.avg_ent.value}, " + \
            f"cloud entropy_criterion(n_filter_samples={self.f_entropy_criterion.n_filter_samples})"
