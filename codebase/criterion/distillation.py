import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .tta import entropy

_logger = logging.getLogger(__name__)


class TTADistillationLoss(_Loss):
    def __init__(self, alpha, beta, gamma, temperature, is_reciprocal_temperature, mse_balanced, E_max,
                 size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)

        assert isinstance(is_reciprocal_temperature, bool)
        assert isinstance(mse_balanced, bool)
        if mse_balanced and is_reciprocal_temperature:
            raise ValueError("mse_balanced and is_reciprocal_temperature can not both be true")

        assert not is_reciprocal_temperature
        assert not mse_balanced

        self.alpha = alpha  # weight for entropy minim. loss
        self.beta = beta  # weight for distillation kl-loss
        self.gamma = gamma  # weight for distillation ce-loss

        self.temperature = temperature  # temperature in distillation

        self.E_max = E_max

        self.is_reciprocal_temperature = is_reciprocal_temperature
        self.mse_balanced = mse_balanced

        self.eps = 1e-4

        _logger.info(f"Build TTADistillationLoss with alpha={alpha}, beta={beta}, gamma={gamma}, "
                     f"temperature={temperature}, is_reciprocal_temperature={is_reciprocal_temperature}, "
                     f"mse_balanced={mse_balanced}")

        if reduction != "mean":
            raise ValueError(f"The parameter 'reduction' must be in ['mean'], bot got {self.reduction}")

    def forward(self, source_logits: torch.Tensor, target_logits: torch.Tensor):
        batch_size, *_ = source_logits.shape

        # ent: torch.Tensor = entropy(e_logits, reduction="none")
            #     # coeff = 1 / (torch.exp(ent.clone().detach() - self.ent_high_margin))
            #     # loss = ent.mul(coeff).mean()

        ent: torch.Tensor = entropy(source_logits, reduction="none")
        if self.E_max is not None:
            coeff = 1 / (torch.exp(ent.clone().detach() - self.E_max))
        else:
            coeff = 1
        ent_loss = ent.mul(coeff).mean()

        # if self.is_reciprocal_temperature:
        #     kl_loss: torch.Tensor = F.kl_div(
        #         F.log_softmax(source_logits/self.temperature, dim=-1),
        #         F.softmax(target_logits/(1/self.temperature), dim=-1),
        #         reduction=self.reduction,
        #         log_target=False
        #     )
        # elif self.mse_balanced:
        #     kl_loss: torch.Tensor = F.kl_div(
        #         F.log_softmax(source_logits/self.temperature, dim=-1),
        #         F.softmax(target_logits/self.temperature, dim=-1),
        #         reduction="none",
        #         log_target=False
        #     ) * (self.temperature**2)
        #     mse_distance = F.mse_loss(F.softmax(source_logits, dim=-1), F.softmax(target_logits, dim=-1), reduction="none").mean(-1).detach()
        #     weights = 1/(mse_distance+self.eps)
        #     # _logger.debug(f"weights={weights}")
        #     kl_loss = torch.mean(kl_loss.mean(-1) * weights)
        # else:
        if self.E_max is not None:
            ce_ent: torch.Tensor = entropy(target_logits, reduction="none")
            ce_coeff = 1 / (torch.exp(ce_ent.clone().detach() - self.E_max))
        else:
            ce_coeff = 1

        kl_loss_per_sample: torch.Tensor = F.kl_div(
            F.log_softmax(source_logits/self.temperature, dim=-1),
            F.softmax(target_logits/self.temperature, dim=-1),
            reduction="none",
            log_target=False
        ).sum(-1) * (self.temperature**2)
        kl_loss = kl_loss_per_sample.mul(ce_coeff).mean()

        with torch.no_grad():
            pseudo_labels = torch.argmax(target_logits, dim=-1, keepdim=False)
        ce_loss_per_sample: torch.Tensor = F.cross_entropy(source_logits, pseudo_labels, reduction="none")
        ce_loss = ce_loss_per_sample.mul(ce_coeff).mean()

        loss: torch.Tensor = self.alpha*ent_loss+self.beta*kl_loss+self.gamma*ce_loss

        _logger.debug(f"In TTADistillationLoss, batch_size={batch_size}, ent_loss={ent_loss.item():.8f}, kl_loss={kl_loss.item():.8f}, "
                      f"ce_loss={ce_loss.item():.8f}, loss={loss.item():.8f}")

        return loss
