import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .register import CRITERION

_logger = logging.getLogger(__name__)


def entropy(
    logits: torch.Tensor,
    reduction="none",
):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    losses = -p_log_p.sum(-1)
    if reduction == "none":
        return losses
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "mean":
        return losses.mean()
    else:
        raise ValueError(f"The parameter 'reduction' must be in ['none','mean','sum'], bot got {reduction}")


def maxp(
    logits: torch.Tensor,
    reduction="none",
):
    assert reduction == "none"

    return logits.max(dim=-1)


def maxp_minus_secdp(
    logits: torch.Tensor,
    reduction="none",
):
    assert reduction == "none"

    topk_p, _ = torch.topk(logits, k=2, dim=-1)
    topk_p: torch.Tensor
    maxp, secdp = topk_p.unbind(dim=-1)

    return maxp - secdp


def maxp_minus_minp(
    logits: torch.Tensor,
    reduction="none",
):
    assert reduction == "none"

    maxp = logits.max(dim=-1)
    minp = logits.min(dim=-1)

    return maxp - minp
