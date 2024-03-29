import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CloudEdgeTestTimeAdaptation
from ..bn_utils import bn_mean_var, replace_bn_forward_with


def bn_adapt_bn_forward_impl(self: nn.BatchNorm2d, x: torch.Tensor, momentum: float):
    mean, var = bn_mean_var(x)

    batch_running_mean = (1 - momentum) * self.running_mean + momentum * mean
    batch_running_var = (1 - momentum) * self.running_var + momentum * var

    return F.batch_norm(
        x,
        batch_running_mean,
        batch_running_var,
        self.weight,
        self.bias,
        training=False,
        momentum=self.momentum,
        eps=self.eps
    )


class BNAdaptation(CloudEdgeTestTimeAdaptation):
    def __init__(self, model: nn.Module, momentum: float) -> None:
        self.momentum = momentum

        super(BNAdaptation, self).__init__(model)
        self.model.eval()

        replace_bn_forward_with(
            model=self.model,
            fn=functools.partial(
                bn_adapt_bn_forward_impl,
                momentum=momentum
            )
        )

    def __str__(self) -> str:
        return f"BNAdaptation with momentum={self.momentum}"
