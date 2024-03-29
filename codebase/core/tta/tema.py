import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CloudEdgeTestTimeAdaptation
from ..bn_utils import bn_mean_var, replace_bn_forward_with


def tema_bn_forward_impl(self: nn.BatchNorm2d, x: torch.Tensor, momentum: float):
    mean, var = bn_mean_var(x)

    self.running_mean.copy_((1 - momentum) * self.running_mean + momentum * mean)
    self.running_var.copy_((1 - momentum) * self.running_var + momentum * var)

    return F.batch_norm(
        x,
        self.running_mean,
        self.running_var,
        self.weight,
        self.bias,
        training=False,
        momentum=self.momentum,
        eps=self.eps
    )


class TEMA(CloudEdgeTestTimeAdaptation):
    def __init__(self, model: nn.Module, momentum: float) -> None:
        self.momentum = momentum
        super(TEMA, self).__init__(model)
        self.model.eval()

        replace_bn_forward_with(
            model=self.model,
            fn=functools.partial(
                tema_bn_forward_impl,
                momentum=momentum
            )
        )

    def __str__(self) -> str:
        return f"TEMA with momentum={self.momentum}"
