import torch
import torch.nn as nn
import torch.nn.functional as F

from .tent import Tent
from ..bn_utils import replace_bn_forward_with


def tbr_bn_forward_impl(self: nn.BatchNorm2d, x: torch.Tensor):
    
    batch_var, batch_mean = torch.var_mean(x, dim=(0, 2, 3), keepdim=True)
    batch_std = torch.sqrt(batch_var+self.eps)

    if self.running_mean is None:
        self.running_mean, self.running_var = batch_mean.clone().detach(), batch_var.clone().detach()

    self.running_mean, self.running_var = self.running_mean.view(1, -1, 1, 1), self.running_var.view(1, -1, 1, 1)

    r = batch_std.detach() / torch.sqrt(self.running_var+self.eps)
    d = (batch_mean.detach() - self.running_mean) / torch.sqrt(self.running_var+self.eps)
    x = ((x - batch_mean) / batch_std) * r + d
    
    if self.training:
        self.running_mean += self.momentum * (batch_mean.detach() - self.running_mean)
        self.running_var += self.momentum * (batch_var.detach() - self.running_var)
    else:
        pass

    x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)

    return x
    # else:
    #     return F.batch_norm(
    #         x,
    #         self.running_mean,
    #         self.running_var,
    #         self.weight,
    #         self.bias,
    #         training=False,
    #         momentum=self.momentum,
    #         eps=self.eps
    #     )


class TBR(Tent):
    def __init__(self, model: nn.Module, tta_lr: float) -> None:
        super(TBR, self).__init__(model, tta_lr)
        replace_bn_forward_with(
            model=self.model,
            fn=tbr_bn_forward_impl
        )

    def __str__(self) -> str:
        return f"TBR with tta_lr {self.tta_lr}"
