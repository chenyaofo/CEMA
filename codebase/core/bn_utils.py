import functools
import logging
import types
from statistics import mode
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import torch
from torch import nn

def bn_mean_var(x: torch.Tensor):
    var, mean = torch.var_mean(x, dim=[0, 2, 3], keepdim=False, unbiased=True)
    return mean, var


def replace_bn_forward_with(model: nn.Module, fn):
    for name, bn in get_bn_from_model(model):
        bn: nn.BatchNorm2d
        bn.forward = types.MethodType(fn, bn)

BN_MODULE_TYPES: Tuple[Type[nn.Module]] = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)

_logger: logging.Logger = logging.getLogger(__name__)


def get_bn_from_model(model: nn.Module):
    return [
        (name, bn) for name, bn in model.named_modules() if isinstance(bn, BN_MODULE_TYPES)
    ]


def set_tta_bn_pre_hook(model: nn.Module, momentum: float, ema_mode: bool = False):
    def tta_bn_pre_hook(
        module: nn.BatchNorm2d, input: Tuple[torch.Tensor], momentum: float,
        train_running_mean: torch.Tensor, train_running_var: torch.Tensor,
        ema_mode: bool
    ) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            x = input[0]

            assert isinstance(
                x, torch.Tensor
            ), f"BN layer should take tensor as input. Got {input}"

            mean = x.mean(dim=[0, 2, 3], keepdim=False)
            var = x.var(dim=[0, 2, 3], keepdim=False, unbiased=False)

            if ema_mode:
                module.running_mean.copy_((1 - momentum) * module.running_mean + momentum * mean)
                module.running_var.copy_((1 - momentum) * module.running_var + momentum * var)
            else:
                module.running_mean.copy_((1 - momentum) * train_running_mean + momentum * mean)
                module.running_var.copy_((1 - momentum) * train_running_var + momentum * var)

        return (x,)

    bn_layers = get_bn_from_model(model)

    hooks_to_remove = []

    for name, bn in bn_layers:
        bn: nn.BatchNorm2d
        with torch.no_grad():
            train_running_mean = bn.running_mean.clone()
            train_running_var = bn.running_var.clone()

        hook = bn.register_forward_pre_hook(
            functools.partial(
                tta_bn_pre_hook,
                momentum=momentum,
                train_running_mean=train_running_mean,
                train_running_var=train_running_var,
                ema_mode=ema_mode
            )
        )

        hooks_to_remove.append(hook)

        _logger.info(f"Set TTA BatchNorm Hook for the layer {name} with momentum={momentum:.4f}")

    return hooks_to_remove


def model_diff(original:nn.Module, current:nn.Module) -> torch.Tensor:
    with torch.no_grad():
        original_bns = get_bn_from_model(original)
        current_bns = get_bn_from_model(current)

        diffs = 0

        numel = 0

        for (name1, bn1), (name2, bn2) in zip(original_bns, current_bns):
            bn1:nn.BatchNorm2d
            bn2:nn.BatchNorm2d

            diffs += torch.abs(bn1.weight-bn2.weight).sum()
            numel += bn1.weight.numel()

            diffs += torch.abs(bn1.bias-bn2.bias).sum()
            numel += bn1.bias.numel()

        return diffs / numel