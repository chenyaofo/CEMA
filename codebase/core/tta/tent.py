import copy
import functools
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from .base import CloudEdgeTestTimeAdaptation
from codebase.criterion.tta import entropy


def excute_tent_style_tta(
    model: nn.Module,
    samples: torch.Tensor,
    loss_fn: Callable,
    optimizer: optim.Optimizer,
):
    outputs: torch.Tensor = model(samples)

    loss: torch.Tensor = loss_fn(outputs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return outputs, loss


def prepare_model_for_tent_tta(model: nn.Module):
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
    return model


class Tent(CloudEdgeTestTimeAdaptation):
    def __init__(self, model: nn.Module, tta_lr: float, is_episodic: bool = False) -> None:
        self.tta_lr = tta_lr

        assert isinstance(is_episodic, bool)
        self.is_episodic = is_episodic

        super(Tent, self).__init__(model)
        self.model.train()

        prepare_model_for_tent_tta(self.model)

        self.tta_optimizer = optim.SGD(
            params=[p for name, p in self.model.named_parameters() if p.requires_grad],
            lr=tta_lr,
            momentum=0.9
        )

        self.loss_fn = functools.partial(entropy, reduction="mean")

        if self.is_episodic:
            self.ori_model_state = copy.deepcopy(self.model.state_dict())
            self.ori_optimizer_state = copy.deepcopy(self.tta_optimizer.state_dict())

    def reset(self):
        if self.is_episodic:
            self.model.load_state_dict(self.ori_model_state)
            self.tta_optimizer.load_state_dict(self.ori_optimizer_state)
        else:
            raise ValueError("Function reset should not be called when is_episodic=False.")

    def infer(self, x: torch.Tensor, mask_fn=None):
        outputs: torch.Tensor = self.model(x)

        loss: torch.Tensor = entropy(outputs, reduction="none")
        if mask_fn is None:
            loss = loss.mean()
            rev = outputs
        else:
            mask = mask_fn(outputs=outputs)
            # print(mask)
            loss = (loss*mask).mean()
            rev = (outputs, mask.sum().item(), mask.numel())

        self.tta_optimizer.zero_grad()
        loss.backward()
        self.tta_optimizer.step()
        # outputs, loss = excute_tent_style_tta(
        #     model=self.model,
        #     samples=x,
        #     loss_fn=self.loss_fn,
        #     optimizer=self.tta_optimizer
        # )
        return rev

    def __str__(self) -> str:
        return f"Tent with tta_lr {self.tta_lr}"
