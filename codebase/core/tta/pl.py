import functools
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from .base import CloudEdgeTestTimeAdaptation
from .tent import prepare_model_for_tent_tta


def excute_pl_style_tta(
    model: nn.Module,
    samples: torch.Tensor,
    loss_fn: Callable,
    optimizer: optim.Optimizer,
):
    outputs: torch.Tensor = model(samples)

    with torch.no_grad():
        pseudo_labels = torch.argmax(outputs, dim=-1, keepdim=False)
    loss: torch.Tensor = loss_fn(outputs, pseudo_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return outputs, loss


class PL(CloudEdgeTestTimeAdaptation):
    def __init__(self, model: nn.Module, tta_lr: float) -> None:
        self.tta_lr = tta_lr
        super(PL, self).__init__(model)
        self.model.train()

        prepare_model_for_tent_tta(self.model)

        self.tta_optimizer = optim.SGD(
            params=[p for name, p in self.model.named_parameters() if p.requires_grad],
            lr=tta_lr,
            momentum=0.9
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def infer(self, x: torch.Tensor):
        outputs, loss = excute_pl_style_tta(
            model=self.model,
            samples=x,
            loss_fn=self.loss_fn,
            optimizer=self.tta_optimizer
        )
        return outputs

    def __str__(self) -> str:
        return f"PL with tta_lr {self.tta_lr}"
