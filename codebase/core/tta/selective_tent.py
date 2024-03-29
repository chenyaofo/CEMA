import logging
import functools


import torch
import torch.nn as nn
import torch.optim as optim

from .base import CloudEdgeTestTimeAdaptation
from codebase.criterion.tta import entropy

from .tent import prepare_model_for_tent_tta
from ..ste import autograd_ge, autograd_lt


_logger = logging.getLogger(__name__)


class Selection(nn.Module):
    def __init__(self) -> None:
        super(Selection, self).__init__()

        self.weight = nn.Parameter(-torch.ones(1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        return autograd_lt(x * self.weight + self.bias, 0.0)


class SelectiveTent(CloudEdgeTestTimeAdaptation):
    def __init__(self, model: nn.Module, tta_lr: float, S_lr: float) -> None:
        self.tta_lr = tta_lr
        super(SelectiveTent, self).__init__(model)
        self.model.train()

        self.S = Selection().to(device=list(model.parameters())[0].device)

        prepare_model_for_tent_tta(self.model)

        self.tta_optimizer = optim.SGD(
            params=[p for name, p in self.model.named_parameters() if p.requires_grad],
            lr=tta_lr,
            momentum=0.9
        )
        self.S_optimizer = optim.SGD(
            params=list(self.S.parameters()),
            lr=S_lr,
            momentum=0.9
        )

        self.loss_fn = functools.partial(entropy, reduction="mean")

        self.n_selective = 0
        self.n_total = 0

    def infer(self, x: torch.Tensor):
        outputs: torch.Tensor = self.model(x)

        ent: torch.Tensor = entropy(outputs, reduction="none")

        selection: torch.Tensor = self.S(ent.detach())

        loss = torch.mean(ent*selection)

        self.tta_optimizer.zero_grad()
        self.S_optimizer.zero_grad()
        loss.backward()
        self.tta_optimizer.step()
        self.S_optimizer.step()

        self.n_selective += selection.sum().item()
        self.n_total += selection.numel()

        _logger.info(f"inlcude={self.n_selective}, non-include={self.n_total-self.n_selective}, "
                     f"weight={self.S.weight.tolist()}, bias={self.S.bias.tolist()}")

        return outputs

    def __str__(self) -> str:
        return f"Tent with tta_lr {self.tta_lr}"
