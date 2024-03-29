import torch
import torch.nn as nn

from .base import CloudEdgeTestTimeAdaptation


class PredictionBatchNorm(CloudEdgeTestTimeAdaptation):
    def __init__(self, model: nn.Module) -> None:
        super(PredictionBatchNorm, self).__init__(model)
        self.model.train()

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)

    def __str__(self) -> str:
        return f"PredictionBatchNorm"
