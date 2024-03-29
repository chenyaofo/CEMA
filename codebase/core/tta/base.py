import torch
import torch.nn as nn

import logging

_logger = logging.getLogger(__name__)


class CloudEdgeTestTimeAdaptation:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

        _logger.info(f"Build {self}")

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            x = x[-1]
        with torch.no_grad():
            return self.model(x)
    
    def reset(self):
        pass

    def __str__(self) -> str:
        return f"TTABaseline"
