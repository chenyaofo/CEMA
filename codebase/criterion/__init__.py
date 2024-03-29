import torch.nn as nn

from .register import CRITERION

# from .label_smooth import LabelSmoothCrossEntropyLoss
from .distillation import TTADistillationLoss

CRITERION.register(nn.CrossEntropyLoss)
CRITERION.register(TTADistillationLoss)
