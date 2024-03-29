import math

import torch
from codebase.criterion.tta import entropy
from codebase.core.tta.base import CloudEdgeTestTimeAdaptation


# def evaluate(weights: torch.Tensor, ttadapter: CloudEdgeTestTimeAdaptation, mask_fn):
#     for weight in weights.unbind(dim=0):
#         _ = ttadapter.infer(inputs, mask_fn=functools.partial(mask_with_entropy, weights=act))

#             with torch.no_grad():
#                 outputs = model(inputs)
#                 # loss: torch.Tensor = criterion(outputs, targets)
#                 # entropy_loss: torch.Tensor = entropy(outputs, reduction="mean")
#             accuracy_metric.update(outputs, targets)
#             rewards = torch.tensor([[accuracy_metric.at(1).rate]], device=outputs.device)



def mask_with_entropy(weights: torch.Tensor, outputs: torch.Tensor, n_classes: int = 1000):
    with torch.no_grad():
        # if len(weights.shape) == 1:
        #     weights = weights.unsqueeze(dim=0)
        ent = entropy(outputs, reduction="none")
        normalized_ent = ent / math.log(n_classes)
        # metrics = normalized_ent
        one = torch.ones_like(normalized_ent)
        metrics = torch.stack([normalized_ent, one], dim=1)

        mask = (metrics*weights).sum(dim=-1, keepdim=False) < 0
        # import ipdb; ipdb.set_trace()
        return mask.float()
