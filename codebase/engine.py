import copy
import logging
import functools

import torch
import torch.nn as nn
import torch.utils.data as data

from codebase.core.tta.base import CloudEdgeTestTimeAdaptation
from codebase.torchutils.distributed import world_size
from codebase.torchutils.metrics import AccuracyMetric, AverageMetric, EstimatedTimeArrival, MovingAverageMetric
from codebase.criterion.tta import entropy
from codebase.torchutils.common import ThroughputTester, time_enumerate
from codebase.core.bn_utils import model_diff
# from codebase.core.pg import ContinuePolicyGradient
from codebase.core.evaluator import mask_with_entropy

_logger = logging.getLogger(__name__)


def tta_one_epoch(
    corruption_type: str,
    model: nn.Module,
    loader: data.DataLoader,
    ttadapter: CloudEdgeTestTimeAdaptation,
    criterion: nn.modules.loss._Loss,
    device: str,
    log_interval: int
):
    original_model = copy.deepcopy(model)

    time_cost_metric = AverageMetric("time_cost")
    loss_metric = AverageMetric("loss")
    entropy_metric = AverageMetric("entropy")
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    eta = EstimatedTimeArrival(len(loader))
    speed_tester = ThroughputTester()

    for time_cost, iter_, (inputs, targets) in time_enumerate(loader, start=1):
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        outputs = ttadapter.infer(inputs)

        with torch.no_grad():
            loss: torch.Tensor = criterion(outputs, targets)
            entropy_loss: torch.Tensor = entropy(outputs, reduction="mean")

        time_cost_metric.update(time_cost)
        loss_metric.update(loss)
        entropy_metric.update(entropy_loss)
        accuracy_metric.update(outputs, targets)
        eta.step()
        speed_tester.update(targets)

        if iter_ % log_interval == 0 or iter_ == len(loader):
            _logger.info(", ".join([
                f"corruption_type={corruption_type}",
                f"iter={iter_:05d}/{len(loader):05d}",
                f"fetch data time cost={time_cost_metric.compute()*1000:.2f}ms",
                f"fps={speed_tester.compute()*world_size():.0f} images/s",
                f"{loss_metric}",
                f"{entropy_metric}",
                f"{accuracy_metric}",
                # f"model_diff={model_diff(model, original_model).item():.8f}",
                f"{eta}",
            ]))
            if hasattr(ttadapter, "extra_log"):
                _logger.info(ttadapter.extra_log())
            time_cost_metric.reset()
            speed_tester.reset()

    # torch.save(ttadapter.labels_collection, "labels_collection.pt")
    results = {
        f"loss": loss_metric.compute(),
        f"entropy": entropy_metric.compute(),
        f"top1_acc": accuracy_metric.at(1).rate,
        f"top5_acc": accuracy_metric.at(5).rate,
        f"extra_log": "",
    }
    if hasattr(ttadapter, "extra_log"):
        results["extra_log"] = ttadapter.extra_log()
    return results

