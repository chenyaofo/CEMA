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


# def tta_search_criterion(
#     corruption_type: str,
#     n_iters: int,
#     eval_iters: int,
#     policy: ContinuePolicyGradient,
#     model: nn.Module,
#     loader: data.DataLoader,
#     ttadapter: CloudEdgeTestTimeAdaptation,
#     criterion: nn.modules.loss._Loss,
#     device: str,
#     log_interval: int
# ):

#     time_cost_metric = AverageMetric("time_cost")
#     accuracy_metric = AccuracyMetric(topk=(1,))
#     eta = EstimatedTimeArrival(n_iters)
#     speed_tester = ThroughputTester()

#     cnt_iter = 0
#     n_remain = 0
#     n_total = 0
#     while True:
#         for time_cost, iter_, (inputs, targets) in time_enumerate(loader, start=1):
#             inputs = inputs.to(device=device, non_blocking=True)
#             targets = targets.to(device=device, non_blocking=True)

#             if cnt_iter % eval_iters == 0:
#                 dist, act, sigma = policy.make_decisions()
#             outputs, remain, total = ttadapter.infer(inputs, mask_fn=functools.partial(mask_with_entropy, weights=act))
#             n_remain += remain
#             n_total += total

#             accuracy_metric.update(outputs, targets)
#             if cnt_iter % eval_iters == 0:
#                 rewards = torch.tensor([[accuracy_metric.at(1).rate*100]], device=outputs.device)
#                 policy.learn(dist, act, rewards)
#             time_cost_metric.update(time_cost)


#             eta.step()
#             speed_tester.update(inputs)

#             if cnt_iter % eval_iters == 0:
#                 with torch.no_grad():
#                     dist, act, sigma = policy.make_decisions(eval_mode=True)
#                 _logger.info(", ".join([
#                     f"corruption_type={corruption_type}",
#                     f"iter={cnt_iter:05d}/{n_iters:05d}",
#                     f"fetch data time cost={time_cost_metric.compute()*1000:.2f}ms",
#                     f"fps={speed_tester.compute()*world_size():.0f} images/s",
#                     f"{accuracy_metric}",
#                     f"remain={n_remain}/{n_total} ({n_remain/n_total*100:.2f}%)",
#                     f"weights={act}",
#                     f"sigma={sigma}",
#                     f"{eta}",
#                 ]))
#                 time_cost_metric.reset()
#                 speed_tester.reset()

#                 accuracy_metric.reset()
#                 ttadapter.reset()
#                 n_remain = 0
#                 n_total = 0

#             cnt_iter += 1
#             if cnt_iter >= n_iters:
#                 with torch.no_grad():
#                     dist, act, sigma = policy.make_decisions(eval_mode=True)
#                     _logger.info(f"search complete, weights={act}, sigma={sigma}")
#                     return act

# def tta_search_criterion(
#     corruption_type: str,
#     alpha: float,
#     n_iters: int,
#     eval_iters: int,
#     policy: ContinuePolicyGradient,
#     model: nn.Module,
#     loader: data.DataLoader,
#     ttadapter: CloudEdgeTestTimeAdaptation,
#     criterion: nn.modules.loss._Loss,
#     device: str,
#     log_interval: int
# ):

#     time_cost_metric = AverageMetric("time_cost")
#     accuracy_metric = AccuracyMetric(topk=(1,))
#     eta = EstimatedTimeArrival(n_iters)
#     speed_tester = ThroughputTester()

#     cnt_iter = 0
#     n_remain = 0
#     n_total = 0
#     while True:
#         dist, act, sigma = policy.make_decisions()
#         for time_cost, iter_, (inputs, targets) in time_enumerate(loader, start=1):
#             inputs = inputs.to(device=device, non_blocking=True)
#             targets = targets.to(device=device, non_blocking=True)
                
#             outputs, remain, total = ttadapter.infer(inputs, mask_fn=functools.partial(mask_with_entropy, weights=act))
#             n_remain += remain
#             n_total += total

#             accuracy_metric.update(outputs, targets)

#             time_cost_metric.update(time_cost)
            
#             speed_tester.update(inputs)

#             if iter_ == 3:
#                 remain_percent = n_remain / n_total
#                 if remain_percent < 0.1 or remain_percent > 0.9:
#                     invalid = True
#                 else:
#                     invalid = False
#             if cnt_iter >= n_iters:
#                 with torch.no_grad():
#                     dist, act, sigma = policy.make_decisions(eval_mode=True)
#                     _logger.info(f"search complete, weights={act}, sigma={sigma}")
#                     return act
#         if invalid:
#             _logger.info(f"Search with remain={n_remain}/{n_total} ({n_remain/n_total*100:.2f}%), give up")
#         if not invalid:
#             cnt_iter += 1 
#             eta.step()

#             r = accuracy_metric.at(1).rate*100 + alpha*((n_total-n_remain)/n_total*100)
#             rewards = torch.tensor([[r]], device=outputs.device)
#             policy.learn(dist, act, rewards)

#             avg_r = policy.avg_reward.value

#             with torch.no_grad():
#                 dist, eval_act, eval_sigma = policy.make_decisions(eval_mode=True)
#             _logger.info(", ".join([
#                 f"corruption_type={corruption_type}",
#                 f"iter={cnt_iter:05d}/{n_iters:05d}",
#                 f"fetch data time cost={time_cost_metric.compute()*1000:.2f}ms",
#                 f"fps={speed_tester.compute()*world_size():.0f} images/s",
#                 f"reward={r:.2f}",
#                 f"avg_reward={avg_r:.2f}",
#                 f"{accuracy_metric}",
#                 f"remain={n_remain}/{n_total} ({n_remain/n_total*100:.2f}%)",
#                 f"training weights={act}",
#                 f"eval weights={eval_act}",
#                 f"sigma={eval_sigma}",
#                 f"{eta}",
#             ]))
#         time_cost_metric.reset()
#         speed_tester.reset()

#         accuracy_metric.reset()
#         ttadapter.reset()
#         n_remain = 0
#         n_total = 0