from datetime import datetime
import typing

import torch
import torch.distributed as dist

EPSILON = 1e-8

_str_2_reduceop = dict(
    sum=dist.ReduceOp.SUM,
    mean=dist.ReduceOp.SUM,
    product=dist.ReduceOp.PRODUCT,
    min=dist.ReduceOp.MIN,
    max=dist.ReduceOp.MAX,
)


def _all_reduce(*args, reduction="sum"):
    t = torch.tensor(args, dtype=torch.float).cuda()
    dist.all_reduce(t, op=_str_2_reduceop[reduction])
    rev = t.tolist()
    if reduction == "mean":
        world_size = dist.get_world_size()
        rev = [item/world_size for item in rev]
    return rev


class Accuracy(object):
    def __init__(self):
        self._is_distributed = dist.is_available() and dist.is_initialized()
        self.reset()

    def reset(self):
        self._n_correct = 0.0
        self._n_total = 0.0
        self._reset_buffer()

    @property
    def rate(self):
        self.sync()
        return self._n_correct / (self._n_total+1e-8)

    @property
    def n_correct(self):
        self.sync()
        return self._n_correct

    @property
    def n_total(self):
        self.sync()
        return self._n_total

    def _reset_buffer(self):
        self._n_correct_since_last_sync = 0.0
        self._n_total_since_last_sync = 0.0
        self._is_synced = True

    def update(self,  n_correct, n_total):
        self._n_correct_since_last_sync += n_correct
        self._n_total_since_last_sync += n_total
        self._is_synced = False

    def sync(self):
        if self._is_synced:
            return
        n_correct = self._n_correct_since_last_sync
        n_total = self._n_total_since_last_sync
        if self._is_distributed:
            n_correct, n_total = _all_reduce(n_correct, n_total, reduction="sum")

        self._n_correct += n_correct
        self._n_total += n_total

        self._reset_buffer()


class AccuracyMetric(object):
    def __init__(self, topk: typing.Iterable[int] = (1,),):
        self.topk = sorted(list(topk))
        self.reset()

    def reset(self) -> None:
        self.accuracies = [Accuracy() for _ in self.topk]

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        maxk = max(self.topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))

        for accuracy, k in zip(self.accuracies, self.topk):
            correct_k = correct[:k].sum().item()
            accuracy.update(correct_k, batch_size)

    def at(self, topk: int) -> Accuracy:
        if topk not in self.topk:
            raise ValueError(f"topk={topk} is not in registered topks={self.topk}")
        accuracy = self.accuracies[self.topk.index(topk)]
        accuracy.sync()
        return accuracy

    def __str__(self):
        items = [f"top{k}-acc={self.at(k).rate*100:.2f}%" for k in self.topk]
        return ", ".join(items)


class ConfusionMatrix(object):
    def __init__(self, num_classes: int):
        self._is_distributed = dist.is_available() and dist.is_initialized()
        self.num_classes = num_classes
        self.matrix = None
        self.reset()

    def reset(self):
        self.matrix = torch.zeros(size=(self.num_classes,)*2,
                                  dtype=torch.int64, device="cuda")
        self._reset_buffer()

    def _reset_buffer(self):
        self._matrix_since_last_sync = torch.zeros(size=(self.num_classes,)*2,
                                                   dtype=torch.int64, device="cuda")
        self._is_synced = True

    def update(self, targets: torch.Tensor, predictions: torch.Tensor):
        predictions = torch.argmax(predictions, dim=1)
        targets, predictions = targets.flatten(), predictions.flatten()
        indices = targets * self.num_classes + predictions
        m = torch.bincount(indices, minlength=self.num_classes **
                           2).reshape(self.num_classes, self.num_classes)

        self._matrix_since_last_sync += m.to(device=self.matrix.device)
        self._is_synced = False

    def sync(self):
        if self._is_synced:
            return
        self._is_synced = True

    def pixel_accuracy(self) -> float:
        self.sync()
        m = self.matrix.float()
        return (m.diag().sum()/(m.sum()+EPSILON)).item()

    def mean_pixel_accuracy(self) -> float:
        self.sync()
        m = self.matrix.float()
        return (m.diag()/m.sum(dim=1)).mean().item()

    def mean_intersection_over_union(self) -> float:
        self.sync()
        m = self.matrix.float()
        diag = m.diag()
        return (diag/(m.sum(dim=0)+m.sum(dim=1)-diag+EPSILON)).mean().item()


class AverageMetric(object):
    def __init__(self, name, format_digits=4):
        self.name = name
        self.format_digits = format_digits
        self._is_distributed = dist.is_available() and dist.is_initialized()
        self.reset()

    def reset(self,) -> None:
        self._n = 0
        self._value = 0.
        self._reset_buf()

    def _reset_buf(self):
        self._n_buf = 0
        self._value_vuf = 0.
        self._is_synced = True

    def sync(self):
        if self._is_synced:
            return
        n = self._n_buf
        value = self._value_vuf
        if self._is_distributed:
            n, value = _all_reduce(n, value)
        self._n += n
        self._value += value
        self._reset_buf()

    def update(self, value) -> None:
        if torch.is_tensor(value):
            self._value_vuf += value.item()
        elif isinstance(value, (int, float)):
            self._value_vuf += value
        else:
            raise ValueError("The parameter 'value' should be int, float or pytorch scalar tensor, but found {}"
                             .format(type(value)))
        self._n_buf += 1
        self._is_synced = False

    def compute(self) -> float:
        self.sync()
        return self._value / (self._n+EPSILON)

    def __str__(self) -> str:
        return f"{self.name}={self.compute():.{self.format_digits}f}"


class GroupMetric(object):
    def __init__(self, n, metric_cls):
        self.metrics = [metric_cls() for _ in range(n)]

    def reset(self):
        for m in self.metrics:
            m.reset()

    def update(self, values):
        for m, v in zip(self.metrics, values):
            if isinstance(v, (list, tuple)):
                m.update(*v)
            else:
                m.update(v)

    def compute(self):
        return [m.compute() for m in self.metrics]


class MetricGroup(object):
    def __init__(self, metrics=list()) -> None:
        self.metrics = metrics

    def append(self, metric, format_string, is_export=False):
        pass

    def reset():
        pass

    def update():
        pass


class EstimatedTimeArrival(object):
    def __init__(self, total):
        self.times = [datetime.now()]
        self.total = total

    def step(self):
        self.times.append(datetime.now())

    @property
    def remaining_time(self) -> datetime:
        if len(self.times) == 1:
            raise Exception("Cannot compute the remaining_time")

        n_internals = len(self.times) - 1
        remain = max(self.total-n_internals, 0)
        return (self.times[-1]-self.times[0])/n_internals*(remain)

    @property
    def arrival_time(self) -> datetime:
        return datetime.now() + self.remaining_time

    @property
    def cost_time(self) -> datetime:
        return self.times[-1]-self.times[0]

    def __str__(self) -> str:
        return f"remaining_time={self.remaining_time}, " + \
            f"arrival_time={self.arrival_time}, " + \
            f"cost_time={self.cost_time}"


class MovingAverageMetric(object):
    def __init__(self, gamma=0.9):
        self.n = 0
        self._value = 0.
        self.last = 0.
        self.gamma = gamma

    def reset(self) -> None:
        self.n = 0
        self._value = 0.
        self.last = 0.

    def update(self, value) -> None:
        if torch.is_tensor(value):
            self.last = value.item()
        elif isinstance(value, (int, float)):
            self.last = value
        else:
            raise ValueError("The parameter 'value' should be int, float or pytorch scalar tensor, but found {}"
                             .format(type(value)))
        self.n += 1
        if self.n == 1:
            self._value = self.last
        else:
            self._value = self._value * self.gamma + self.last * (1-self.gamma)

    @property
    def value(self) -> float:
        if self.n == 0:
            raise ValueError("The container is empty.")
        return self._value