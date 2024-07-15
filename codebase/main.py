import os
import copy
import logging
import dataclasses
import pathlib
import pprint

import torch
from torch import optim
import torch.cuda
import torch.utils.data
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.tensorboard import SummaryWriter
from pyhocon import ConfigTree

from codebase.config import Args
from codebase.data import DATA
from codebase.models import MODEL
from codebase.optimizer import OPTIMIZER
from codebase.scheduler import SCHEDULER
from codebase.criterion import CRITERION
from codebase.core.tta import TTADAPTER
from codebase.engine import tta_one_epoch
from codebase.constant import iter_imagenet_c

# from codebase.core.pg import Actor, ContinuePolicyGradient

from codebase.torchutils.common import set_cudnn_auto_tune, set_reproducible, generate_random_seed, disable_debug_api
from codebase.torchutils.common import set_proper_device, get_device
from codebase.torchutils.common import only_master
from codebase.torchutils.logging import init_logger, create_code_snapshot


_logger = logging.getLogger(__name__)


def prepare_for_tta(
        model_config: ConfigTree,
        data_config: ConfigTree,
        criterion_config: ConfigTree,
        tta_config: ConfigTree,
        output_dir: str,
        local_rank: int):
    model: nn.Module = MODEL.build_from(model_config)
    if torch.cuda.is_available():
        model = model.to(device=get_device())
    print(tta_config)
    ttadapter = TTADAPTER.build_from(
        tta_config, default_args=dict(model=model))
    # print(ttadapter.foundation_model.preprocess)
    # test_loader = DATA.build_from(data_config, dict(local_rank=local_rank,clip_preprocess=ttadapter.foundation_model.preprocess))
    # print(model_config)
    # if "clip" in model_config.get("type_"):
    #     test_loader = DATA.build_from(data_config, dict(local_rank=local_rank,clip_preprocess=model.preprocess))
    # else:
    test_loader = DATA.build_from(data_config, dict(local_rank=local_rank))

    criterion = CRITERION.build_from(criterion_config)

    if torch.cuda.is_available():
        # model = model.to(device=get_device())
        criterion = criterion.to(device=get_device())

    return model, ttadapter, test_loader, criterion


def _init(local_rank: int, ngpus_per_node: int, args: Args):
    set_proper_device(local_rank)
    rank = args.node_rank*ngpus_per_node+local_rank
    init_logger(rank=rank, filenmae=args.output_dir/"default.log")

    set_reproducible(generate_random_seed())

    create_code_snapshot(name="code", include_suffix=[".py", ".conf"],
                         source_directory=".", store_directory=args.output_dir)

    _logger.info("Collect envs from system:\n" + get_pretty_env_info())
    _logger.info("Args:\n" + pprint.pformat(dataclasses.asdict(args)))


def main_worker(
    local_rank: int,
    ngpus_per_node: int,
    args: Args,
    conf: ConfigTree,
    hyper_params_search=False
):

    _init(local_rank=local_rank, ngpus_per_node=ngpus_per_node, args=args)

    data_config: ConfigTree = conf.get("data")

    corruptions = data_config.pop("corruptions")
    severities = data_config.pop("severities")

    for corrup_fine_grained_type, seve, path in iter_imagenet_c(
        root=os.path.join(data_config.get("root")),
        corruptions=corruptions,
        severities=severities,
        pass_through=data_config.get_bool("pass_through"),
        mixed=data_config.get_bool("mixed"),
    ):
        data_config_copy: ConfigTree = copy.deepcopy(data_config)
        data_config_copy["root"] = path

        _logger.info(
            f"Start TTA with corruption_type={corrup_fine_grained_type} (level={seve}) at path={path}")

        model, ttadapter, test_loader, criterion = prepare_for_tta(
            conf.get("model"),
            data_config_copy,
            conf.get("criterion"),
            conf.get('tta_strategy'),
            args.output_dir,
            local_rank
        )

        metrics = tta_one_epoch(
            corruption_type=corrup_fine_grained_type,
            model=model,
            ttadapter=ttadapter,
            loader=test_loader,
            criterion=criterion,
            device=get_device(),
            log_interval=conf.get_int("log_interval")
        )

        top1_acc = metrics['top1_acc']
        top5_acc = metrics['top5_acc']
        _logger.info(f"TTA completes for {corrup_fine_grained_type}(severity={seve}), "
                     f"val top1-acc={top1_acc*100:.4f}% (err={(1-top1_acc)*100:.4f}%), "
                     f"top5-acc={top5_acc*100:.4f}% (err={(1-top5_acc)*100:.4f}%), "
                     f"{metrics['extra_log']}")
        if hyper_params_search:
            return top1_acc


def main(args: Args, hyper_params_search=False):
    distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args, args.conf))
    else:
        local_rank = 0
        return main_worker(local_rank, ngpus_per_node, args, args.conf, hyper_params_search=hyper_params_search)
