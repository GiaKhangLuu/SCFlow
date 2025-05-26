import os, time
import argparse
from functools import partial
from os import path as osp
import warnings
import torch
import mmcv
from mmengine import Config
from mmengine.logging import MMLogger
from torch.nn.parallel import DataParallel as MMDataParallel
from mmengine.model import MMDistributedDataParallel
from mmengine.runner import Runner
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from mmengine.dist import (
    get_dist_info, init_dist)

#from models import build_refiner
#from datasets import build_dataset
from tools.eval import single_gpu_test, multi_gpu_test

from datasets import collate

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose refiner')
    parser.add_argument('--config', default='configs/refine_models/scflow.py', help='train config file path')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--launcher', default='none', choices=['none', 'slurm', 'mpi', 'pytorch'], help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    args = parse_args()

    cfg_path = args.config
    launcher = args.launcher
    
    cfg = Config.fromfile(cfg_path)
    if launcher != 'none':
        distributed = True
        init_dist(launcher, **cfg.get('dist_param', {}))
        _, world_size = get_dist_info()
    else:
        distributed = False

    cfg.resume = args.resume

    runner = Runner.from_cfg(cfg)

    runner.train()