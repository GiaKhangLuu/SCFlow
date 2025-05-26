import argparse
import time
from os import path as osp
from functools import partial

import mmcv
import mmengine
from mmengine.structures import BaseDataElement
import torch
import random
import numpy as np
from mmengine import Config, DictAction
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from mmengine.runner import  load_checkpoint
from mmengine.model import MMDistributedDataParallel
from torch.nn.parallel import DataParallel as MMDataParallel
from mmengine.dist import get_dist_info
from registry import DATASETS, MODELS
from mmengine.runner import Runner

from datasets import collate
from tools.eval import single_gpu_test, multi_gpu_test

# profile

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a pose estimator'
    )
    parser.add_argument(
        '--config', help='test config file path', default='configs/flow_refine/mvc_raft_flow.py')
    parser.add_argument(
        '--checkpoint', type=str, help='checkpoint file', default='',)
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results'
    )
    parser.add_argument(
        '--eval', action='store_true', help='whether to evaluate the results')
    parser.add_argument(
        '--format-only', action='store_true', help='whether to save the results in BOP format')
    parser.add_argument(
        '--save-dir', type=str, default='debug/results', help='directory for saving the formatted results')
    parser.add_argument(
        '--eval-options',
        action=DictAction,
        nargs='+',
        help='custom options for formating results, the key-value pair in xxx=yyy'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args  = parse_args()
    #if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        #raise ValueError('The output file must be a pickle file')
    cfg = Config.fromfile(args.config)

    cfg.load_from = args.checkpoint
    cfg.val_evaluator.save_dir = args.save_dir
    cfg.test_evaluator.save_dir = args.save_dir
    
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.get('dist_param', {}))

    runner = Runner.from_cfg(cfg)

    runner.test()