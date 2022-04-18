# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmgen.apis import set_random_seed
from mmgen.core import build_metric, offline_evaluation, online_evaluation
from mmgen.datasets import build_dataloader, build_dataset
from mmgen.models import build_model
from mmgen.utils import get_root_logger

_distributed_metrics = ['FID', 'IS']


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a Generation model')
    parser.add_argument('config', help='evaluation config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--batch-size', type=int, default=10, help='batch size of dataloader')
    parser.add_argument(
        '--samples-path',
        type=str,
        default=None,
        help='path to store images. If not given, remove it after evaluation\
             finished')
    parser.add_argument(
        '--sample-model',
        type=str,
        default='ema',
        choices=['ema', 'orig'],
        help='use which mode (ema/orig) in sampling')
    parser.add_argument(
        '--eval',
        nargs='*',
        type=str,
        default=None,
        help='select the metrics you want to access')
    parser.add_argument(
        '--online',
        action='store_true',
        help='whether to use online mode for evaluation')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=-1,
        help='The number of images to be sampled for evaluation.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank = 0
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
        assert args.online or world_size == 1, (
            'We only support online mode for distrbuted evaluation.')

    dirname = os.path.dirname(args.checkpoint)
    ckpt = os.path.basename(args.checkpoint)

    if 'http' in args.checkpoint:
        log_path = None
    else:
        log_name = ckpt.split('.')[0] + '_eval_log' + '.txt'
        log_path = os.path.join(dirname, log_name)

    logger = get_root_logger(
        log_file=log_path, log_level=cfg.log_level, file_mode='a')
    logger.info('evaluation')

    # set random seeds
    if args.seed is not None:
        if rank == 0:
            mmcv.print_log(f'set random seed to {args.seed}', 'mmgen')
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # sanity check for models without ema
    if not model.use_ema:
        args.sample_model = 'orig'

    mmcv.print_log(f'Sampling model: {args.sample_model}', 'mmgen')

    model.eval()

    if args.eval:
        if args.eval[0] == 'none':
            # only sample images
            metrics = []
            assert args.num_samples is not None and args.num_samples > 0
        else:
            metrics = [
                build_metric(cfg.metrics[metric]) for metric in args.eval
            ]
    else:
        metrics = [build_metric(cfg.metrics[metric]) for metric in cfg.metrics]

    # check metrics for dist evaluation
    if distributed and metrics:
        for metric in metrics:
            assert metric.name in _distributed_metrics, (
                f'We only support {_distributed_metrics} for multi gpu '
                f'evaluation, but receive {args.eval}.')

    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    basic_table_info = dict(
        train_cfg=os.path.basename(cfg._filename),
        ckpt=ckpt,
        sample_model=args.sample_model)

    if len(metrics) == 0:
        basic_table_info['num_samples'] = args.num_samples
        data_loader = None
    else:
        basic_table_info['num_samples'] = -1
        # build the dataloader
        if cfg.data.get('test', None) and cfg.data.test.get('imgs_root', None):
            dataset = build_dataset(cfg.data.test)
        elif cfg.data.get('val', None) and cfg.data.val.get('imgs_root', None):
            dataset = build_dataset(cfg.data.val)
        elif cfg.data.get('train', None):
            # we assume that the train part should work well
            dataset = build_dataset(cfg.data.train)
        else:
            raise RuntimeError('There is no valid dataset config to run, '
                               'please check your dataset configs.')

        # The default loader config
        loader_cfg = dict(
            samples_per_gpu=args.batch_size,
            workers_per_gpu=cfg.data.get('val_workers_per_gpu',
                                         cfg.data.workers_per_gpu),
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=True)
        # The overall dataloader settings
        loader_cfg.update({
            k: v
            for k, v in cfg.data.items() if k not in [
                'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
                'test_dataloader'
            ]
        })

        # specific config for test loader
        test_loader_cfg = {**loader_cfg, **cfg.data.get('test_dataloader', {})}

        data_loader = build_dataloader(dataset, **test_loader_cfg)
    if args.sample_cfg is None:
        args.sample_cfg = dict()

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

    # online mode will not save samples
    if args.online and len(metrics) > 0:
        online_evaluation(model, data_loader, metrics, logger,
                          basic_table_info, args.batch_size, **args.sample_cfg)
    else:
        offline_evaluation(model, data_loader, metrics, logger,
                           basic_table_info, args.batch_size,
                           args.samples_path, **args.sample_cfg)


if __name__ == '__main__':
    main()
