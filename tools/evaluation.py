import argparse
import os

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmgen.apis import set_random_seed
from mmgen.core import (build_metric, single_gpu_evaluation,
                        single_gpu_online_evaluation)
from mmgen.datasets import build_dataloader, build_dataset
from mmgen.models import build_model
from mmgen.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a GAN model')
    parser.add_argument('config', help='evaluation config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
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
        help='whether to use online mode for evaluation')
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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

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
    if not distributed:
        _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])

        # build metrics
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
            metrics = [
                build_metric(cfg.metrics[metric]) for metric in cfg.metrics
            ]

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
            if cfg.data.get('test', None) and cfg.data.test.get(
                    'imgs_root', None):
                dataset = build_dataset(cfg.data.test)
            elif cfg.data.get('val', None) and cfg.data.val.get(
                    'imgs_root', None):
                dataset = build_dataset(cfg.data.val)
            elif cfg.data.get('train', None):
                # we assume that the train part should work well
                dataset = build_dataset(cfg.data.train)
            else:
                raise RuntimeError('There is no valid dataset config to run, '
                                   'please check your dataset configs.')
            data_loader = build_dataloader(
                dataset,
                samples_per_gpu=args.batch_size,
                workers_per_gpu=cfg.data.get('val_workers_per_gpu',
                                             cfg.data.workers_per_gpu),
                dist=distributed,
                shuffle=True)

        if args.sample_cfg is None:
            args.sample_cfg = dict()

        # online mode will not save samples
        if args.online and len(metrics) > 0:
            single_gpu_online_evaluation(model, data_loader, metrics, logger,
                                         basic_table_info, args.batch_size,
                                         **args.sample_cfg)
        else:
            single_gpu_evaluation(model, data_loader, metrics, logger,
                                  basic_table_info, args.batch_size,
                                  args.samples_path, **args.sample_cfg)
    else:
        raise NotImplementedError("We hasn't implemented multi gpu eval yet.")


if __name__ == '__main__':
    main()
