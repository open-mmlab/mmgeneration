import argparse
import os
import shutil
import sys

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from torchvision.utils import save_image

from mmgen.apis import set_random_seed
from mmgen.core import build_metric
from mmgen.core.evaluation import make_metrics_table, make_vanilla_dataloader
from mmgen.datasets import build_dataloader, build_dataset
from mmgen.models import build_model
from mmgen.models.gans import CycleGAN, Pix2Pix
from mmgen.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a GAN model')
    parser.add_argument('config', help='evaluation config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='batch size of dataloader')
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
    args = parser.parse_args()
    return args


_supported_model = (Pix2Pix, CycleGAN)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

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
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    assert isinstance(model, _supported_model)
    # sanity check for models without ema
    if not model.use_ema:
        args.sample_model = 'orig'

    mmcv.print_log(f'Sampling model: {args.sample_model}', 'mmgen')

    model.eval()

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
        metrics = [build_metric(cfg.metrics[metric]) for metric in cfg.metrics]

    basic_table_info = dict(
        train_cfg=os.path.basename(cfg._filename),
        ckpt=ckpt,
        sample_model=args.sample_model)

    # build the dataloader
    if len(metrics) == 0:
        basic_table_info['num_samples'] = args.num_samples
        data_loader = None
    else:
        basic_table_info['num_samples'] = -1
        if cfg.data.get('test', None):
            dataset = build_dataset(cfg.data.test)
        else:
            dataset = build_dataset(cfg.data.train)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=args.batch_size,
            workers_per_gpu=cfg.data.get('val_workers_per_gpu',
                                         cfg.data.workers_per_gpu),
            dist=False,
            shuffle=True)

    # decide samples path
    samples_path = args.samples_path
    delete_samples_path = False
    if samples_path:
        mmcv.mkdir_or_exist(samples_path)
    else:
        temp_path = './work_dirs/temp_samples'
        # if temp_path exists, add suffix
        suffix = 1
        samples_path = temp_path
        while os.path.exists(samples_path):
            samples_path = temp_path + '_' + str(suffix)
            suffix += 1
        os.makedirs(samples_path)
        delete_samples_path = True

    # sample images
    num_exist = len(
        list(
            mmcv.scandir(
                samples_path, suffix=('.jpg', '.png', '.jpeg', '.JPEG'))))
    if basic_table_info['num_samples'] > 0:
        max_num_images = basic_table_info['num_samples']
    else:
        max_num_images = max(metric.num_images for metric in metrics)
    num_needed = max(max_num_images - num_exist, 0)

    if num_needed > 0:
        mmcv.print_log(f'Sample {num_needed} fake images for evaluation',
                       'mmgen')
        # define mmcv progress bar
        pbar = mmcv.ProgressBar(num_needed)
    # select key to fetch fake images
    fake_key = 'fake_b'
    if isinstance(model.module, CycleGAN):
        fake_key = 'fake_b' if model.module.test_direction == 'a2b' else \
            'fake_a'
    # if no images, `num_exist` should be zero
    for begin in range(num_exist, num_needed, args.batch_size):
        end = min(begin + args.batch_size, max_num_images)
        # for translation model, we feed them images from dataloader
        data_loader_iter = iter(data_loader)
        data_batch = next(data_loader_iter)
        output_dict = model(test_mode=True, **data_batch)
        fakes = output_dict[fake_key]
        pbar.update(end - begin)
        for i in range(end - begin):
            images = fakes[i:i + 1]
            images = ((images + 1) / 2)
            images = images[:, [2, 1, 0], ...]
            images = images.clamp_(0, 1)
            image_name = str(begin + i) + '.png'
            save_image(images, os.path.join(samples_path, image_name))

    if num_needed > 0:
        sys.stdout.write('\n')

    # return if only save sampled images
    if len(metrics) == 0:
        return

    # empty cache to release GPU memory
    torch.cuda.empty_cache()
    fake_dataloader = make_vanilla_dataloader(samples_path, args.batch_size)
    # select key to fetch real images
    if isinstance(model.module, CycleGAN):
        real_key = 'img_b' if model.module.test_direction == 'a2b' else 'img_a'
        if model.module.direction == 'b2a':
            real_key = 'img_a' if real_key == 'img_b' else 'img_b'

    if isinstance(model.module, Pix2Pix):
        real_key = 'img_b' if model.module.direction == 'a2b' else 'img_a'

    for metric in metrics:
        mmcv.print_log(f'Evaluate with {metric.name} metric.', 'mmgen')
        metric.prepare()
        # feed in real images
        for data in data_loader:
            reals = data[real_key]
            num_left = metric.feed(reals, 'reals')
            if num_left <= 0:
                break
        # feed in fake images
        for data in fake_dataloader:
            fakes = data['real_img']
            num_left = metric.feed(fakes, 'fakes')
            if num_left <= 0:
                break
        metric.summary()
    table_str = make_metrics_table(basic_table_info['train_cfg'],
                                   basic_table_info['ckpt'],
                                   basic_table_info['sample_model'], metrics)
    logger.info('\n' + table_str)
    if delete_samples_path:
        shutil.rmtree(samples_path)


if __name__ == '__main__':
    main()
