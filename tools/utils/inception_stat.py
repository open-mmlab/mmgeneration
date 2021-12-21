# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import pickle
import sys

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv import Config, print_log

# yapf: disable
sys.path.append(osp.abspath(osp.join(__file__, '../../..')))  # isort:skip  # noqa

from mmgen.core.evaluation.metric_utils import extract_inception_features  # isort:skip  # noqa
from mmgen.datasets import (UnconditionalImageDataset, build_dataloader,  # isort:skip  # noqa
                            build_dataset)  # isort:skip  # noqa
from mmgen.models.architectures import InceptionV3  # isort:skip  # noqa
# yapf: enable

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-calculate inception data and save it in pkl file')
    parser.add_argument(
        '--imgsdir', type=str, default=None, help='the dir containing images.')
    parser.add_argument(
        '--data-cfg',
        type=str,
        default=None,
        help='the config file for test data pipeline')
    parser.add_argument(
        '--pklname', type=str, help='the name of inception pkl')
    parser.add_argument(
        '--pkl-dir',
        type=str,
        default='work_dirs/inception_pkl',
        help='path to save pkl file')
    parser.add_argument(
        '--pipeline-cfg',
        type=str,
        default=None,
        help=('config file containing dataset pipeline. If None, the default'
              ' pipeline will be adopted'))
    parser.add_argument(
        '--flip', action='store_true', help='whether to flip real images')
    parser.add_argument(
        '--size',
        type=int,
        nargs='+',
        default=(299, 299),
        help='image size in the data pipeline')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=25,
        help='batch size used in extracted features')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=50000,
        help=('the number of total samples, if input -1, '
              'automaticly use all samples in the subset'))
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='not use shuffle in data loader')
    parser.add_argument(
        '--subset',
        default='test',
        help='which subset and corresponding pipeline to use')
    parser.add_argument(
        '--inception-style',
        choices=['stylegan', 'pytorch'],
        default='pytorch',
        help='which inception network to use')
    parser.add_argument(
        '--inception-pth',
        type=str,
        default='work_dirs/cache/inception-2015-12-05.pt')
    args = parser.parse_args()

    # dataset pipeline (only be used when args.imgsdir is not None)
    if args.pipeline_cfg is not None:
        pipeline = Config.fromfile(args.pipeline_cfg)['inception_pipeline']
    elif args.imgsdir is not None:
        if isinstance(args.size, list) and len(args.size) == 2:
            size = args.size
        elif isinstance(args.size, list) and len(args.size) == 1:
            size = (args.size[0], args.size[0])
        elif isinstance(args.size, int):
            size = (args.size, args.size)
        else:
            raise TypeError(
                f'args.size mush be int or tuple but got {args.size}')

        pipeline = [
            dict(type='LoadImageFromFile', key='real_img'),
            dict(
                type='Resize', keys=['real_img'], scale=size,
                keep_ratio=False),
            dict(
                type='Normalize',
                keys=['real_img'],
                mean=[127.5] * 3,
                std=[127.5] * 3,
                to_rgb=True),  # default to RGB images
            dict(type='Collect', keys=['real_img'], meta_keys=[]),
            dict(type='ImageToTensor', keys=['real_img'])
        ]
        # insert flip aug
        if args.flip:
            pipeline.insert(
                1,
                dict(type='Flip', keys=['real_img'], direction='horizontal'))

    # build dataloader
    if args.imgsdir is not None:
        dataset = UnconditionalImageDataset(args.imgsdir, pipeline)
    elif args.data_cfg is not None:
        # Please make sure the dataset will sample images in `RGB` order.
        data_config = Config.fromfile(args.data_cfg)
        subset_config = data_config.data.get(args.subset, None)
        print_log(subset_config, 'mmgen')
        dataset = build_dataset(subset_config)
    else:
        raise RuntimeError('Please provide imgsdir or data_cfg')

    data_loader = build_dataloader(
        dataset, args.batch_size, 4, dist=False, shuffle=(not args.no_shuffle))

    mmcv.mkdir_or_exist(args.pkl_dir)

    # build inception network
    if args.inception_style == 'stylegan':
        inception = torch.jit.load(args.inception_pth).eval().cuda()
        inception = nn.DataParallel(inception)
        print_log('Adopt Inception network in StyleGAN', 'mmgen')
    else:
        inception = nn.DataParallel(
            InceptionV3([3], resize_input=True, normalize_input=False).cuda())
        inception.eval()

    if args.num_samples == -1:
        print_log('Use all samples in subset', 'mmgen')
        num_samples = len(dataset)
    else:
        num_samples = args.num_samples

    features = extract_inception_features(data_loader, inception, num_samples,
                                          args.inception_style).numpy()

    # sanity check for the number of features
    assert features.shape[
        0] == num_samples, 'the number of features != num_samples'
    print_log(f'Extract {num_samples} features', 'mmgen')

    mean = np.mean(features, 0)
    cov = np.cov(features, rowvar=False)

    with open(osp.join(args.pkl_dir, args.pklname), 'wb') as f:
        pickle.dump(
            {
                'mean': mean,
                'cov': cov,
                'size': num_samples,
                'name': args.pklname
            }, f)
