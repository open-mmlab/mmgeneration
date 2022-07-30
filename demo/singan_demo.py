# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys

import mmcv
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, set_random_seed
from mmengine import print_log
from mmengine.logging import MMLogger

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.core import *  # isort:skip  # noqa: F401,F403,E402
from mmgen.datasets import *  # isort:skip  # noqa: F401,F403,E402
from mmgen.models import *  # isort:skip  # noqa: F401,F403,E402

from mmgen.utils import register_all_modules  # isort:skip  # noqa
from mmgen.registry import MODELS  # isort:skip  # noqa

# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='SinGAN demo')
    parser.add_argument('config', help='evaluation config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos/singan_demo/',
        help=('path to store images. If not given, remove it after evaluation '
              'finished'))
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')
    parser.add_argument(
        '--save-prev-res',
        action='store_true',
        help='whether to store the results from previous stages')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='the number of synthesized samples')
    args = parser.parse_args()
    return args


def _tensor2img(img):
    img = img.permute(1, 2, 0)
    img = ((img + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

    return img.cpu().numpy()


@torch.no_grad()
def main():
    MMLogger.get_instance('mmgen')

    args = parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # set scope manually
    cfg.model['_scope_'] = 'mmgen'
    # build the model and load checkpoint
    model = MODELS.build(cfg.model)

    model.eval()

    # load ckpt
    print_log(f'Loading ckpt from {args.checkpoint}', 'mmgen')
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    model.to(args.device)

    pbar = mmcv.ProgressBar(args.num_samples)
    for sample_iter in range(args.num_samples):
        outputs = model.test_step(
            dict(num_batches=1, get_prev_res=args.save_prev_res))

        # batch size must be 1
        outputs = outputs[0]
        # store results from previous stages
        if args.save_prev_res:
            # fake_img = outputs['fake_img']
            fake_img = outputs.fake_img.data
            # prev_res_list = outputs['prev_res_list']
            prev_res_list = outputs.prev_res_list
            prev_res_list.append(fake_img)
            for i, img in enumerate(prev_res_list):
                img = _tensor2img(img)
                mmcv.imwrite(
                    img,
                    os.path.join(args.save_path, f'stage{i}',
                                 f'rand_sample_{sample_iter}.png'))
        # just store the final result
        else:
            img = _tensor2img(outputs)
            mmcv.imwrite(
                img,
                os.path.join(args.save_path, f'rand_sample_{sample_iter}.png'))

        pbar.update()

    # change the line after pbar
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
