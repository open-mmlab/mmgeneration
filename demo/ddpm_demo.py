# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import sys

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import init_model, sample_ddpm_model  # isort:skip  # noqa
# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='DDPM demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos/ddpm_samples.png',
        help='path to save unconditional samples')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')

    # args for inference/sampling
    parser.add_argument(
        '--num-batches', type=int, default=4, help='Batch size in inference')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=12,
        help='The total number of samples')
    parser.add_argument(
        '--sample-model',
        type=str,
        default='ema',
        help='Which model to use for sampling')
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')
    parser.add_argument(
        '--same-noise',
        action='store_true',
        help='whether use same noise as input (x_T)')
    parser.add_argument(
        '--n-skip',
        type=int,
        default=25,
        help=('Skip how many steps before selecting one to visualize. This is '
              'helpful with denoising timestep is too much. Only work with '
              '`save-path` is end with \'.gif\'.'))

    # args for image grid
    parser.add_argument(
        '--padding', type=int, default=0, help='Padding in the image grid.')
    parser.add_argument(
        '--nrow',
        type=int,
        default=2,
        help=('Number of images displayed in each row of the grid. '
              'This argument would work only when label is not given.'))

    # args for image channel order
    parser.add_argument(
        '--is-rgb',
        action='store_true',
        help=('If true, color channels will not be permuted, This option is '
              'useful when inference model trained with rgb images.'))

    args = parser.parse_args()
    return args


def create_gif(results, gif_name, fps=60, n_skip=1):
    """Create gif through imageio.

    Args:
        frames (torch.Tensor): Image frames, shape like [bz, 3, H, W].
        gif_name (str): Saved gif name.
        fps (int, optional): Frames per second of the generated gif.
            Defaults to 60.
        n_skip (int, optional): Skip how many steps before selecting one to
            visualize. Defaults to 1.
    """
    try:
        import imageio
    except ImportError:
        raise RuntimeError('imageio is not installed,'
                           'Please use “pip install imageio” to install')
    frames_list = []
    for frame in results[::n_skip]:
        frames_list.append(
            (frame.permute(1, 2, 0).cpu().numpy() * 255.).astype(np.uint8))

    # ensure the final denoising results in frames_list
    if not (len(results) % n_skip == 0):
        frames_list.append((results[-1].permute(1, 2, 0).cpu().numpy() *
                            255.).astype(np.uint8))

    imageio.mimsave(gif_name, frames_list, 'GIF', fps=fps)


def main():
    args = parse_args()
    model = init_model(
        args.config, checkpoint=args.checkpoint, device=args.device)

    if args.sample_cfg is None:
        args.sample_cfg = dict()

    suffix = osp.splitext(args.save_path)[-1]
    if suffix == '.gif':
        args.sample_cfg['save_intermedia'] = True

    results = sample_ddpm_model(model, args.num_samples, args.num_batches,
                                args.sample_model, args.same_noise,
                                **args.sample_cfg)

    # save images
    mmcv.mkdir_or_exist(os.path.dirname(args.save_path))
    if suffix == '.gif':
        # concentrate all output of each timestep
        results_timestep_list = []
        for t in results.keys():
            # make grid
            results_timestep = utils.make_grid(
                results[t], nrow=args.nrow, padding=args.padding)
            # unsqueeze at 0, because make grid output is size like [H', W', 3]
            results_timestep_list.append(results_timestep[None, ...])

        # Concatenates to [n_timesteps, H', W', 3]
        results_timestep = torch.cat(results_timestep_list, dim=0)
        if not args.is_rgb:
            results_timestep = results_timestep[:, [2, 1, 0]]
        results_timestep = (results_timestep + 1.) / 2.
        create_gif(results_timestep, args.save_path, n_skip=args.n_skip)
    else:
        if not args.is_rgb:
            results = results[:, [2, 1, 0]]

        results = (results + 1.) / 2.
        utils.save_image(
            results, args.save_path, nrow=args.nrow, padding=args.padding)


if __name__ == '__main__':
    main()
