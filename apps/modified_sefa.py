"""Modified SeFa (closed-form factorization)

This gan editing method is modified according to Sefa. More details can be
found in Positional Encoding as Spatial Inductive Bias in GANs, CVPR2021.

The major modifications are:
- Calculate eigen vectors on the matrix with all style modulation weights in
  styleconvs;
- Allow to adopt unsymetric degree to be more robust to different samples.
"""

import argparse
import os
import sys

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import load_checkpoint
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import set_random_seed  # isort:skip  # noqa
from mmgen.models import build_model  # isort:skip  # noqa

# yapf: enable


def calc_eigens(args, state_dict):
    # get all of the style modulation weights except for weights in `to_rgb`
    modulated = {
        k: v
        for k, v in state_dict.items()
        if 'style_modulation' in k and 'to_rgb' not in k and 'weight' in k
    }

    weight_mat = []
    for _, v in modulated.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, dim=0)
    eigen_vector = torch.svd(W).V

    # save eigen vector
    output_path = os.path.splitext(args.ckpt)[0] + '_eigen-vec-mod.pth'
    torch.save({'ckpt': args.ckpt, 'eigen_vector': eigen_vector}, output_path)

    return eigen_vector


if __name__ == '__main__':
    # set device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # set grad enabled = False
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(
        description='Apply modified closed form factorization')

    # sefa args
    parser.add_argument(
        '-i', '--index', type=int, default=0, help='index of eigenvector')
    parser.add_argument(
        '-d',
        '--degree',
        type=float,
        nargs='+',
        default=2,
        help='scalar factors for moving latent vectors along eigenvector',
    )
    parser.add_argument(
        '--degree-step',
        type=float,
        default=0.25,
        help='The step of changing degrees')
    parser.add_argument('-l', '--layer-num', nargs='+', type=int, default=None)
    parser.add_argument(
        '--eigen-vector',
        type=str,
        default=None,
        help='Path to the eigen vectors')

    # gan args
    parser.add_argument(
        '--randomize-noise',
        action='store_true',
        help='whether to use random noise in the middle layers')
    parser.add_argument('--ckpt', type=str, help='Path to the checkpoint')
    parser.add_argument('--config', type=str, help='Path to model config')
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--noise-channels', type=int, default=512)
    parser.add_argument('--input-scale', type=int, default=4)
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')

    # system args
    parser.add_argument('--num-samples', type=int, default=2)
    parser.add_argument('--sample-path', type=str, default=None)
    parser.add_argument('--random-seed', type=int, default=2020)

    args = parser.parse_args()

    set_random_seed(args.random_seed)
    cfg = mmcv.Config.fromfile(args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    mmcv.print_log('Building models and loading checkpoints', 'mmgen')
    # build model
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    model.eval()
    load_checkpoint(model, args.ckpt, map_location='cpu')

    # get generator
    if model.use_ema:
        generator = model.generator_ema
    else:
        generator = model.generator

    generator = generator.to(device)
    generator.eval()

    mmcv.print_log('Calculating or loading eigen vectors', 'mmgen')
    # load/calculate eigen vector for current weights
    if args.eigen_vector is None:
        eigen_vector = calc_eigens(args, generator.state_dict())
    else:
        eigen_vector = torch.load(args.eigen_vector)['eigen_vector']
        eigen_vector = eigen_vector.to(device)

    if args.truncation < 1:
        # TODO: get mean latent
        mean_latent = generator.get_mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    noise = torch.randn((args.num_samples, args.noise_channels), device=device)
    latent = generator.style_mapping(noise)

    # kwargs for different gan models
    kwargs = dict()
    # mspie-stylegan2
    if args.input_scale > 0:
        kwargs['chosen_scale'] = args.input_scale

    if args.sample_cfg is None:
        args.sample_cfg = dict()

    mmcv.print_log('Sampling images with modified SeFa', 'mmgen')
    sample = generator([latent], input_is_latent=True, **args.sample_cfg)

    # the first line is the original samples
    img_list = [sample]
    if len(args.degree) == 1:
        factor_list = np.arange(-args.degree[0], args.degree[0] + 0.001,
                                args.degree_step)
    else:
        factor_list = np.arange(args.degree[0], args.degree[1] + 0.001,
                                args.degree_step)

    for fac in factor_list:
        direction = fac * eigen_vector[:, args.index].unsqueeze(0)
        if args.layer_num is None:
            latent_input = [latent + direction]
        else:
            latent_all = latent.unsqueeze(1).repeat(1, generator.num_latents,
                                                    1)
            for l_num in args.layer_num:
                latent_all[:, l_num] = latent + direction
            latent_input = [latent_all]
        sample = generator(
            latent_input, input_is_latent=True, **args.sample_cfg)
        img_list.append(sample)

    mmcv.mkdir_or_exist(args.sample_path)
    if args.layer_num is None:
        filename = (
            f'{args.sample_path}/entangle-i{args.index}-d{args.degree}'
            f'-t{args.degree_step}_{str(args.random_seed).zfill(6)}.png')
    else:
        filename = (f'{args.sample_path}/entangle-i{args.index}-d{args.degree}'
                    f'-t{args.degree_step}-l{args.layer_num}'
                    f'_{str(args.random_seed).zfill(6)}.png')

    img = torch.cat(img_list, dim=0)[:, [2, 1, 0]]
    utils.save_image(
        img,
        filename,
        nrow=args.num_samples,
        padding=0,
        normalize=True,
        range=(-1, 1))

    mmcv.print_log(f'Save images to {filename}', 'mmgen')
