import argparse
import os
import sys

import imageio
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint

from mmgen.core.evaluation.metric_utils import rotation_matrix

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import set_random_seed # isort:skip  # noqa
from mmgen.core.evaluation import slerp # isort:skip  # noqa
from mmgen.models import build_model # isort:skip  # noqa
from mmgen.models.architectures.common import get_module_device # isort:skip  # noqa
from mmgen.models.architectures import StyleGANv3Generator # isort:skip  # noqa
# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sampling from latents\' interpolation')
    parser.add_argument('config', help='evaluation config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--use-cpu',
        action='store_true',
        help='whether to use cpu device for sampling')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--samples-path',
        type=str,
        default='work_dirs/viz',
        help='path to save video')
    parser.add_argument(
        '--sample-model',
        type=str,
        default='ema',
        help='use which mode (ema/orig) in sampling.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='batch size used in generator sampling.')
    parser.add_argument(
        '--translate_max',
        type=float,
        default=0.125,
        help='The translate ratio of width/height/angle. Range: (0,1].')
    parser.add_argument(
        '--transform',
        choices=['x_t', 'y_t', 'rotate'],
        default='x_t',
        help='Interpolation space.')
    parser.add_argument(
        '--num-samples', type=int, default=60, help='The number of samplings.')
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')
    args = parser.parse_args()
    return args


@torch.no_grad()
def sampling_images(generator, interval, translate_max=0.125, eq_type='x_t'):
    device = get_module_device(generator)

    identity_matrix = torch.eye(3, device=device)
    transform_matrix = getattr(
        getattr(getattr(generator, 'synthesis', None), 'input', None),
        'transform', None)

    z = torch.randn([1, generator.noise_size], device=device)
    ws = generator.style_mapping(z=z)

    results = []
    ts = torch.linspace(-1, 1, interval)
    for t in ts:
        t = t * translate_max
        t = (t * generator.out_size).round() / generator.out_size

        transform_matrix[:] = identity_matrix
        if eq_type == 'x_t':
            transform_matrix[0, 2] = -t
        elif eq_type == 'y_t':
            transform_matrix[1, 2] = -t
        else:
            angle = t * (1 * np.pi)
            transform_matrix[:] = rotation_matrix(-angle)

        img = generator.synthesis(ws=ws)
        results.append(img.cpu())
        del img
        torch.cuda.empty_cache()

    return results


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # set random seeds
    if args.seed is not None:
        print('set random seed to', args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # sanity check for models without ema
    if not model.use_ema:
        args.sample_model = 'orig'
    if args.sample_model == 'ema':
        generator = model.generator_ema
    else:
        generator = model.generator

    assert isinstance(generator, StyleGANv3Generator)

    mmcv.print_log(f'Sampling model: {args.sample_model}', 'mmgen')
    mmcv.print_log(f'Samples path: {args.samples_path}', 'mmgen')

    generator.eval()

    if not args.use_cpu:
        generator = generator.cuda()

    results = sampling_images(
        generator,
        args.num_samples,
        eq_type=args.transform,
        translate_max=args.translate_max)

    # reorder results
    results = torch.cat(results, dim=0)
    results = ((results + 1) / 2)
    mmcv.mkdir_or_exist(args.samples_path)
    # render video.
    video_frames = []
    for result in results:
        img = result.permute(1, 2, 0)
        img = (img * 255.).clamp(0, 255).to(torch.uint8)
        img = img.cpu().numpy()
        video_frames.append(img)
    imageio.mimsave(
        os.path.join(args.samples_path,
                     f'eq-{args.transform}-seed-{args.seed}.mp4'),
        video_frames,
        fps=40)


if __name__ == '__main__':
    main()
