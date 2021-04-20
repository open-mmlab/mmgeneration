import argparse
import os

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from torchvision.utils import save_image

from mmgen.apis import set_random_seed
from mmgen.core.evaluation import slerp
from mmgen.models import build_model


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
        '--samples-path', type=str, help='path to store images.')
    parser.add_argument(
        '--sample-model',
        type=str,
        default='ema',
        help='use which mode (ema/orig) in sampling.')
    parser.add_argument(
        '--show-mode',
        choices=['group', 'sequence'],
        default='sequence',
        help='mode to show interpolation result.')
    parser.add_argument(
        '--interp-mode',
        choices=['lerp', 'slerp'],
        default='lerp',
        help='mode to sample from endpoints\'s interpolation.')
    parser.add_argument(
        '--endpoint', type=int, default=2, help='The number of endpoints.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='batch size used in generator sampling.')
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='The number of intervals between two endpoints.')
    parser.add_argument(
        '--space',
        choices=['z', 'w'],
        default='z',
        help='Interpolation space.')
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')
    args = parser.parse_args()
    return args


@torch.no_grad()
def batch_inference(generator,
                    noise,
                    num_batches=-1,
                    max_batch_size=16,
                    dict_key=None,
                    **kwargs):
    # split noise into groups
    if noise is not None:
        if isinstance(noise, torch.Tensor):
            num_batches = noise.shape[0]
            noise_group = torch.split(noise, max_batch_size, 0)
        else:
            num_batches = noise[0].shape[0]
            noise_group = torch.split(noise[0], max_batch_size, 0)
            noise_group = [[noise_tensor] for noise_tensor in noise_group]
    else:
        noise_group = [None] * (
            num_batches // max_batch_size +
            (1 if num_batches % max_batch_size > 0 else 0))
    # split batchsize into groups
    batchsize_group = [max_batch_size] * (num_batches // max_batch_size)
    if num_batches % max_batch_size > 0:
        batchsize_group += [num_batches % max_batch_size]
    outputs = []
    for _noise, _num_batches in zip(noise_group, batchsize_group):
        if isinstance(_noise, torch.Tensor):
            _noise = _noise.cuda()
        if isinstance(_noise, list):
            _noise = [ele.cuda() for ele in _noise]
        output = generator(_noise, num_batches=_num_batches, **kwargs)
        output = output[dict_key] if dict_key else output
        if isinstance(output, list):
            output = output[0]
        # once we get sampling results, immediately put them into cpu to save
        # cuda memory
        outputs.append(output.to('cpu'))
    outputs = torch.cat(outputs, dim=0)
    return outputs


@torch.no_grad()
def sample_from_path(generator,
                     latent_a,
                     latent_b,
                     intervals,
                     interp_mode='lerp',
                     space='z',
                     **kwargs):
    interp_alphas = np.linspace(0, 1, intervals)
    interp_samples = []

    for alpha in interp_alphas:
        if interp_mode == 'lerp':
            latent_interp = torch.lerp(latent_a, latent_b, alpha)
        else:
            assert latent_a.ndim == latent_b.ndim == 2
            latent_interp = slerp(latent_a, latent_b, alpha)
        if space == 'w':
            latent_interp = [latent_interp]
        sample = batch_inference(generator, latent_interp, **kwargs)
        interp_samples.append(sample)

    return interp_samples


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
    mmcv.print_log(f'Sampling model: {args.sample_model}', 'mmgen')
    mmcv.print_log(f'Show mode: {args.show_mode}', 'mmgen')
    mmcv.print_log(f'Samples path: {args.samples_path}', 'mmgen')

    generator.eval()

    if not args.use_cpu:
        generator = generator.cuda()
    if args.show_mode == 'sequence':
        assert args.endpoint >= 2
    else:
        assert args.endpoint >= 2 and args.endpoint % 2 == 0

    kwargs = dict(max_batch_size=args.batch_size)
    if args.sample_cfg is None:
        args.sample_cfg = dict()
    kwargs.update(args.sample_cfg)

    # get noises corresponding to each endpoint
    noise_batch = batch_inference(
        generator,
        None,
        num_batches=args.endpoint,
        dict_key='noise_batch' if args.space == 'z' else 'latent',
        return_noise=True,
        **kwargs)

    if args.space == 'w':
        kwargs['truncation_latent'] = generator.get_mean_latent()
        kwargs['input_is_latent'] = True

    if args.show_mode == 'sequence':
        results = sample_from_path(generator, noise_batch[:-1, ],
                                   noise_batch[1:, ], args.interval,
                                   args.interp_mode, args.space, **kwargs)
    else:
        results = sample_from_path(generator, noise_batch[::2, ],
                                   noise_batch[1::2, ], args.interval,
                                   args.interp_mode, args.space, **kwargs)
    # reorder results
    results = torch.stack(results).permute(1, 0, 2, 3, 4)
    _, _, ch, h, w = results.shape
    results = results.reshape(-1, ch, h, w)
    # rescale value range to [0, 1]
    results = ((results + 1) / 2)
    results = results[:, [2, 1, 0], ...]
    results = results.clamp_(0, 1)
    # save image
    mmcv.mkdir_or_exist(args.samples_path)
    if args.show_mode == 'sequence':
        for i in range(results.shape[0]):
            image = results[i:i + 1]
            save_image(
                image,
                os.path.join(args.samples_path, '{:0>5d}'.format(i) + '.png'))
    else:
        save_image(
            results,
            os.path.join(args.samples_path, 'group.png'),
            nrow=args.interval)


if __name__ == '__main__':
    main()
