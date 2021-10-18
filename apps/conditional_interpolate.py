import argparse
import os
import sys

import mmcv
import torch
import torch.nn as nn
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from torchvision.utils import save_image

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import set_random_seed # isort:skip  # noqa
from mmgen.core.evaluation import slerp # isort:skip  # noqa
from mmgen.models import build_model # isort:skip  # noqa
from mmgen.models.architectures import BigGANDeepGenerator, BigGANGenerator # isort:skip  # noqa
from mmgen.models.architectures.common import get_module_device # isort:skip  # noqa

# yapf: enable

_default_embedding_name = dict(
    BigGANGenerator='shared_embedding',
    BigGANDeepGenerator='shared_embedding',
    SNGANGenerator='NULL',
    SAGANGenerator='NULL')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sampling from latents\' interpolation')
    parser.add_argument('config', help='evaluation config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--use-cpu',
        action='store_true',
        help='whether to use cpu device for sampling')
    parser.add_argument(
        '--embedding-name',
        type=str,
        default=None,
        help='name of conditional model\'s embedding layer')
    parser.add_argument(
        '--fix-z',
        action='store_true',
        help='whether to fix the noise for conditional model')
    parser.add_argument(
        '--fix-y',
        action='store_true',
        help='whether to fix the label for conditional model')
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
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')
    args = parser.parse_args()
    return args


@torch.no_grad()
def batch_inference(generator,
                    noise,
                    embedding=None,
                    num_batches=-1,
                    max_batch_size=16,
                    dict_key=None,
                    **kwargs):
    """Inference function to get a batch of desired data from output dictionary
    of generator.

    Args:
        generator (nn.Module): Generator of a conditional model.
        noise (Tensor | list[torch.tensor] | None): A batch of noise
            Tensor.
        embedding (Tensor, optional): Embedding tensor of label for
            conditional models. Defaults to None.
        num_batches (int, optional): The number of batchs for
            inference. Defaults to -1.
        max_batch_size (int, optional): The number of batch size for
            inference. Defaults to 16.
        dict_key (str, optional): key used to get results from output
            dictionary of generator. Defaults to None.

    Returns:
        torch.Tensor: Tensor of output image, noise batch or label
            batch.
    """
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

    # split embedding into groups
    if embedding is not None:
        assert isinstance(embedding, torch.Tensor)
        num_batches = embedding.shape[0]
        embedding_group = torch.split(embedding, max_batch_size, 0)
    else:
        embedding_group = [None] * (
            num_batches // max_batch_size +
            (1 if num_batches % max_batch_size > 0 else 0))

    # split batchsize into groups
    batchsize_group = [max_batch_size] * (num_batches // max_batch_size)
    if num_batches % max_batch_size > 0:
        batchsize_group += [num_batches % max_batch_size]

    device = get_module_device(generator)
    outputs = []
    for _noise, _embedding, _num_batches in zip(noise_group, embedding_group,
                                                batchsize_group):
        if isinstance(_noise, torch.Tensor):
            _noise = _noise.to(device)
        if isinstance(_noise, list):
            _noise = [ele.to(device) for ele in _noise]
        if _embedding is not None:
            _embedding = _embedding.to(device)
        output = generator(
            _noise, label=_embedding, num_batches=_num_batches, **kwargs)
        output = output[dict_key] if dict_key else output
        if isinstance(output, list):
            output = output[0]
        # once obtaining sampled results, we immediately put them into cpu
        # to save cuda memory
        outputs.append(output.to('cpu'))
    outputs = torch.cat(outputs, dim=0)
    return outputs


@torch.no_grad()
def sample_from_path(generator,
                     latent_a,
                     latent_b,
                     label_a,
                     label_b,
                     intervals,
                     embedding_name=None,
                     interp_mode='lerp',
                     **kwargs):
    interp_alphas = torch.linspace(0, 1, intervals)
    interp_samples = []

    device = get_module_device(generator)
    if embedding_name is None:
        generator_name = generator.__class__.__name__
        assert generator_name in _default_embedding_name
        embedding_name = _default_embedding_name[generator_name]
    embedding_fn = getattr(generator, embedding_name, nn.Identity())
    embedding_a = embedding_fn(label_a.to(device))
    embedding_b = embedding_fn(label_b.to(device))

    for alpha in interp_alphas:
        # calculate latent interpolation
        if interp_mode == 'lerp':
            latent_interp = torch.lerp(latent_a, latent_b, alpha)
        else:
            assert latent_a.ndim == latent_b.ndim == 2
            latent_interp = slerp(latent_a, latent_b, alpha)

        # calculate embedding interpolation
        embedding_interp = embedding_a + (
            embedding_b - embedding_a) * alpha.to(embedding_a.dtype)
        if isinstance(generator, (BigGANDeepGenerator, BigGANGenerator)):
            kwargs.update(dict(use_outside_embedding=True))
        sample = batch_inference(generator, latent_interp, embedding_interp,
                                 **kwargs)
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
        dict_key='noise_batch',
        return_noise=True,
        **kwargs)

    # get labels corresponding to each endpoint
    label_batch = batch_inference(
        generator,
        None,
        num_batches=args.endpoint,
        dict_key='label',
        return_noise=True,
        **kwargs)
    # set label fixed
    if args.fix_y:
        label_batch = label_batch[0] * torch.ones_like(label_batch)
    # set noise fixed
    if args.fix_z:
        noise_batch = torch.cat(
            [noise_batch[0:1, ]] * noise_batch.shape[0], dim=0)

    if args.show_mode == 'sequence':
        results = sample_from_path(generator, noise_batch[:-1, ],
                                   noise_batch[1:, ], label_batch[:-1, ],
                                   label_batch[1:, ], args.interval,
                                   args.embedding_name, args.interp_mode,
                                   **kwargs)
    else:
        results = sample_from_path(generator, noise_batch[::2, ],
                                   noise_batch[1::2, ], label_batch[:-1, ],
                                   label_batch[1:, ], args.interval,
                                   args.embedding_name, args.interp_mode,
                                   **kwargs)
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
