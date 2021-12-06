import argparse
import os
import sys

import imageio
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from torchvision.utils import save_image

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import set_random_seed # isort:skip  # noqa
from mmgen.core.evaluation import slerp # isort:skip  # noqa
from mmgen.models import build_model # isort:skip  # noqa

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
    parser.add_argument(
        '--export-video',
        action='store_true',
        help='If true, export video rather than images')
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
        '--proj-latent',
        type=str,
        default=None,
        help='Projection image files produced by stylegan_projector.py. If this \
        argument is given, then the projected latent will be used as the input\
        noise.')
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
        default='w',
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
    """Inference function to get a batch of desired data from output dictionary
    of generator.

    Args:
        generator (nn.Module): Generator of a conditional model.
        noise (Tensor | list[torch.tensor] | None): A batch of noise
            Tensor.
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
        # once obtaining sampled results, we immediately put them into cpu
        # to save cuda memory
        outputs.append(output.to('cpu'))
    outputs = torch.cat(outputs, dim=0)
    return outputs


def layout_grid(video_out,
                all_img,
                grid_w=1,
                grid_h=1,
                float_to_uint8=True,
                chw_to_hwc=True,
                to_numpy=True):
    r"""Arrange images into video frames.

        Ref: https://github.com/NVlabs/stylegan3/blob/a5a69f58294509598714d1e88c9646c3d7c6ec94/gen_video.py#L28 # noqa

    Args:
        video_out (Writer): Video writer.
        all_img (torch.Tensor): All images to be displayed in video.
        grid_w (int, optional): Column number in a frame. Defaults to 1.
        grid_h (int, optional): Row number in a frame. Defaults to 1.
        float_to_uint8 (bool, optional): Change torch value from `float` to `uint8`. Defaults to True.
        chw_to_hwc (bool, optional): Change channel order from `chw` to `hwc`. Defaults to True.
        to_numpy (bool, optional): Change image format from `torch.Tensor` to `np.array`. Defaults to True.

    Returns:
        Writer: Video writer.
    """
    batch_size, channels, img_h, img_w = all_img.shape
    assert batch_size % (grid_w * grid_h) == 0
    images_per_frame = grid_w * grid_h
    n_frames = batch_size // images_per_frame
    all_img = all_img.reshape(images_per_frame, n_frames, channels, img_h,
                              img_w).permute(1, 0, 2, 3, 4).reshape(
                                  n_frames, images_per_frame, channels, img_h,
                                  img_w)
    for i in range(0, n_frames):
        img = all_img[i]
        if float_to_uint8:
            img = (img * 255.).clamp(0, 255).to(torch.uint8)
        img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
        img = img.permute(2, 0, 3, 1, 4)
        img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
        if chw_to_hwc:
            img = img.permute(1, 2, 0)
        if to_numpy:
            img = img.cpu().numpy()
        video_out.append_data(img)
    return video_out


def crack_integer(integer):
    """Cracking an integer into the product of two nearest integers.

    Args:
        integer (int): An positive integer.

    Returns:
        tuple: Two integers.
    """
    start = int(np.sqrt(integer))
    factor = integer / start
    while int(factor) != factor:
        start += 1
        factor = integer / start
    return int(factor), start


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

    # if given proj_latent, reset args.endpoint
    if args.proj_latent is not None:
        mmcv.print_log(f'Load projected latent: {args.proj_latent}', 'mmgen')
        proj_file = torch.load(args.proj_latent)
        proj_n = len(proj_file)
        setattr(args, 'endpoint', proj_n)
        assert args.space == 'w', 'Projected latent are w or w-plus latent.'
        noise_batch = []
        for img_path in proj_file:
            noise_batch.append(proj_file[img_path]['latent'].unsqueeze(0))
        noise_batch = torch.cat(noise_batch, dim=0).cuda()
        if args.use_cpu:
            noise_batch = noise_batch.to('cpu')

    if args.show_mode == 'sequence':
        assert args.endpoint >= 2
    else:
        assert args.endpoint >= 2 and args.endpoint % 2 == 0,\
            '''We need paired images in group mode,
            so keep endpoint an even number'''

    kwargs = dict(max_batch_size=args.batch_size)
    if args.sample_cfg is None:
        args.sample_cfg = dict()
    kwargs.update(args.sample_cfg)
    # remind users to fixed injected noise
    if kwargs.get('randomize_noise', 'True'):
        mmcv.print_log(
            '''Hint: For Style-Based GAN, you can add
            `--sample-cfg randomize_noise=False` to fix injected noises''',
            'mmgen')

    # get noises corresponding to each endpoint
    if not args.proj_latent:
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
        if args.export_video:
            # render video.
            video_out = imageio.get_writer(
                os.path.join(args.samples_path, 'lerp.mp4'),
                mode='I',
                fps=60,
                codec='libx264',
                bitrate='12M')
            video_out = layout_grid(video_out, results)
            video_out.close()
        else:
            for i in range(results.shape[0]):
                image = results[i:i + 1]
                save_image(
                    image,
                    os.path.join(args.samples_path,
                                 '{:0>5d}'.format(i) + '.png'))
    else:
        if args.export_video:
            # render video.
            video_out = imageio.get_writer(
                os.path.join(args.samples_path, 'lerp.mp4'),
                mode='I',
                fps=60,
                codec='libx264',
                bitrate='12M')
            n_pair = args.endpoint // 2
            grid_w, grid_h = crack_integer(n_pair)
            video_out = layout_grid(
                video_out, results, grid_h=grid_h, grid_w=grid_w)
            video_out.close()
        else:
            save_image(
                results,
                os.path.join(args.samples_path, 'group.png'),
                nrow=args.interval)


if __name__ == '__main__':
    main()
