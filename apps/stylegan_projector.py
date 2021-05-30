r"""
    This app is used to invert the styleGAN series synthesis network. We find
    the matching latent vector w for given images so that we can manipulate
    images in the latent feature space.
    Ref: https://github.com/rosinality/stylegan2-pytorch/blob/master/projector.py # noqa
"""
import argparse
import os

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from PIL import Image
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from mmgen.apis import set_random_seed
from mmgen.models import build_model
from mmgen.models.architectures.lpips import PerceptualLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description='Image projector to the StyleGAN-based generator latent \
            spaces')
    parser.add_argument('config', help='evaluation config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'files',
        metavar='FILES',
        nargs='+',
        help='path to image files to be projected')
    parser.add_argument(
        '--results-path', type=str, help='path to store projection results.')
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
        '--sample-model',
        type=str,
        default='ema',
        help='use which mode (ema/orig) in sampling.')
    parser.add_argument(
        '--lr-rampup',
        type=float,
        default=0.05,
        help='proportion of the learning rate warmup iters in the total iters')
    parser.add_argument(
        '--lr-rampdown',
        type=float,
        default=0.25,
        help='proportion of the learning rate decay iters in the total iters')
    parser.add_argument(
        '--lr', type=float, default=0.1, help='maximum learning rate')
    parser.add_argument(
        '--noise',
        type=float,
        default=0.05,
        help='strength of the noise level')
    parser.add_argument(
        '--noise-ramp',
        type=float,
        default=0.75,
        help='proportion of the noise level decay iters in the total iters',
    )
    parser.add_argument(
        '--total-iters', type=int, default=1000, help='optimize iterations')
    parser.add_argument(
        '--noise-regularize',
        type=float,
        default=1e5,
        help='weight of the noise regularization',
    )
    parser.add_argument(
        '--mse', type=float, default=0, help='weight of the mse loss')
    parser.add_argument(
        '--n-mean-latent',
        type=int,
        default=10000,
        help='sampling times to obtain the mean latent')
    parser.add_argument(
        '--w-plus',
        action='store_true',
        help='allow to use distinct latent codes to each layers',
    )
    args = parser.parse_args()
    return args


def noise_regularize(noises):
    loss = 0
    for noise in noises:
        size = noise.shape[2]
        while True:
            loss = (
                loss +
                (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2) +
                (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2))
            if size <= 8:
                break
            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2
    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength
    return latent + noise


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

    generator.eval()
    device = 'cpu'
    if not args.use_cpu:
        generator = generator.cuda()
        device = 'cuda'

    img_size = min(generator.out_size, 256)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # read images
    imgs = []
    for imgfile in args.files:
        img = Image.open(imgfile).convert('RGB')
        img = transform(img)
        img = img[[2, 1, 0], ...]
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    # get mean and standard deviation of style latents
    with torch.no_grad():
        noise_sample = torch.randn(
            args.n_mean_latent, generator.style_channels, device=device)
        latent_out = generator.style_mapping(noise_sample)
        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() /
                      args.n_mean_latent)**0.5
    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(
        imgs.shape[0], 1)
    if args.w_plus:
        latent_in = latent_in.unsqueeze(1).repeat(1, generator.num_latents, 1)
    latent_in.requires_grad = True

    # define lpips loss
    percept = PerceptualLoss(use_gpu=device.startswith('cuda'))

    # initialize layer noises
    noises_single = generator.make_injected_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())
    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)
    pbar = tqdm(range(args.total_iters))
    # run optimization
    for i in pbar:
        t = i / args.total_iters
        lr = get_lr(t, args.lr, args.lr_rampdown, args.lr_rampup)
        optimizer.param_groups[0]['lr'] = lr
        noise_strength = latent_std * args.noise * max(
            0, 1 - t / args.noise_ramp)**2
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen = generator([latent_n],
                            input_is_latent=True,
                            injected_noise=noises)

        batch, channel, height, width = img_gen.shape

        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(batch, channel, height // factor, factor,
                                      width // factor, factor)
            img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen, imgs).sum()
        n_loss = noise_regularize(noises)
        mse_loss = F.mse_loss(img_gen, imgs)

        loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        pbar.set_description(
            f' perceptual: {p_loss.item():.4f}, noise regularize:'
            f'{n_loss.item():.4f}, mse: {mse_loss.item():.4f}, lr: {lr:.4f}')

    results = generator([latent_in.detach().clone()],
                        input_is_latent=True,
                        injected_noise=noises)
    # rescale value range to [0, 1]
    results = ((results + 1) / 2)
    results = results[:, [2, 1, 0], ...]
    results = results.clamp_(0, 1)

    mmcv.mkdir_or_exist(args.results_path)
    # save projection results
    result_file = {}
    for i, input_name in enumerate(args.files):
        noise_single = []
        for noise in noises:
            noise_single.append(noise[i:i + 1])
        result_file[input_name] = {
            'img': img_gen[i],
            'latent': latent_in[i],
            'injected_noise': noise_single,
        }
        img_name = os.path.splitext(
            os.path.basename(input_name))[0] + '-project.png'
        save_image(results[i], os.path.join(args.results_path, img_name))

    torch.save(result_file, os.path.join(args.results_path,
                                         'project_result.pt'))


if __name__ == '__main__':
    main()
