import argparse
import math
import os

try:
    import clip
except ImportError:
    raise 'To use styleclip, openai clip need to be installed first'
import mmcv
import torch
import torchvision
from mmcv import Config, DictAction
from torch import optim
from tqdm import tqdm

from mmgen.apis import init_model
from mmgen.models.losses import CLIPLoss, FaceIdLoss

from mmgen.apis import set_random_seed  # isort:skip  # noqa


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='model config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--use-cpu',
        action='store_true',
        help='whether to use cpu device for sampling')
    parser.add_argument(
        '--description',
        type=str,
        default='a person with purple hair',
        help='the text that guides the editing/generation')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument(
        '--mode',
        type=str,
        default='generate',
        choices=['edit', 'generate'],
        help='choose between edit an image an generate a free one')
    parser.add_argument(
        '--l2-lambda',
        type=float,
        default=0.008,
        help='weight of the latent distance, used for editing only')
    parser.add_argument(
        '--id-lambda',
        type=float,
        default=0.000,
        help='weight of id loss, used for editing only')
    parser.add_argument(
        '--proj-latent',
        type=str,
        default=None,
        help='Projection image files produced by stylegan_projector.py. If this \
        argument is given, then the projected latent will be used as the init\
        latent.')
    parser.add_argument(
        '--truncation',
        type=float,
        default=0.7,
        help='used only for the initial latent vector, and only when a latent '
        'code path is not provided')
    parser.add_argument(
        '--step', type=int, default=2000, help='Optimization iterations')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=20,
        help='if > 0 then saves intermidate results during the optimization')
    parser.add_argument(
        '--results-dir', type=str, default='work_dirs/styleclip/')
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # set cudnn_benchmark
    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # set random seeds
    if args.seed is not None:
        print('set random seed to', args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    os.makedirs(args.results_dir, exist_ok=True)

    text_inputs = torch.cat([clip.tokenize(args.description)]).cuda()

    model = init_model(args.config, args.checkpoint, device='cpu')
    g_ema = model.generator_ema
    g_ema.eval()
    if not args.use_cpu:
        g_ema = g_ema.cuda()

    mean_latent = g_ema.get_mean_latent()

    # if given proj_latent
    if args.proj_latent is not None:
        mmcv.print_log(f'Load projected latent: {args.proj_latent}', 'mmgen')
        proj_file = torch.load(args.proj_latent)
        proj_n = len(proj_file)
        assert proj_n == 1
        noise_batch = []
        for img_path in proj_file:
            noise_batch.append(proj_file[img_path]['latent'].unsqueeze(0))
        latent_code_init = torch.cat(noise_batch, dim=0).cuda()
    elif args.mode == 'edit':
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            results = g_ema([latent_code_init_not_trunc],
                            return_latents=True,
                            truncation=args.truncation,
                            truncation_latent=mean_latent)
            latent_code_init = results['latent']
    else:
        latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

    with torch.no_grad():
        img_orig = g_ema([latent_code_init],
                         input_is_latent=True,
                         randomize_noise=False)

    latent = latent_code_init.detach().clone()
    latent.requires_grad = True

    clip_loss = CLIPLoss(clip_model=dict(in_size=g_ema.out_size))
    id_loss = FaceIdLoss(
        facenet=dict(type='ArcFace', ir_se50_weights=None, device='cuda'))

    optimizer = optim.Adam([latent], lr=args.lr)

    pbar = tqdm(range(args.step))
    mmcv.print_log(f'Description: {args.description}')
    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]['lr'] = lr

        img_gen = g_ema([latent], input_is_latent=True, randomize_noise=False)

        img_gen = img_gen[:, [2, 1, 0], ...]

        # clip loss
        c_loss = clip_loss(image=img_gen, text=text_inputs)

        if args.id_lambda > 0:
            i_loss = id_loss(pred=img_gen, gt=img_orig)[0]
        else:
            i_loss = 0

        if args.mode == 'edit':
            l2_loss = ((latent_code_init - latent)**2).sum()
            loss = c_loss + args.l2_lambda * l2_loss + args.id_lambda * i_loss
        else:
            loss = c_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description((f'loss: {loss.item():.4f};'))
        if args.save_interval > 0 and (i % args.save_interval == 0):
            with torch.no_grad():
                img_gen = g_ema([latent],
                                input_is_latent=True,
                                randomize_noise=False)

            img_gen = img_gen[:, [2, 1, 0], ...]

            torchvision.utils.save_image(
                img_gen,
                os.path.join(args.results_dir, f'{str(i).zfill(5)}.png'),
                normalize=True,
                range=(-1, 1))

    if args.mode == 'edit':
        img_orig = img_orig[:, [2, 1, 0], ...]
        final_result = torch.cat([img_orig, img_gen])
    else:
        final_result = img_gen

    torchvision.utils.save_image(
        final_result.detach().cpu(),
        os.path.join(args.results_dir, 'final_result.png'),
        normalize=True,
        scale_each=True,
        range=(-1, 1))


if __name__ == '__main__':
    main()
