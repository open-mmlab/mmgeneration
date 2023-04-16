# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
import sys
from copy import deepcopy

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from prettytable import PrettyTable
from torchvision.utils import save_image

from mmgen.datasets import build_dataloader, build_dataset


def make_metrics_table(train_cfg, ckpt, eval_info, metrics):
    """Arrange evaluation results into a table.

    Args:
        train_cfg (str): Name of the training configuration.
        ckpt (str): Path of the evaluated model's weights.
        metrics (Metric): Metric objects.

    Returns:
        str: String of the eval table.
    """
    table = PrettyTable()
    table.set_style(14)
    table.add_column('Training configuration', [train_cfg])
    table.add_column('Checkpoint', [ckpt])
    table.add_column('Eval', [eval_info])
    for metric in metrics:
        table.add_column(metric.name, [metric.result_str])
    return table.get_string()


def make_vanilla_dataloader(img_path, batch_size, dist=False):
    pipeline = [
        dict(type='LoadImageFromFile', key='real_img', io_backend='disk'),
        dict(
            type='Normalize',
            keys=['real_img'],
            mean=[127.5] * 3,
            std=[127.5] * 3,
            to_rgb=False),
        dict(type='ImageToTensor', keys=['real_img']),
        dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
    ]
    dataset = build_dataset(
        dict(
            type='UnconditionalImageDataset',
            imgs_root=img_path,
            pipeline=pipeline,
        ))
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=4,
        dist=dist,
        shuffle=True)
    return dataloader


def make_npz_dataloader(npz_path, batch_size, dist=False):
    pipeline = [
        # permute the color channel. Because in npz_dataloader's pipeline, we
        # direct load image in RGB order from npz file and we must convert it
        # to BGR by setting ``to_rgb=True``.
        dict(
            type='Normalize',
            keys=['real_img'],
            mean=[127.5] * 3,
            std=[127.5] * 3,
            to_rgb=True),
        dict(type='ImageToTensor', keys=['real_img']),
        dict(type='Collect', keys=['real_img'])
    ]

    dataset = build_dataset(
        dict(type='FileDataset', file_path=npz_path, pipeline=pipeline))
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=0,
        dist=dist,
        shuffle=True)
    return dataloader


def parse_npz_file_name(npz_file):
    """Parse the basic information from the give npz file.
    Args:
        npz_file (str): The file name of the npz file.

    Returns:
        tuple(int): A tuple of (num_samples, H, W, num_channels).
    """
    # remove 'samples_' (8) and '.npz' (4)
    num_samples, H, W, num_channles = npz_file[8:-4].split('x')
    assert num_samples.isdigit()
    assert H.isdigit()
    assert W.isdigit()
    assert num_channles.isdigit()
    return int(num_samples), int(H), int(W), int(num_channles)


def parse_npz_folder(npz_folder_path):
    """Parse the npz files under the given folder.
    Args:
        npz_folder_path: The folder contains npz file.

    Returns:
        tuple(list, int): A tuple contains a list valid npz files' names and
            a int of existing image numbers.
    """

    npz_files = [
        f for f in list(mmcv.scandir(npz_folder_path, suffix=('.npz')))
        if 'samples' in f
    ]
    valid_npz_files = []
    img_shape = None
    num_exist = 0
    for npz_file in npz_files:
        try:
            n_samples, H, W, n_channels = parse_npz_file_name(npz_file)

            # shape checking
            if img_shape is None:
                img_shape = (H, W, n_channels)
            else:
                if img_shape != (H, W, n_channels):
                    raise ValueError(
                        'Image shape conflicting under sample path:'
                        f'\'{npz_folder_path}\'. Find {img_shape} vs. '
                        f'{(H, W, n_channels)}.')

            valid_npz_files.append(npz_file)
            num_exist += n_samples
        except AssertionError:
            mmcv.print_log(
                f'Find npz file \'{npz_file}\' does not conform to the '
                'standard naming convention.', 'mmgen')
    return valid_npz_files, num_exist


@torch.no_grad()
def offline_evaluation(model,
                       data_loader,
                       metrics,
                       logger,
                       basic_table_info,
                       batch_size,
                       samples_path=None,
                       save_npz=False,
                       **kwargs):
    """Evaluate model in offline mode.

    This method first save generated images at local and then load them by
    dataloader.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): PyTorch data loader.
        metrics (list): List of metric objects.
        logger (Logger): logger used to record results of evaluation.
        batch_size (int): Batch size of images fed into metrics.
        basic_table_info (dict): Dictionary containing the basic information \
            of the metric table include training configuration and ckpt.
        samples_path (str): Used to save generated images. If it's none, we'll
            give it a default directory and delete it after finishing the
            evaluation. Default to None.
        save_npz (bool, optional): Whether save the generated images to a npz
            file named 'samples_{NUM_IMAGES}x{H}x{W}x{NUM_CHANNELS}.npz' If
            true, dataset will be build upon npz file instead of image files.
            Defaults to True.
        kwargs (dict): Other arguments.
    """
    # eval special and recon metric online only
    online_metric_name = ['PPL', 'GaussianKLD']
    for metric in metrics:
        assert metric.name not in online_metric_name, 'Please eval '\
             f'{metric.name} online'

    rank, ws = get_dist_info()

    delete_samples_path = False
    if samples_path:
        mmcv.mkdir_or_exist(samples_path)
    else:
        temp_path = './work_dirs/temp_samples'
        # if temp_path exists, add suffix
        suffix = 1
        samples_path = temp_path
        while os.path.exists(samples_path):
            samples_path = temp_path + '_' + str(suffix)
            suffix += 1
        os.makedirs(samples_path)
        delete_samples_path = True

    # check existing images
    if save_npz:
        npz_file_exist, num_exist = parse_npz_folder(samples_path)
    else:
        num_exist = len(
            list(
                mmcv.scandir(
                    samples_path, suffix=('.jpg', '.png', '.jpeg', '.JPEG'))))
    if basic_table_info['num_samples'] > 0:
        max_num_images = basic_table_info['num_samples']
    else:
        max_num_images = max(metric.num_images for metric in metrics)
    num_needed = max(max_num_images - num_exist, 0)

    if num_needed > 0 and rank == 0:
        mmcv.print_log(f'Sample {num_needed} fake images for evaluation',
                       'mmgen')
        # define mmcv progress bar
        pbar = mmcv.ProgressBar(num_needed)

    fake_img_list = []
    # if no images, `num_needed` should be zero
    total_batch_size = batch_size * ws
    for begin in range(0, num_needed, total_batch_size):
        end = min(begin + batch_size, max_num_images)
        fakes = model(
            None,
            num_batches=end - begin,
            return_loss=False,
            sample_model=basic_table_info['sample_model'],
            **kwargs)
        global_end = min(begin + total_batch_size, max_num_images)
        if rank == 0:
            pbar.update(global_end - begin)

        # gather generated images
        if ws > 1:
            placeholder = [torch.zeros_like(fakes) for _ in range(ws)]
            dist.all_gather(placeholder, fakes)
            fakes = torch.cat(placeholder, dim=0)

        # save as three-channel
        if fakes.size(1) == 3:
            fakes = fakes[:, [2, 1, 0], ...]
        elif fakes.size(1) == 1:
            fakes = torch.cat([fakes] * 3, dim=1)
        else:
            raise RuntimeError('Generated images must have one or three '
                               'channels in the first dimension, '
                               'not %d' % fakes.size(1))

        if rank == 0:
            for i in range(global_end - begin):
                images = fakes[i:i + 1]
                images = ((images + 1) / 2)
                images = images.clamp_(0, 1)
                if save_npz:
                    # permute to [H, W, chn] and rescale to [0, 255]
                    fake_img_list.append(
                        images.permute(0, 2, 3, 1).cpu().numpy() * 255)
                else:
                    image_name = str(num_exist + begin + i) + '.png'
                    save_image(images, os.path.join(samples_path, image_name))

    if save_npz:
        # only one npz file and fake_img_list is empty --> do not need to save
        if len(npz_file_exist) == 1 and len(fake_img_list) == 0:
            npz_path = os.path.join(samples_path, npz_file_exist[0])
            if rank == 0:
                mmcv.print_log(
                    f'Existing npz file \'{npz_path}\' has already met '
                    'requirements.', 'mmgen')
        else:
            # load from locl file and merge to one
            if rank == 0:
                fake_img_exist_list = []
                for exist_npz_file in npz_file_exist:
                    fake_imgs_ = np.load(
                        os.path.join(samples_path, exist_npz_file))['real_img']
                    fake_img_exist_list.append(fake_imgs_)

                # merge fake_img_exist_list and fake_img_list
                fake_imgs = np.concatenate(
                    fake_img_exist_list + fake_img_list, axis=0)
                num_imgs, H, W, num_channels = fake_imgs.shape

                npz_path = os.path.join(
                    samples_path,
                    f'samples_{num_imgs}x{H}x{W}x{num_channels}.npz')

                # save new npz file -->
                # set key as ``real_img`` to align with vanilla dataset
                np.savez(npz_path, real_img=fake_imgs)
                mmcv.print_log(f'Save new npz_file to \'{npz_path}\'.',
                               'mmgen')

                # delete old npz files
                for npz_file in npz_file_exist:
                    os.remove(os.path.join(samples_path, npz_file))
                    mmcv.print_log(
                        'Remove useless npz file '
                        f'\'{os.path.join(samples_path, npz_file)}\'.',
                        'mmgen')

            # waiting for rank-0 to save the new npz file
            if ws > 1:
                dist.barrier()
            # get npz_path.
            # We have delete useless npz files then there should only one
            # file under the sample_path. Check and directly load it!
            npz_files = [
                f for f in list(mmcv.scandir(samples_path, suffix=('.npz')))
                if 'samples' in f
            ]
            assert len(npz_files) == 1
            npz_path = os.path.join(samples_path, npz_files[0])

    if num_needed > 0 and rank == 0:
        sys.stdout.write('\n')

    # return if only save sampled images
    if len(metrics) == 0:
        return

    # empty cache to release GPU memory
    torch.cuda.empty_cache()
    if save_npz:
        fake_dataloader = make_npz_dataloader(
            npz_path, batch_size, dist=ws > 1)
    else:
        fake_dataloader = make_vanilla_dataloader(
            samples_path, batch_size, dist=ws > 1)

    for metric in metrics:
        mmcv.print_log(f'Evaluate with {metric.name} metric.', 'mmgen')
        metric.prepare()
        if rank == 0:
            # prepare for pbar
            total_need = (
                metric.num_real_need + metric.num_fake_need -
                metric.num_real_feeded - metric.num_fake_feeded)
            pbar = mmcv.ProgressBar(total_need)
        # feed in real images
        for data in data_loader:
            # key for unconditional GAN
            if 'real_img' in data:
                reals = data['real_img']
            # key for conditional GAN
            elif 'img' in data:
                reals = data['img']
            else:
                raise KeyError('Cannot found key for images in data_dict. '
                               'Only support `real_img` for unconditional '
                               'datasets and `img` for conditional '
                               'datasets.')

            if reals.shape[1] == 1:
                reals = torch.cat([reals] * 3, dim=1)
            num_left = metric.feed(reals, 'reals')
            if num_left <= 0:
                break
            if rank == 0:
                pbar.update(reals.shape[0] * ws)
        # feed in fake images
        for data in fake_dataloader:
            fakes = data['real_img']
            if fakes.shape[1] == 1:
                fakes = torch.cat([fakes] * 3, dim=1)
            num_left = metric.feed(fakes, 'fakes')
            if num_left <= 0:
                break
            if rank == 0:
                pbar.update(fakes.shape[0] * ws)
        if rank == 0:
            # only call summary at main device
            metric.summary()
            sys.stdout.write('\n')
    if rank == 0:
        table_str = make_metrics_table(basic_table_info['train_cfg'],
                                       basic_table_info['ckpt'],
                                       basic_table_info['sample_model'],
                                       metrics)
        logger.info('\n' + table_str)
        if delete_samples_path:
            shutil.rmtree(samples_path)


@torch.no_grad()
def online_evaluation(model, data_loader, metrics, logger, basic_table_info,
                      batch_size, **kwargs):
    """Evaluate model in online mode.

    This method evaluate model and displays eval progress bar.
    Different form `offline_evaluation`, this function will not save
    the images or read images from disks. Namely, there do not exist any IO
    operations in this function. Thus, in general, `online` mode will achieve a
    faster evaluation. However, this mode will take much more memory cost.
    To be noted that, we only support distributed evaluation for FID and IS
    currently.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): PyTorch data loader.
        metrics (list): List of metric objects.
        logger (Logger): logger used to record results of evaluation.
        batch_size (int): Batch size of images fed into metrics.
        basic_table_info (dict): Dictionary containing the basic information \
            of the metric table include training configuration and ckpt.
        kwargs (dict): Other arguments.
    """
    # separate metrics into special metrics, probabilistic metrics and vanilla
    # metrics.
    # For vanilla metrics, images are generated in a random way, and are
    # shared by these metrics. For special metrics like 'PPL', images are
    # generated in a metric-special way and not shared between different
    # metrics.
    # For reconstruction metrics like 'GaussianKLD', they do not
    # receive images but receive a dict with corresponding probabilistic
    # parameter.

    rank, ws = get_dist_info()

    special_metrics = []
    recon_metrics = []
    vanilla_metrics = []
    special_metric_name = ['PPL']
    recon_metric_name = ['GaussianKLD']
    for metric in metrics:
        if ws > 1:
            assert metric.name in [
                'FID', 'IS'
            ], ('We only support FID and IS for distributed evaluation '
                f'currently, but receive {metric.name}')

        if metric.name in special_metric_name:
            special_metrics.append(metric)
        elif metric.name in recon_metric_name:
            recon_metrics.append(metric)
        else:
            vanilla_metrics.append(metric)

    # define mmcv progress bar
    max_num_images = 0
    for metric in vanilla_metrics + recon_metrics:
        metric.prepare()
        max_num_images = max(max_num_images,
                             metric.num_real_need - metric.num_real_feeded)
    if rank == 0:
        mmcv.print_log(f'Sample {max_num_images} real images for evaluation',
                       'mmgen')
        pbar = mmcv.ProgressBar(max_num_images)

    # avoid `data_loader` is None
    data_loader = [] if data_loader is None else data_loader
    for data in data_loader:
        if 'real_img' in data:
            reals = data['real_img']
        # key for conditional GAN
        elif 'img' in data:
            reals = data['img']
        else:
            raise KeyError('Cannot found key for images in data_dict. '
                           'Only support `real_img` for unconditional '
                           'datasets and `img` for conditional '
                           'datasets.')

        if reals.shape[1] not in [1, 3]:
            raise RuntimeError('real images should have one or three '
                               'channels in the first, '
                               'not % d' % reals.shape[1])
        if reals.shape[1] == 1:
            reals = reals.repeat(1, 3, 1, 1)

        num_feed = 0
        for metric in vanilla_metrics:
            num_feed_ = metric.feed(reals, 'reals')
            num_feed = max(num_feed_, num_feed)
        for metric in recon_metrics:
            kwargs_ = deepcopy(kwargs)
            kwargs_['mode'] = 'reconstruction'
            prob_dict = model(reals, return_loss=False, **kwargs_)
            num_feed_ = metric.feed(prob_dict, 'reals')
            num_feed = max(num_feed_, num_feed)

        if num_feed <= 0:
            break

        if rank == 0:
            pbar.update(num_feed)

    if rank == 0:
        # finish the pbar stdout
        sys.stdout.write('\n')

    # define mmcv progress bar
    max_num_images = 0 if len(vanilla_metrics) == 0 else max(
        metric.num_fake_need for metric in vanilla_metrics)
    if rank == 0:
        mmcv.print_log(f'Sample {max_num_images} fake images for evaluation',
                       'mmgen')
        pbar = mmcv.ProgressBar(max_num_images)
    # sampling fake images and directly send them to metrics
    total_batch_size = batch_size * ws
    for _ in range(0, max_num_images, total_batch_size):
        fakes = model(
            None,
            num_batches=batch_size,
            return_loss=False,
            sample_model=basic_table_info['sample_model'],
            **kwargs)

        if fakes.shape[1] not in [1, 3]:
            raise RuntimeError('fakes images should have one or three '
                               'channels in the first, '
                               'not % d' % fakes.shape[1])
        if fakes.shape[1] == 1:
            fakes = torch.cat([fakes] * 3, dim=1)

        for metric in vanilla_metrics:
            # feed in fake images
            metric.feed(fakes, 'fakes')

        if rank == 0:
            pbar.update(total_batch_size)

    if rank == 0:
        # finish the pbar stdout
        sys.stdout.write('\n')

    # feed special metric, we do not consider distributed eval here
    for metric in special_metrics:
        metric.prepare()
        fakedata_iterator = iter(
            metric.get_sampler(model.module, batch_size,
                               basic_table_info['sample_model']))
        mmcv.print_log(
            f'Sample {metric.num_images} samples for evaluating {metric.name}',
            'mmgen')
        pbar = mmcv.ProgressBar(metric.num_images)
        for fakes in fakedata_iterator:
            num_left = metric.feed(fakes, 'fakes')
            pbar.update(fakes.shape[0])
            if num_left <= 0:
                break

        # finish the pbar stdout
        sys.stdout.write('\n')

    if rank == 0:
        for metric in metrics:
            metric.summary()

        table_str = make_metrics_table(basic_table_info['train_cfg'],
                                       basic_table_info['ckpt'],
                                       basic_table_info['sample_model'],
                                       metrics)
        logger.info('\n' + table_str)
