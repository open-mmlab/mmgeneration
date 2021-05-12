import os
import shutil
import sys

import mmcv
import torch
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


def make_vanilla_dataloader(img_path, batch_size):
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
        dist=False,
        shuffle=True)
    return dataloader


@torch.no_grad()
def single_gpu_evaluation(model,
                          data_loader,
                          metrics,
                          logger,
                          basic_table_info,
                          batch_size,
                          samples_path=None,
                          **kwargs):
    """Evaluate model with a single gpu.

    This method evaluate model with a single gpu and displays eval progress
     bar.

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
        kwargs (dict): Other arguments.
    """
    # eval special metric online only
    special_metric_name = ['PPL']
    for metric in metrics:
        assert metric.name not in special_metric_name, 'Please eval '\
             f'{metric.name} online'

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

    # sample images
    num_exist = len(
        list(
            mmcv.scandir(
                samples_path, suffix=('.jpg', '.png', '.jpeg', '.JPEG'))))
    if basic_table_info['num_samples'] > 0:
        max_num_images = basic_table_info['num_samples']
    else:
        max_num_images = max(metric.num_images for metric in metrics)
    num_needed = max(max_num_images - num_exist, 0)

    if num_needed > 0:
        mmcv.print_log(f'Sample {num_needed} fake images for evaluation',
                       'mmgen')
        # define mmcv progress bar
        pbar = mmcv.ProgressBar(num_needed)

    # if no images, `num_exist` should be zero
    for begin in range(num_exist, num_needed, batch_size):
        end = min(begin + batch_size, max_num_images)
        fakes = model(
            None,
            num_batches=end - begin,
            return_loss=False,
            sample_model=basic_table_info['sample_model'],
            **kwargs)
        pbar.update(end - begin)

        # save as three-channel
        if fakes.size(1) == 3:
            fakes = fakes[:, [2, 1, 0], ...]
        elif fakes.size(1) == 1:
            fakes = torch.cat([fakes] * 3, dim=1)
        else:
            raise RuntimeError('Generated images must have one or three '
                               'channels in the first dimension, '
                               'not %d' % fakes.size(1))

        for i in range(end - begin):
            images = fakes[i:i + 1]
            images = ((images + 1) / 2)
            images = images.clamp_(0, 1)
            image_name = str(begin + i) + '.png'
            save_image(images, os.path.join(samples_path, image_name))

    if num_needed > 0:
        sys.stdout.write('\n')

    # return if only save sampled images
    if len(metrics) == 0:
        return

    # empty cache to release GPU memory
    torch.cuda.empty_cache()
    fake_dataloader = make_vanilla_dataloader(samples_path, batch_size)
    for metric in metrics:
        mmcv.print_log(f'Evaluate with {metric.name} metric.', 'mmgen')
        metric.prepare()
        # prepare for pbar
        total_need = metric.num_real_need + metric.num_fake_need
        pbar = mmcv.ProgressBar(total_need)
        # feed in real images
        for data in data_loader:
            reals = data['real_img']
            if reals.shape[1] == 1:
                reals = torch.cat([reals] * 3, dim=1)
            num_left = metric.feed(reals, 'reals')
            pbar.update(reals.shape[0])
            if num_left <= 0:
                break
        # feed in fake images
        for data in fake_dataloader:
            fakes = data['real_img']
            if fakes.shape[1] == 1:
                fakes = torch.cat([fakes] * 3, dim=1)
            num_left = metric.feed(fakes, 'fakes')
            pbar.update(fakes.shape[0])
            if num_left <= 0:
                break
        metric.summary()
        sys.stdout.write('\n')
    table_str = make_metrics_table(basic_table_info['train_cfg'],
                                   basic_table_info['ckpt'],
                                   basic_table_info['sample_model'], metrics)
    logger.info('\n' + table_str)
    if delete_samples_path:
        shutil.rmtree(samples_path)


@torch.no_grad()
def single_gpu_online_evaluation(model, data_loader, metrics, logger,
                                 basic_table_info, batch_size, **kwargs):
    """Evaluate model with a single gpu in online mode.

    This method evaluate model with a single gpu and displays eval progress
    bar. Different form `single_gpu_evaluation`, this function will not save
    the images or read images from disks. Namely, there do not exist any IO
    operations in this function. Thus, in general, `online` mode will achieve a
    faster evaluation. However, this mode will take much more memory cost.
    Therefore This evaluation function is recommended to evaluate your model
    with a single metric.

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
    # separate metrics into special metrics and vanilla metrics.
    # For vanilla metrics, images are generated in a random way, and are
    # shared by these metrics. For special metrics like 'PPL', images are
    # generated in a metric-special way and not shared between different
    # metrics.
    special_metrics = []
    vanilla_metrics = []
    special_metric_name = ['PPL']
    for metric in metrics:
        if metric.name in special_metric_name:
            special_metrics.append(metric)
        else:
            vanilla_metrics.append(metric)

    max_num_images = 0 if len(vanilla_metrics) == 0 else max(
        metric.num_images for metric in vanilla_metrics)
    for metric in vanilla_metrics:
        mmcv.print_log(f'Feed reals to {metric.name} metric.', 'mmgen')
        metric.prepare()
        pbar = mmcv.ProgressBar(metric.num_real_need)
        # feed in real images
        for data in data_loader:
            reals = data['real_img']

            if reals.shape[1] not in [1, 3]:
                raise RuntimeError('real images should have one or three '
                                   'channels in the first, '
                                   'not % d' % reals.shape[1])
            if reals.shape[1] == 1:
                reals = torch.cat([reals] * 3, dim=1)
            num_feed = metric.feed(reals, 'reals')
            if num_feed <= 0:
                break

            pbar.update(num_feed)

        # finish the pbar stdout
        sys.stdout.write('\n')

    mmcv.print_log(f'Sample {max_num_images} fake images for evaluation',
                   'mmgen')
    # define mmcv progress bar
    max_num_images = 0 if len(vanilla_metrics) == 0 else max(
        metric.num_fake_need for metric in vanilla_metrics)
    pbar = mmcv.ProgressBar(max_num_images)
    # sampling fake images and directly send them to metrics
    for begin in range(0, max_num_images, batch_size):
        end = min(begin + batch_size, max_num_images)
        fakes = model(
            None,
            num_batches=end - begin,
            return_loss=False,
            sample_model=basic_table_info['sample_model'],
            **kwargs)

        if fakes.shape[1] not in [1, 3]:
            raise RuntimeError('fakes images should have one or three '
                               'channels in the first, '
                               'not % d' % fakes.shape[1])
        if fakes.shape[1] == 1:
            fakes = torch.cat([fakes] * 3, dim=1)
        pbar.update(end - begin)
        fakes = fakes[:end - begin]

        for metric in vanilla_metrics:
            # feed in fake images
            _ = metric.feed(fakes, 'fakes')

    # finish the pbar stdout
    sys.stdout.write('\n')

    # feed special metric
    for metric in special_metrics:
        metric.prepare()
        fakedata_iterator = iter(
            metric.get_sampler(model.module, batch_size,
                               basic_table_info['sample_model']))
        pbar = mmcv.ProgressBar(metric.num_images)
        for fakes in fakedata_iterator:
            num_left = metric.feed(fakes, 'fakes')
            pbar.update(fakes.shape[0])
            if num_left <= 0:
                break

        # finish the pbar stdout
        sys.stdout.write('\n')

    for metric in metrics:
        metric.summary()

    table_str = make_metrics_table(basic_table_info['train_cfg'],
                                   basic_table_info['ckpt'],
                                   basic_table_info['sample_model'], metrics)
    logger.info('\n' + table_str)
