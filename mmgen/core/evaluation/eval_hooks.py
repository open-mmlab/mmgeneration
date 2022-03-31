# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import os.path as osp
import sys
import warnings
from bisect import bisect_right

import mmcv
import torch
from mmcv.runner import HOOKS, Hook, get_dist_info

from ..registry import build_metric


@HOOKS.register_module()
class GenerativeEvalHook(Hook):
    """Evaluation Hook for Generative Models.

    This evaluation hook can be used to evaluate unconditional and conditional
    models. Note that only ``FID`` and ``IS`` metric are supported for the
    distributed training now. In the future, we will support more metrics for
    the evaluation during the training procedure.

    In our config system, you only need to add `evaluation` with the detailed
    configureations. Below is several usage cases for different situations.
    What you need to do is to add these lines at the end of your config file.
    Then, you can use this evaluation hook in the training procedure.

    To be noted that, this evaluation hook support evaluation with dynamic
    intervals for FID or other metrics may fluctuate frequently at the end of
    the training process.

    # TODO: fix the online doc

    #. Only use FID for evaluation

    .. code-block:: python
        :linenos:

        evaluation = dict(
            type='GenerativeEvalHook',
            interval=10000,
            metrics=dict(
                type='FID',
                num_images=50000,
                inception_pkl='work_dirs/inception_pkl/ffhq-256-50k-rgb.pkl',
                bgr2rgb=True),
            sample_kwargs=dict(sample_model='ema'))

    #. Use FID and IS simultaneously and save the best checkpoints respectively

    .. code-block:: python
        :linenos:

        evaluation = dict(
            type='GenerativeEvalHook',
            interval=10000,
            metrics=[dict(
                type='FID',
                num_images=50000,
                inception_pkl='work_dirs/inception_pkl/ffhq-256-50k-rgb.pkl',
                bgr2rgb=True),
                dict(type='IS',
                num_images=50000)],
            best_metric=['fid', 'is'],
            sample_kwargs=dict(sample_model='ema'))

    #. Use dynamic evaluation intervals

    .. code-block:: python
        :linenos:

        # interval = 10000 if iter < 50000,
        # interval = 4000, if 50000 <= iter < 750000,
        # interval = 2000, if iter >= 750000

        evaluation = dict(
            type='GenerativeEvalHook',
            interval=dict(milestones=[500000, 750000],
                          interval=[10000, 4000, 2000])
            metrics=[dict(
                type='FID',
                num_images=50000,
                inception_pkl='work_dirs/inception_pkl/ffhq-256-50k-rgb.pkl',
                bgr2rgb=True),
                dict(type='IS',
                num_images=50000)],
            best_metric=['fid', 'is'],
            sample_kwargs=dict(sample_model='ema'))


    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int | dict): Evaluation interval. If int is passed,
            ``eval_hook`` would run under given interval. If a dict is passed,
            The key and value would be interpret as 'milestones' and 'interval'
            of the evaluation.  Default: 1.
        dist (bool, optional): Whether to use distributed evaluation.
            Defaults to True.
        metrics (dict | list[dict], optional): Configs for metrics that will be
            used in evaluation hook. Defaults to None.
        sample_kwargs (dict | None, optional): Additional keyword arguments for
            sampling images. Defaults to None.
        save_best_ckpt (bool, optional): Whether to save the best checkpoint
            according to ``best_metric``. Defaults to ``True``.
        best_metric (str | list, optional): Which metric to be used in saving
            the best checkpoint. Multiple metrics have been supported by
            inputing a list of metric names, e.g., ``['fid', 'is']``.
            Defaults to ``'fid'``.
    """
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -math.inf, 'less': math.inf}
    greater_keys = ['acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'is']
    less_keys = ['loss', 'fid']
    _supported_best_metrics = ['fid', 'is']

    def __init__(self,
                 dataloader,
                 interval=1,
                 dist=True,
                 metrics=None,
                 sample_kwargs=None,
                 save_best_ckpt=True,
                 best_metric='fid'):
        assert metrics is not None
        self.dataloader = dataloader
        self.dist = dist
        self.sample_kwargs = sample_kwargs if sample_kwargs else dict()
        self.save_best_ckpt = save_best_ckpt
        self.best_metric = best_metric

        if isinstance(interval, int):
            self.interval = interval
        elif isinstance(interval, dict):
            if 'milestones' not in interval or 'interval' not in interval:
                raise KeyError(
                    '`milestones` and `interval` must exist in interval dict '
                    'if you want to use the dynamic interval evaluation '
                    f'strategy. But receive [{[k for k in interval.keys()]}] '
                    'in the interval dict.')

            self.milestones = interval['milestones']
            self.interval = interval['interval']
            # check if length of interval match with the milestones
            if len(self.interval) != len(self.milestones) + 1:
                raise ValueError(
                    f'Length of `interval`(={len(self.interval)}) cannot '
                    f'match length of `milestones`(={len(self.milestones)}).')

            # check if milestones is in order
            for idx in range(len(self.milestones) - 1):
                former, latter = self.milestones[idx], self.milestones[idx + 1]
                if former >= latter:
                    raise ValueError(
                        'Elements in `milestones` should in ascending order.')
        else:
            raise TypeError('`interval` only support `int` or `dict`,'
                            f'recieve {type(self.interval)} instead.')

        if isinstance(best_metric, str):
            self.best_metric = [self.best_metric]

        if self.save_best_ckpt:
            not_supported = set(self.best_metric) - set(
                self._supported_best_metrics)
            assert len(not_supported) == 0, (
                f'{not_supported} is not supported for saving best ckpt')

        self.metrics = build_metric(metrics)

        if isinstance(metrics, dict):
            self.metrics = [self.metrics]

        for metric in self.metrics:
            metric.prepare()

        # add support for saving best ckpt
        if self.save_best_ckpt:
            self.rule = {}
            self.compare_func = {}
            self._curr_best_score = {}
            self._curr_best_ckpt_path = {}
            for name in self.best_metric:
                if name in self.greater_keys:
                    self.rule[name] = 'greater'
                else:
                    self.rule[name] = 'less'
                self.compare_func[name] = self.rule_map[self.rule[name]]
                self._curr_best_score[name] = self.init_value_map[
                    self.rule[name]]
                self._curr_best_ckpt_path[name] = None

    def get_current_interval(self, runner):
        """Get current evaluation interval.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if isinstance(self.interval, int):
            return self.interval
        else:
            curr_iter = runner.iter + 1
            index = bisect_right(self.milestones, curr_iter)
            return self.interval[index]

    def before_run(self, runner):
        """The behavior before running.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if self.save_best_ckpt is not None:
            if runner.meta is None:
                warnings.warn('runner.meta is None. Creating an empty one.')
                runner.meta = dict()
            runner.meta.setdefault('hook_msgs', dict())

    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        interval = self.get_current_interval(runner)
        if not self.every_n_iters(runner, interval):
            return

        runner.model.eval()

        batch_size = self.dataloader.batch_size
        rank, ws = get_dist_info()
        total_batch_size = batch_size * ws

        # sample real images
        max_real_num_images = max(metric.num_images - metric.num_real_feeded
                                  for metric in self.metrics)
        # define mmcv progress bar
        if rank == 0 and max_real_num_images > 0:
            mmcv.print_log(
                f'Sample {max_real_num_images} real images for evaluation',
                'mmgen')
            pbar = mmcv.ProgressBar(max_real_num_images)

        if max_real_num_images > 0:
            for data in self.dataloader:
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
                for metric in self.metrics:
                    num_feed_ = metric.feed(reals, 'reals')
                    num_feed = max(num_feed_, num_feed)

                if num_feed <= 0:
                    break

                if rank == 0:
                    pbar.update(num_feed)

        max_num_images = max(metric.num_images for metric in self.metrics)
        if rank == 0:
            mmcv.print_log(
                f'Sample {max_num_images} fake images for evaluation', 'mmgen')

        # define mmcv progress bar
        if rank == 0:
            pbar = mmcv.ProgressBar(max_num_images)

        # sampling fake images and directly send them to metrics
        for _ in range(0, max_num_images, total_batch_size):

            with torch.no_grad():
                fakes = runner.model(
                    None,
                    num_batches=batch_size,
                    return_loss=False,
                    **self.sample_kwargs)

                for metric in self.metrics:
                    # feed in fake images
                    metric.feed(fakes, 'fakes')

            if rank == 0:
                pbar.update(total_batch_size)

        runner.log_buffer.clear()
        # a dirty walkround to change the line at the end of pbar
        if rank == 0:
            sys.stdout.write('\n')
            for metric in self.metrics:
                with torch.no_grad():
                    metric.summary()
                for name, val in metric._result_dict.items():
                    runner.log_buffer.output[name] = val

                    # record best metric and save the best ckpt
                    if self.save_best_ckpt and name in self.best_metric:
                        self._save_best_ckpt(runner, val, name)

            runner.log_buffer.ready = True
        runner.model.train()

        # clear all current states for next evaluation
        for metric in self.metrics:
            metric.clear()

    def _save_best_ckpt(self, runner, new_score, metric_name):
        """Save checkpoint with best metric score.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
            new_score (float): New metric score.
            metric_name (str): Name of metric.
        """
        curr_iter = f'iter_{runner.iter + 1}'

        if self.compare_func[metric_name](new_score,
                                          self._curr_best_score[metric_name]):
            best_ckpt_name = f'best_{metric_name}_{curr_iter}.pth'
            runner.meta['hook_msgs'][f'best_score_{metric_name}'] = new_score

            if self._curr_best_ckpt_path[metric_name] and osp.isfile(
                    self._curr_best_ckpt_path[metric_name]):
                os.remove(self._curr_best_ckpt_path[metric_name])

            self._curr_best_ckpt_path[metric_name] = osp.join(
                runner.work_dir, best_ckpt_name)
            runner.save_checkpoint(
                runner.work_dir, best_ckpt_name, create_symlink=False)
            runner.meta['hook_msgs'][
                f'best_ckpt_{metric_name}'] = self._curr_best_ckpt_path[
                    metric_name]

            self._curr_best_score[metric_name] = new_score
            runner.logger.info(
                f'Now best checkpoint is saved as {best_ckpt_name}.')
            runner.logger.info(f'Best {metric_name} is {new_score:0.4f} '
                               f'at {curr_iter}.')


@HOOKS.register_module()
class TranslationEvalHook(GenerativeEvalHook):
    """Evaluation Hook for Translation Models.

    This evaluation hook can be used to evaluate translation models. Note
    that only ``FID`` and ``IS`` metric are supported for the distributed
    training now. In the future, we will support more metrics for the
    evaluation during the training procedure.

    In our config system, you only need to add `evaluation` with the detailed
    configureations. Below is several usage cases for different situations.
    What you need to do is to add these lines at the end of your config file.
    Then, you can use this evaluation hook in the training procedure.

    To be noted that, this evaluation hook support evaluation with dynamic
    intervals for FID or other metrics may fluctuate frequently at the end of
    the training process.

    # TODO: fix the online doc

    #. Only use FID for evaluation

    .. code-blcok:: python
        :linenos

        evaluation = dict(
            type='TranslationEvalHook',
            target_domain='photo',
            interval=10000,
            metrics=dict(type='FID', num_images=106, bgr2rgb=True))

    #. Use FID and IS simultaneously and save the best checkpoints respectively

    .. code-block:: python
        :linenos

        evaluation = dict(
            type='TranslationEvalHook',
            target_domain='photo',
            interval=10000,
            metrics=[
                dict(type='FID', num_images=106, bgr2rgb=True),
                dict(
                    type='IS',
                    num_images=106,
                    inception_args=dict(type='pytorch'))
            ],
            best_metric=['fid', 'is'])

    #. Use dynamic evaluation intervals

    .. code-block:: python
        :linenos

        # interval = 10000 if iter < 100000,
        # interval = 4000, if 100000 <= iter < 200000,
        # interval = 2000, if iter >= 200000

        evaluation = dict(
            type='TranslationEvalHook',
            interval=dict(milestones=[100000, 200000],
                          interval=[10000, 4000, 2000]),
            target_domain='zebra',
            metrics=[
                dict(type='FID', num_images=140, bgr2rgb=True),
                dict(type='IS', num_images=140)
            ],
            best_metric=['fid', 'is'])


    Args:
        target_domain (str): Target domain of output image.
    """

    def __init__(self, *args, target_domain, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_domain = target_domain

    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        interval = self.get_current_interval(runner)
        if not self.every_n_iters(runner, interval):
            return

        runner.model.eval()
        source_domain = runner.model.module.get_other_domains(
            self.target_domain)[0]
        # feed real images
        max_num_images = max(metric.num_images for metric in self.metrics)
        for metric in self.metrics:
            if metric.num_real_feeded >= metric.num_real_need:
                continue
            mmcv.print_log(f'Feed reals to {metric.name} metric.', 'mmgen')
            # feed in real images
            for data in self.dataloader:
                # key for translation model
                if f'img_{self.target_domain}' in data:
                    reals = data[f'img_{self.target_domain}']
                # key for conditional GAN
                else:
                    raise KeyError(
                        'Cannot found key for images in data_dict. ')
                num_feed = metric.feed(reals, 'reals')
                if num_feed <= 0:
                    break

        mmcv.print_log(f'Sample {max_num_images} fake images for evaluation',
                       'mmgen')

        rank, ws = get_dist_info()

        # define mmcv progress bar
        if rank == 0:
            pbar = mmcv.ProgressBar(max_num_images)

        # feed in fake images
        for data in self.dataloader:
            # key for translation model
            if f'img_{source_domain}' in data:
                with torch.no_grad():
                    output_dict = runner.model(
                        data[f'img_{source_domain}'],
                        test_mode=True,
                        target_domain=self.target_domain,
                        **self.sample_kwargs)
                fakes = output_dict['target']
            # key Error
            else:
                raise KeyError('Cannot found key for images in data_dict. ')
            # sampling fake images and directly send them to metrics
            # pbar update number for one proc
            num_update = 0
            for metric in self.metrics:
                if metric.num_fake_feeded >= metric.num_fake_need:
                    continue
                num_feed = metric.feed(fakes, 'fakes')
                num_update = max(num_update, num_feed)
                if num_feed <= 0:
                    break

            if rank == 0:
                if num_update > 0:
                    pbar.update(num_update * ws)

        runner.log_buffer.clear()
        # a dirty walkround to change the line at the end of pbar
        if rank == 0:
            sys.stdout.write('\n')
            for metric in self.metrics:
                with torch.no_grad():
                    metric.summary()
                for name, val in metric._result_dict.items():
                    runner.log_buffer.output[name] = val

                    # record best metric and save the best ckpt
                    if self.save_best_ckpt and name in self.best_metric:
                        self._save_best_ckpt(runner, val, name)

            runner.log_buffer.ready = True
        runner.model.train()

        # clear all current states for next evaluation
        for metric in self.metrics:
            metric.clear()
