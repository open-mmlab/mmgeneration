import math
import os
import os.path as osp
import sys
import warnings

import mmcv
import torch
from mmcv.runner import HOOKS, Hook, get_dist_info

from ..registry import build_metric


@HOOKS.register_module()
class GenerativeEvalHook(Hook):
    """Evaluation Hook for Generative Models.

    Currently, this evaluation hook can be used to evaluate unconditional
    models. Note that only ``FID`` and ``IS`` metric is supported for the
    distributed training now. In the future, we will support more metrics for
    the evaluation during the training procedure.

    In our config system, you only need to add `evaluation` with the detailed
    configureations. Below is serveral usage cases for different situations.
    What you need to do is to add these lines at the end of your config file.
    Then, you can use this evaluation hook in the training procedure.

    #. Only use FID for evaluation

    .. code-blcok:: python
        :linenos

        evaluation = dict(
            type='GenerativeEvalHook',
            interval=10000,
            metrics=dict(
                type='FID',
                num_images=50000,
                inception_pkl='work_dirs/inception_pkl/ffhq-256-50k-rgb.pkl',
                bgr2rgb=True),
            sample_kwargs=dict(sample_model='ema'))

    #. Use FID and IS simutaneously and save the best checkpoints respectively

    .. code-block:: python
        :linenos

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

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval. Default: 1.
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
        self.interval = interval
        self.dist = dist
        self.sample_kwargs = sample_kwargs if sample_kwargs else dict()
        self.save_best_ckpt = save_best_ckpt
        self.best_metric = best_metric

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
        if not self.every_n_iters(runner, self.interval):
            return

        runner.model.eval()

        # sample fake images
        max_num_images = max(metric.num_images for metric in self.metrics)
        for metric in self.metrics:
            if metric.num_real_feeded >= metric.num_real_need:
                continue
            mmcv.print_log(f'Feed reals to {metric.name} metric.', 'mmgen')
            # feed in real images
            for data in self.dataloader:
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
                num_feed = metric.feed(reals, 'reals')
                if num_feed <= 0:
                    break

        mmcv.print_log(f'Sample {max_num_images} fake images for evaluation',
                       'mmgen')
        batch_size = self.dataloader.batch_size

        rank, ws = get_dist_info()
        total_batch_size = batch_size * ws

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
                    num_left = metric.feed(fakes, 'fakes')
                    if num_left <= 0:
                        break

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
