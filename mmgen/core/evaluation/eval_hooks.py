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
    models. Note that only ``FID`` metric is supported for the distributed
    training now. In the future, we will support more metrics for evaluation
    during the training procedure.

    TODO: Support ``save_best_ckpt`` feature.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval. Default: 1.
        dist (bool, optional): Whether to use distributed evaluation.
            Defaults to True.
        metrics (dict | list[dict], optional): Configs for metrics that will be
            used in evaluation hook. Defaults to None.
        sample_kwargs (dict | None, optional): Additional keyword arguments for
            sampling images. Defaults to None.
    """
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -math.inf, 'less': math.inf}
    greater_keys = ['acc', 'top', 'AR@', 'auc', 'precision', 'mAP']
    less_keys = ['loss', 'fid']
    _supported_best_metrics = ['fid']

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

        if self.save_best_ckpt:
            assert self.best_metric in self._supported_best_metrics, (
                f'{self.best_metric} is not supported for saving best ckpt')

        self.metrics = build_metric(metrics)

        if isinstance(metrics, dict):
            self.metrics = [self.metrics]

        for metric in self.metrics:
            metric.prepare()

        # add support for saving best ckpt
        if self.save_best_ckpt:
            if best_metric in self.greater_keys:
                self.rule = 'greater'
            else:
                self.rule = 'less'
            self.compare_func = self.rule_map[self.rule]
            self._curr_best_score = self.init_value_map[self.rule]
            self._curr_best_ckpt_path = None

    def before_run(self, runner):
        """The behavior before running.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if self.save_best is not None:
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
                reals = data['real_img']
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
                    if self.save_best_ckpt and name == self.best_metric:
                        self._save_best_ckpt(runner, val)

            runner.log_buffer.ready = True
        runner.model.train()

        # clear all current states for next evaluation
        for metric in self.metrics:
            metric.clear()

    def _save_best_ckpt(self, runner, new_score):
        curr_iter = f'iter_{runner.iter + 1}'

        if self.compare_func(new_score, self._curr_best_score):
            best_ckpt_name = f'best_{self.best_metric}_{curr_iter}.pth'
            runner.meta['hooks_msgs']['best_score'] = new_score

            if self._curr_best_ckpt_path and osp.isfile(
                    self._curr_best_ckpt_path):
                os.remove(self._curr_best_ckpt_path)

            self._curr_best_ckpt_path = osp.join(runner.work_dir,
                                                 best_ckpt_name)
            runner.save_checkpoint(
                runner.work_dir, best_ckpt_name, create_symlink=False)
            runner.meta['hook_msgs']['best_ckpt'] = self._curr_best_ckpt_path

            self._curr_best_score = new_score
            runner.logger.info(
                f'Now best checkpoint is saved as {best_ckpt_name}.')
            runner.logger.info(f'Best {self.best_metric} is {new_score:0.4f} '
                               f'at {curr_iter}.')
