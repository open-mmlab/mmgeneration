# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import is_list_of
from mmengine.visualization import Visualizer

from mmgen.core import PixelData
from mmgen.core.sampler import get_sampler
from mmgen.registry import HOOKS

DATA_BATCH = Sequence[dict]


@HOOKS.register_module()
class GenVisualizationHook(Hook):
    """Generation Visualization Hook. Used to visual output samples in
    training, validation and testing. In this hook, we use a list called
    `sample_kwargs_list` to control how to generate samples and how to
    visualize them. Each element in `sample_kwargs_list`, called
    `sample_kwargs`, may contains the following keywords:

    - Required key words:
        - 'type': Value must be string. Denotes what kind of sampler is used to
            generate image. Refers to `:meth:~mmgen.core.sampler.get_sampler`.
    - Optional key words (If not passed, will use the default value):
        - 'n_rows': Value must be int. The number of images in one row.
        - 'num_samples': Value must be int. The number of samples to visualize.
        - 'vis_mode': Value must be string. How to visualize the generated
            samples (e.g. image, gif).
        - 'fixed_input': Value must be bool. Whether use the fixed input
            during the loop.
        - 'draw_gt': Value must be bool. Whether save the real images.
        - 'target_keys': Value must be string or list of string. The keys of
            the target image to visualize.
        - 'name': Value must be string. If not passed, will use
            `sample_kwargs['type']` as default.

    For convenience, we also define a group of alias of samplers' type for
    models supported in MMGeneration. Refers to
    `:attr:self.SAMPLER_TYPE_MAPPING`.

    Example:
        >>> # for GAN models
        >>> custom_hooks = [
        >>>     dict(
        >>>         type='GenVisualizationHook',
        >>>         interval=1000,
        >>>         fixed_input=True,
        >>>         vis_kwargs_list=dict(type='GAN', name='fake_img'))]
        >>> # for Translation models
        >>> custom_hooks = [
        >>>     dict(
        >>>         type='GenVisualizationHook',
        >>>         interval=10,
        >>>         fixed_input=False,
        >>>         vis_kwargs_list=[dict(type='Translation',
        >>>                                  name='translation_train',
        >>>                                  n_samples=6, draw_gt=True,
        >>>                                  n_rows=3),
        >>>                             dict(type='TranslationVal',
        >>>                                  name='translation_val',
        >>>                                  n_samples=16, draw_gt=True,
        >>>                                  n_rows=4)])]

    # NOTE: user-defined vis_kwargs > vis_kwargs_mapping > hook init args

    Args:
        interval (int): Visualization interval. Default: 1000.
        sampler_kwargs_list (Tuple[List[dict], dict]): The list of sampling
            behavior to generate images.
        fixed_input (bool): The default action of whether use fixed input to
            generate samples during the loop. Defaults to True.
        n_samples (Optional[int]): The default value of number of samples to
            visualize. Defaults to 64.
        n_rows (Optional[int]): The default value of number of images in each
            row in the visualization results. Defaults to 8.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
    """

    priority = 'NORMAL'

    VIS_KWARGS_MAPPING = dict(
        GAN=dict(type='Noise', vis_kwargs={}),
        Translation=dict(type='Data', vis_kwargs={}),
        TranslationVal=dict(type='ValData', vis_kwargs={}),
        TranslationTest=dict(type='TestData', vis_kwargs={}),
        DDPMDenoising=dict(
            type='Arguments',
            forward_mode='sampling',
            name='ddpm_sample',
            n_samples=16,
            n_rows=4,
            vis_mode='gif',
            n_skip=1,
            forward_kwargs=dict(
                forward_mode='sampling',
                sample_kwargs=dict(show_pbar=True, save_intermedia=True))))

    def __init__(self,
                 interval: int = 1000,
                 vis_kwargs_list: Tuple[List[dict], dict] = None,
                 fixed_input: bool = True,
                 n_samples: Optional[int] = 64,
                 n_rows: Optional[int] = 8,
                 save_at_test: bool = True,
                 test_vis_keys_list: Optional[Union[str, List[str]]] = None,
                 show: bool = False,
                 wait_time: float = 0):

        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval

        self.vis_kwargs_list = deepcopy(vis_kwargs_list)
        if isinstance(self.vis_kwargs_list, dict):
            self.vis_kwargs_list = [self.vis_kwargs_list]

        self.fixed_input = fixed_input
        self.inputs_buffer = defaultdict(list)

        self.n_samples = n_samples
        self.n_rows = n_rows

        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.save_at_test = save_at_test
        self.test_vis_keys_list = test_vis_keys_list

    @master_only
    def after_val_iter(self, runner: Runner, batch_idx: int,
                       data_batch: Sequence[dict], outputs) -> None:
        """:class:`GenVisualizationHook` do not support visualize during
        validation.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs: outputs of the generation model
        """
        return

    @master_only
    def after_test_iter(self, runner: Runner, batch_idx, data_batch, outputs):
        """Visualize samples after test iteraiton.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs: outputs of the generation model Defaults to None.
        """
        if not self.save_at_test:
            return

        # get color order, mean and std
        module = runner.model
        if hasattr(module, 'module'):
            module = module.module
        data_preprocessor = module.data_preprocessor
        if hasattr(data_preprocessor, 'output_color_order'):
            output_color_order = data_preprocessor.output_color_order
        else:
            output_color_order = 'bgr'
        mean = data_preprocessor.mean
        std = data_preprocessor.std
        for idx, sample in enumerate(outputs):
            curr_idx = batch_idx * len(outputs) + idx
            if self.test_vis_keys_list is None:
                target_keys = [
                    k for k, v in sample.items()
                    if not k.startswith('_') and isinstance(v, PixelData)
                ]
                assert len(target_keys), (
                    'Cannot found PixelData in outputs. Please specific '
                    '\'vis_test_keys_list\'.')
            elif isinstance(self.test_vis_keys_list, str):
                target_keys = [self.test_vis_keys_list]
            else:
                assert is_list_of(self.test_vis_keys_list, str), (
                    'test_vis_keys_list must be str or list of str or None.')
                target_keys = self.test_vis_keys_list

            for key in target_keys:
                name = key.replace('.', '_')
                self._visualizer.add_datasample(
                    name=name,
                    gen_samples=[sample],
                    batch_idx=curr_idx,
                    target_keys=key,
                    n_rows=1,
                    color_order=output_color_order,
                    target_mean=mean.cpu(),
                    target_std=std.cpu())

    @master_only
    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Visualize samples after train iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            self.vis_sample(runner, batch_idx, data_batch, outputs)

    @torch.no_grad()
    def vis_sample(self,
                   runner: Runner,
                   batch_idx: int,
                   data_batch: DATA_BATCH,
                   outputs: Optional[dict] = None) -> None:
        """Visualize samples.

        Args:
            runner (Runner): The runner conatians model to visualize.
            batch_idx (int): The index of the current batch in loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        if isinstance(data_batch, dict):
            num_batches = data_batch['num_batches']
        else:
            num_batches = len(data_batch)

        module = runner.model
        module.eval()
        if hasattr(module, 'module'):
            module = module.module

        forward_func = module.val_step
        # get color order, mean and std
        data_preprocessor = module.data_preprocessor
        if hasattr(data_preprocessor, 'output_color_order'):
            output_color_order = data_preprocessor.output_color_order
        else:
            output_color_order = 'bgr'
        mean = data_preprocessor.mean
        std = data_preprocessor.std

        for vis_kwargs in self.vis_kwargs_list:
            # pop the sample-unrelated values
            vis_kwargs_ = deepcopy(vis_kwargs)
            sampler_type = vis_kwargs_['type']

            # replace with alias
            for alias in self.VIS_KWARGS_MAPPING.keys():
                if alias.upper() == sampler_type.upper():
                    sampler_alias = deepcopy(self.VIS_KWARGS_MAPPING[alias])
                    vis_kwargs_['type'] = sampler_alias.pop('type')
                    for default_k, default_v in sampler_alias.items():
                        vis_kwargs_.setdefault(default_k, default_v)
                    break

            name = vis_kwargs_.pop('name', None)
            if not name:
                name = sampler_type.lower()

            n_samples = vis_kwargs_.pop('n_samples', self.n_samples)
            n_rows = vis_kwargs_.pop('n_rows', self.n_rows)
            n_rows = min(n_rows, n_samples)

            num_iters = math.ceil(n_samples / num_batches)
            vis_kwargs_['max_times'] = num_iters
            vis_kwargs_['num_batches'] = num_batches
            fixed_input = vis_kwargs_.pop('fixed_input', self.fixed_input)
            target_keys = vis_kwargs_.pop('target_keys', None)
            vis_mode = vis_kwargs_.pop('vis_mode', None)

            output_list = []
            if fixed_input and self.inputs_buffer[sampler_type]:
                sampler = self.inputs_buffer[sampler_type]
            else:
                sampler = get_sampler(vis_kwargs_, runner)
            need_save = fixed_input and not self.inputs_buffer[sampler_type]

            for idx, inputs in enumerate(sampler):
                output_list += [out for out in forward_func(inputs)]

                # save inputs
                if need_save:
                    self.inputs_buffer[sampler_type].append(inputs)

            self._visualizer.add_datasample(
                name=name,
                gen_samples=output_list[:n_samples],
                target_keys=target_keys,
                vis_mode=vis_mode,
                n_rows=n_rows,
                color_order=output_color_order,
                target_mean=mean.cpu(),
                target_std=std.cpu(),
                show=self.show,
                wait_time=self.wait_time,
                step=batch_idx + 1,
                **vis_kwargs_)

        module.train()
