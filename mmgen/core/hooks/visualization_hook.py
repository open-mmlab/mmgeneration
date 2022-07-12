# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple

import torch
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import is_list_of
from mmengine.visualization import Visualizer
from torch import Tensor

from mmgen.core.data_structures import GenDataSample
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
        - 'gt_keys': Value must be string or list of string. The keys of the
            real images to visualize.
        - 'name': Value must be string. If not passed, will use
            `sample_kwargs['type']` as default.

    For convenience, we also define a group of alias of samplers' type for
    models supported in MMGeneration. Refers to
    `:attr:self.SAMPLER_TYPE_MAPPING`.

    Example:
        >>> # for GAN models
        >>> custom_hooks = [
        >>>     dict(
        >>>         type='GenerationVisualizationHook',
        >>>         interval=1000,
        >>>         fixed_input=True,
        >>>         sample_kwargs_list=dict(type='GAN', name='fake_img'))]
        >>> # for Translation models
        >>> custom_hooks = [
        >>>     dict(
        >>>         type='GenerationVisualizationHook',
        >>>         interval=10,
        >>>         fixed_input=False,
        >>>         sample_kwargs_list=[dict(type='Translation',
        >>>                                  name='translation_train',
        >>>                                  n_samples=6, draw_gt=True,
        >>>                                  n_rows=3),
        >>>                             dict(type='TranslationVal',
        >>>                                  name='translation_val',
        >>>                                  n_samples=16, draw_gt=True,
        >>>                                  n_rows=4)])]

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

    SAMPLE_KWARGS_MAPPING = dict(
        GAN=dict(type='noise', sample_kwargs={}, vis_kwargs={}),
        Translation=dict(type='Data', sample_kwargs={}),
        TranslationVal=dict(type='ValData', sample_kwargs={}),
        TranslationTest=dict(type='TestData', sample_kwargs={}),
        DDPMDenoising=dict(
            type='Arguments',
            sample_kwargs=dict(
                forward_mode='sampling',
                name='ddpm_sample',
                n_samples=16,
                n_rows=4,
                vis_mode='gif',
                save_intermedia=True,
                show_pbar=True),
            vis_kwargs=dict(n_skip=1)))

    def __init__(self,
                 interval: int = 1000,
                 sample_kwargs_list: Tuple[List[dict], dict] = None,
                 fixed_input: bool = True,
                 n_samples: Optional[int] = 64,
                 n_rows: Optional[int] = 8,
                 show: bool = False,
                 wait_time: float = 0):

        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval

        self.sample_kwargs_list = deepcopy(sample_kwargs_list)
        if isinstance(self.sample_kwargs_list, dict):
            self.sample_kwargs_list = [self.sample_kwargs_list]

        self.fixed_input = fixed_input
        self.inputs_buffer = defaultdict(list)

        self.n_samples = n_samples
        self.n_rows = n_rows

        # NOTE: copy from mmdet
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time

    @master_only
    def after_val_iter(self, runner: Runner, batch_idx: int,
                       data_batch: Sequence[dict], outputs) -> None:
        """NOTE: do not support visualize during validation

        Visualize samples after valditaion iteraiton.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs: outputs of the generation model
        """
        pass

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
        if self.every_n_inner_iters(batch_idx, self.interval):
            self.vis_sample(
                runner, batch_idx, data_batch, outputs, mode='test')

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
            self.vis_sample(
                runner, batch_idx, data_batch, outputs, mode='train')

    @torch.no_grad()
    def vis_sample(self,
                   runner: Runner,
                   batch_idx: int,
                   data_batch: DATA_BATCH,
                   outputs: Optional[dict] = None,
                   mode: str = 'train') -> None:
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

        if mode in ['train', 'val']:
            forward_func = module.val_step
        else:
            forward_func = module.test_step

        # get color order, mean and std
        data_preprocessor = module.data_preprocessor
        if hasattr(data_preprocessor, 'output_color_order'):
            output_color_order = data_preprocessor.output_color_order
        else:
            output_color_order = 'bgr'
        mean = data_preprocessor.mean
        std = data_preprocessor.std

        for sample_kwargs in self.sample_kwargs_list:

            # pop the sample-unrelated values
            sample_kwargs_ = deepcopy(sample_kwargs)
            sampler_type = sample_kwargs_['type']

            # replace with alias
            for alias in self.SAMPLE_KWARGS_MAPPING.keys():
                if alias.upper() == sampler_type.upper():
                    sampler_alias = self.SAMPLE_KWARGS_MAPPING[alias]
                    sampler_type = sampler_alias['type']
                    default_sample_kwargs = sampler_alias['sample_kwargs']
                    sample_kwargs_['type'] = sampler_type
                    for default_k, default_v in default_sample_kwargs.items():
                        sample_kwargs_.setdefault(default_k, default_v)
                    break

            name = sample_kwargs_.pop('name', None)
            if not name:
                name = sampler_type

            n_samples = sample_kwargs_.pop('n_samples', self.n_samples)
            n_rows = sample_kwargs_.pop('n_rows', self.n_rows)
            num_iters = math.ceil(n_samples / num_batches)
            sample_kwargs_['max_times'] = num_iters
            sample_kwargs_['num_batches'] = num_batches
            fixed_input = sample_kwargs_.pop('fixed_input', self.fixed_input)
            draw_gt = sample_kwargs_.pop('draw_gt', False)
            gt_keys = sample_kwargs_.pop('gt_keys', None)
            vis_mode = sample_kwargs_.pop('vis_mode', None)
            vis_kwargs = sample_kwargs_.pop('vis_kwargs', dict())

            output_list = []
            input_list = []

            if fixed_input and self.inputs_buffer[sampler_type]:
                sampler = self.inputs_buffer[sampler_type]
            else:
                sampler = get_sampler(sample_kwargs_, runner)
            need_save = fixed_input and not self.inputs_buffer[sampler_type]

            for inputs in sampler:
                output_list.append(forward_func(inputs))

                # save inputs
                if need_save:
                    self.inputs_buffer[sampler_type].append(inputs)

                # save processed input to visualize
                processed_input, data_samples = data_preprocessor(inputs)
                processed_input_dict = dict(
                    inputs=processed_input, data_sample=data_samples)
                input_list.append(processed_input_dict)

            if sampler_type != 'Arguments':
                inputs = gather_samples(input_list, max_size=n_samples)
            outputs = gather_samples(output_list, max_size=n_samples)

            self._visualizer.add_datasample(
                name=name,
                gen_samples=outputs,
                gt_samples=inputs,
                draw_gt=draw_gt,
                gt_keys=gt_keys,
                vis_mode=vis_mode,
                n_rows=n_rows,
                color_order=output_color_order,
                target_mean=mean.cpu(),
                target_std=std.cpu(),
                show=self.show,
                wait_time=self.wait_time,
                step=batch_idx + 1,
                **vis_kwargs)

        module.train()


def gather_samples(samples: List[dict],
                   max_size: Optional[int] = None) -> Tuple[dict, Tensor]:
    """Gather a list of dict altogether.

    Args:
        samples (List[dict]): List of sample to gather.
        max_size (Optional[int]): Max size of the sample. If the length of
            `samples` after gathering is larger than `max_size`, `samples`
            will be truncated to `max_size`.

    Returns:
        Tuple [dict, Tensor]: Gathered samples.
    """
    if is_list_of(samples, dict):
        first_element_key = samples[0].keys()
        samples_dict = dict()
        for k in first_element_key:
            sample_gathered = [s[k] for s in samples]
            samples_dict[k] = gather_samples(sample_gathered, max_size)
        return samples_dict

    elif is_list_of(samples, Tensor):
        sample_gathered = torch.cat(samples, dim=0)
        if max_size is not None:
            sample_gathered = sample_gathered[:max_size]
        return sample_gathered

    elif is_list_of(samples, list):
        # list of empty list, directly return
        if all([not s for s in samples]):
            return []
        # flatten the list
        samples_ = [s[0] for s in samples]
        return gather_samples(samples_, max_size)

    elif is_list_of(samples, GenDataSample):
        if max_size is not None:
            return samples[:max_size]
        return samples

    else:
        raise ValueError('Only support list of dict or tensor.')
