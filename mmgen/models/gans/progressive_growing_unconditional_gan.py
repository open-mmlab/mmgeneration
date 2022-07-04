# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Dict, List, Optional, Union

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import MessageHub
from mmengine.dist import get_world_size
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import Tensor

from mmgen.core import GenDataSample
from mmgen.registry import MODELS
from mmgen.typing import ForwardInputs, TrainStepInputs
from ..common import gather_log_vars, get_valid_num_batches, set_requires_grad
from .base_gan import BaseGAN

ModelType = Union[Dict, nn.Module]
TrainInput = Union[dict, Tensor]


# @MODELS.register_module('StyleGANV1')
@MODELS.register_module('PGGAN')
@MODELS.register_module()
class ProgressiveGrowingGAN(BaseGAN):
    """Progressive Growing Unconditional GAN.

    In this GAN model, we implement progressive growing training schedule,
    which is proposed in Progressive Growing of GANs for improved Quality,
    Stability and Variation, ICLR 2018.

    We highly recommend to use ``GrowScaleImgDataset`` for saving computational
    load in data pre-processing.

    Notes for **using PGGAN**:

    #. In official implementation, Tero uses gradient penalty with
       ``norm_mode="HWC"``
    #. We do not implement ``minibatch_repeats`` where has been used in
       official Tensorflow implementation.

    Notes for resuming progressive growing GANs:
    Users should specify the ``prev_stage`` in ``train_cfg``. Otherwise, the
    model is possible to reset the optimizer status, which will bring
    inferior performance. For example, if your model is resumed from the
    `256` stage, you should set ``train_cfg=dict(prev_stage=256)``.

    Args:
        generator (dict): Config for generator.
        discriminator (dict): Config for discriminator.
    """

    def __init__(
            self,
            generator,
            discriminator,
            data_preprocessor,
            noise_size,
            # g_lr_base,
            # d_lr_base,
            nkimgs_per_scale,
            interp_real=None,
            # g_lr_schedule=dict(),
            # d_lr_schedule=dict(),
            transition_kimgs: int = 600,
            # reset_optim_for_new_scale: bool = True,
            prev_stage: int = 0,
            ema_config: Optional[Dict] = None):
        super().__init__(generator, discriminator, data_preprocessor, 1, 1,
                         noise_size, ema_config)

        # register necessary training status
        self.register_buffer('shown_nkimg', torch.tensor(0.))
        self.register_buffer('_curr_transition_weight', torch.tensor(1.))

        if interp_real is None:
            interp_real = dict(mode='bilinear', align_corners=True)
        self.interp_real_to = partial(F.interpolate, **interp_real)

        self.scales, self.nkimgs = [], []
        for k, v in nkimgs_per_scale.items():
            # support for different data types
            if isinstance(k, str):
                k = (int(k), int(k))
            elif isinstance(k, int):
                k = (k, k)
            else:
                assert mmcv.is_tuple_of(k, int)

            # sanity check for the order of scales
            assert len(self.scales) == 0 or k[0] > self.scales[-1][0]
            self.scales.append(k)
            self.nkimgs.append(v)

        self.cum_nkimgs = np.cumsum(self.nkimgs)
        self.curr_stage = 0
        # dirty walkround for avoiding optimizer bug in resuming
        self.prev_stage = prev_stage
        # actually nkimgs shown at the end of per training stage
        self._actual_nkimgs = []
        # In each scale, transit from previous torgb layer to newer torgb layer
        # with `transition_kimgs` imgs
        self.transition_kimgs = transition_kimgs

        # this buffer is used to resume model easily
        self.register_buffer(
            '_next_scale_int',
            torch.tensor(self.scales[0][0], dtype=torch.int32))
        # TODO: init it with the same value as `_next_scale_int`
        # a dirty workaround for testing
        self.register_buffer(
            '_curr_scale_int',
            torch.tensor(self.scales[-1][0], dtype=torch.int32))

    def forward(self,
                batch_inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None):
        """Sample images from noises by using the generator."""
        # use `self.curr_scale` if curr_scale is None
        curr_scale = batch_inputs.get('curr_scale', None)
        if curr_scale is None:
            # in training, 'curr_scale' will be set as attribute
            if hasattr(self, 'curr_scale'):
                curr_scale = self.curr_scale[0]
            # in testing, adopt '_curr_scale_int' from buffer as testing scale
            else:
                curr_scale = self._curr_scale_int.item()

        # use `self._curr_transition_weight` if `transition_weight` is None
        transition_weight = batch_inputs.get('transition_weight', None)
        if transition_weight is None:
            transition_weight = self._curr_transition_weight.item()

        if isinstance(batch_inputs, Tensor):
            noise = batch_inputs
            curr_scale = transition_weight = None
        else:
            noise = batch_inputs.get('noise', None)
            num_batches = get_valid_num_batches(batch_inputs)
            noise = self.noise_fn(noise, num_batches=num_batches)

            curr_scale = batch_inputs.get('curr_scale', None)
            transition_weight = batch_inputs.get('transition_weight', None)

        # use `self.curr_scale` if curr_scale is None
        if curr_scale is None:
            # in training, 'curr_scale' will be set as attribute
            if hasattr(self, 'curr_scale'):
                curr_scale = self.curr_scale[0]
            # in testing, adopt '_curr_scale_int' from buffer as testing scale
            else:
                curr_scale = self._curr_scale_int.item()

        # use `self._curr_transition_weight` if `transition_weight` is None
        if transition_weight is None:
            transition_weight = self._curr_transition_weight.item()

        sample_model = self._get_valid_mode(batch_inputs, None)

        if sample_model in ['ema', 'ema/orig']:
            _model = self.generator_ema
        else:
            _model = self.generator

        outputs = _model(
            noise,
            num_batches=num_batches,
            curr_scale=curr_scale,
            transition_weight=transition_weight,
            # **kwargs
        )

        if sample_model == 'ema/orig':
            _model = self.generator
            outputs_orig = _model(
                noise,
                num_batches=num_batches,
                curr_scale=curr_scale,
                transition_weight=transition_weight,
                # **kwargs
            )
            outputs = dict(ema=outputs, orig=outputs_orig)
        return outputs

    def train_generator(self, inputs: TrainInput,
                        data_samples: List[GenDataSample],
                        optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        real_imgs = inputs
        num_batches = real_imgs.shape[0]
        noise_batch = self.noise_fn(num_batches=num_batches)

        with torch.no_grad():
            fake_imgs = self.generator(
                noise_batch,
                curr_scale=self.curr_scale[0],
                transition_weight=self._curr_transition_weight,
                return_noise=False)
        disc_pred_fake = self.discriminator(
            fake_imgs,
            curr_scale=self.curr_scale[0],
            transition_weight=self._curr_transition_weight)
        disc_pred_real = self.discriminator(
            fake_imgs,
            curr_scale=self.curr_scale[0],
            transition_weight=self._curr_transition_weight)

        parsed_loss, log_vars = self.disc_loss(disc_pred_fake, disc_pred_real)
        optimizer_wrapper.update_params(parsed_loss)
        return log_vars

    def disc_loss(self):
        pass

    def train_discriminator(
            self, inputs: TrainInput, data_samples: List[GenDataSample],
            optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        real_imgs = inputs
        num_batches = real_imgs.shape[0]
        noise_batch = self.noise_fn(num_batches=num_batches)

        fake_imgs = self.generator(
            noise_batch,
            num_batches=num_batches,
            curr_scale=self.curr_scale[0],
            transition_weight=self._curr_transition_weight)
        disc_pred_fake_g = self.discriminator(
            fake_imgs,
            curr_scale=self.curr_scale[0],
            transition_weight=self._curr_transition_weight)

        parsed_loss, log_vars = self.gen_loss(disc_pred_fake_g)
        optimizer_wrapper.update_params(parsed_loss)
        return log_vars

    def gen_loss(self):
        pass

    def train_step(self, data: TrainStepInputs,
                   optim_wrapper: OptimWrapperDict):
        """Train step function.

        This function implements the standard training iteration for
        asynchronous adversarial training. Namely, in each iteration, we first
        update discriminator and then compute loss for generator with the newly
        updated discriminator.

        As for distributed training, we use the ``reducer`` from ddp to
        synchronize the necessary params in current computational graph.

        Args:
            data_batch (dict): Input data from dataloader.
            optimizer (dict): Dict contains optimizer for generator and
                discriminator.
            ddp_reducer (:obj:`Reducer` | None, optional): Reducer from ddp.
                It is used to prepare for ``backward()`` in ddp. Defaults to
                None.
            running_status (dict | None, optional): Contains necessary basic
                information for training, e.g., iteration number. Defaults to
                None.

        Returns:
            dict: Contains 'log_vars', 'num_samples', and 'results'.
        """
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')

        # update current stage
        self.curr_stage = int(
            min(
                sum(self.cum_nkimgs <= self.shown_nkimg.item()),
                len(self.scales) - 1))
        self.curr_scale = self.scales[self.curr_stage]
        self._curr_scale_int = self._next_scale_int.clone()

        if self.curr_stage != self.prev_stage:
            self.prev_stage = self.curr_stage
            self._actual_nkimgs.append(self.shown_nkimg.item())

        inputs, data_sample = self.data_preprocessor(data, True)

        curr_scale = str(self.curr_scale[0])
        disc_optimizer_wrapper: OptimWrapper = optim_wrapper[
            f'discriminator_{curr_scale}']
        gen_optimizer_wrapper: OptimWrapper = optim_wrapper[
            f'generator_{curr_scale}']
        disc_accu_iters = disc_optimizer_wrapper._accumulative_counts

        # update training configs, like transition weight for torgb layers.
        # get current transition weight for interpolating two torgb layers
        if self.curr_stage == 0:
            transition_weight = 1.
        else:
            transition_weight = (
                self.shown_nkimg.item() -
                self._actual_nkimgs[-1]) / self.transition_kimgs
            # clip to [0, 1]
            transition_weight = min(max(transition_weight, 0.), 1.)
        self._curr_transition_weight = torch.tensor(transition_weight).to(
            self._curr_transition_weight)

        if inputs.shape[2:] == self.curr_scale:
            pass
        elif inputs.shape[2] >= self.curr_scale[0] and inputs.shape[
                3] >= self.curr_scale[1]:
            inputs = self.interp_real_to(inputs, size=self.curr_scale)
        else:
            raise RuntimeError(
                f'The scale of real image {inputs.shape[2:]} is smaller '
                f'than current scale {self.curr_scale}.')

        # normal gan training process
        with disc_optimizer_wrapper.optim_context(self.discriminator):
            log_vars = self.train_discriminator(inputs, data_sample,
                                                disc_optimizer_wrapper)

        # add 1 to `curr_iter` because iter is updated in train loop.
        # Whether to update the generator. We update generator with
        # discriminator is fully updated for `self.n_discriminator_steps`
        # iterations. And one full updating for discriminator contains
        # `disc_accu_counts` times of grad accumulations.
        if (curr_iter + 1) % (self.discriminator_steps * disc_accu_iters) == 0:
            set_requires_grad(self.discriminator, False)
            gen_optimizer_wrapper = optim_wrapper['generator']
            gen_accu_iters = gen_optimizer_wrapper._accumulative_counts

            log_vars_gen_list = []
            # init optimizer wrapper status for generator manually
            gen_optimizer_wrapper.initialize_count_status(
                self.generator, 0, self.generator_steps * gen_accu_iters)
            for _ in range(self.generator_steps * gen_accu_iters):
                with gen_optimizer_wrapper.optim_context(self.generator):
                    log_vars_gen = self.train_generator(
                        inputs, data_sample, gen_optimizer_wrapper)

                log_vars_gen_list.append(log_vars_gen)
            log_vars_gen = gather_log_vars(log_vars_gen_list)
            log_vars_gen.pop('loss', None)  # remove 'loss' from gen logs

            set_requires_grad(self.discriminator, True)

            # only do ema after generator update
            if self.with_ema_gen:
                self.generator_ema.update_parameters(self.generator)

            log_vars.update(log_vars_gen)

        # add batch size info to log_vars
        _batch_size = inputs.shape[0] * get_world_size()
        self.show_nkimg += (_batch_size / 1000.)

        log_vars.update(
            dict(
                shown_nkimg=self.shown_nkimg.item(),
                curr_scale=self.curr_scale[0],
                transition_weight=transition_weight))

        # check if a new scale will be added in the next iteration
        _curr_stage = int(
            min(
                sum(self.cum_nkimgs <= self.shown_nkimg.item()),
                len(self.scales) - 1))
        # in the next iteration, we will switch to a new scale
        if _curr_stage != self.curr_stage:
            # `self._next_scale_int` is updated at the end of `train_step`
            self._next_scale_int = self._next_scale_int * 2

        return log_vars
