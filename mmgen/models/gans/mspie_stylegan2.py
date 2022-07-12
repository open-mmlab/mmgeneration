# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmengine import MessageHub
from mmengine.logging import MMLogger
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import Tensor

from mmgen.registry import MODELS
from mmgen.typing import TrainStepInputs
from ..common import gather_log_vars, set_requires_grad
from ..gans.stylegan2 import StyleGAN2

ModelType = Union[Dict, nn.Module]
TrainInput = Union[dict, Tensor]


@MODELS.register_module()
class MSPIEStyleGAN2(StyleGAN2):
    """MS-PIE StyleGAN2.

    In this GAN, we adopt the MS-PIE training schedule so that multi-scale
    images can be generated with a single generator. Details can be found in:
    Positional Encoding as Spatial Inductive Bias in GANs, CVPR2021.

    Args:
        train_settings (dict): Config for training settings.
            Defaults to `dict()`.
    """

    def __init__(self, *args, train_settings=dict(), **kwargs):
        super().__init__(*args, **kwargs)
        self.train_settings = train_settings
        # set the number of upsampling blocks. This value will be used to
        # calculate the current result size according to the size of the input
        # feature map, e.g., positional encoding map
        self.num_upblocks = self.train_settings.get('num_upblocks', 6)

        # multiple input scales (a list of int) that will be added to the
        # original starting scale.
        self.multi_input_scales = self.train_settings.get('multi_input_scales')
        self.multi_scale_probability = self.train_settings.get(
            'multi_scale_probability')

    def train_step(self, data: TrainStepInputs,
                   optim_wrapper: OptimWrapperDict) -> Dict[str, Tensor]:
        """Train GAN model. In the training of GAN models, generator and
        discriminator are updated alternatively. In MMGeneration's design,
        `self.train_step` is called with data input. Therefore we always update
        discriminator, whose updating is relay on real data, and then determine
        if the generator needs to be updated based on the current number of
        iterations. More details about whether to update generator can be found
        in :meth:`should_gen_update`.

        Args:
            data (List[dict]): Data sampled from dataloader.
            optim_wrapper (OptimWrapperDict): OptimWrapperDict instance
                contains OptimWrapper of generator and discriminator.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        inputs_dict, data_sample = self.data_preprocessor(data, True)

        disc_optimizer_wrapper: OptimWrapper = optim_wrapper['discriminator']
        disc_accu_iters = disc_optimizer_wrapper._accumulative_counts

        with disc_optimizer_wrapper.optim_context(self.discriminator):
            log_vars = self.train_discriminator(inputs_dict, data_sample,
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
                        inputs_dict, data_sample, gen_optimizer_wrapper)

                log_vars_gen_list.append(log_vars_gen)
            log_vars_gen = gather_log_vars(log_vars_gen_list)
            log_vars_gen.pop('loss', None)  # remove 'loss' from gen logs

            set_requires_grad(self.discriminator, True)

            # only do ema after generator update
            if self.with_ema_gen:
                self.generator_ema.update_parameters(self.generator)

            log_vars.update(log_vars_gen)

        return log_vars

    def train_generator(self, inputs, data_sample,
                        optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Train generator.

        Args:
            inputs (TrainInput): Inputs from dataloader.
            data_samples (List[GenDataSample]): Data samples from dataloader.
                Do not used in generator's training.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        # num_batches = inputs['real_imgs'].shape[0]
        num_batches = inputs['img'].shape[0]

        noise = self.noise_fn(num_batches=num_batches)
        fake_imgs = self.generator(
            noise, return_noise=False, chosen_scale=self.chosen_scale)

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake, num_batches)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars

    def train_discriminator(
            self, inputs, data_sample,
            optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Train discriminator.

        Args:
            inputs (TrainInput): Inputs from dataloader.
            data_samples (List[GenDataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.
        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        real_imgs = inputs['img']

        if dist.is_initialized():
            # randomly sample a scale for current training iteration
            chosen_scale = np.random.choice(self.multi_input_scales, 1,
                                            self.multi_scale_probability)[0]

            chosen_scale = torch.tensor(chosen_scale, dtype=torch.int).cuda()
            dist.broadcast(chosen_scale, 0)
            chosen_scale = int(chosen_scale.item())

        else:
            logger = MMLogger.get_instance(name='mmgen')
            logger.info(
                'Distributed training has not been initialized. Degrade to '
                'the standard stylegan2')
            chosen_scale = 0

        curr_size = (4 + chosen_scale) * (2**self.num_upblocks)
        # adjust the shape of images
        if real_imgs.shape[-2:] != (curr_size, curr_size):
            real_imgs = F.interpolate(
                real_imgs,
                size=(curr_size, curr_size),
                mode='bilinear',
                align_corners=True)

        num_batches = real_imgs.shape[0]

        noise_batch = self.noise_fn(num_batches=num_batches)
        with torch.no_grad():
            fake_imgs = self.generator(
                noise_batch, return_noise=False, chosen_scale=chosen_scale)
        # store chosen scale for training generator
        setattr(self, 'chosen_scale', chosen_scale)

        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)

        parsed_losses, log_vars = self.disc_loss(disc_pred_fake,
                                                 disc_pred_real, real_imgs)
        optimizer_wrapper.update_params(parsed_losses)
        return log_vars
