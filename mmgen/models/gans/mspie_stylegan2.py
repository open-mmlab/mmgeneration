# Copyright (c) OpenMMLab. All rights reserved.
import logging
from functools import partial

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import _find_tensors

from ..builder import MODELS
from ..common import set_requires_grad
from .static_unconditional_gan import StaticUnconditionalGAN


@MODELS.register_module()
class MSPIEStyleGAN2(StaticUnconditionalGAN):
    """MS-PIE StyleGAN2.

    In this GAN, we adopt the MS-PIE training schedule so that multi-scale
    images can be generated with a single generator. Details can be found in:
    Positional Encoding as Spatial Inductive Bias in GANs, CVPR2021.

    Args:
        generator (dict): Config for generator.
        discriminator (dict): Config for discriminator.
        gan_loss (dict): Config for generative adversarial loss.
        disc_auxiliary_loss (dict): Config for auxiliary loss to
            discriminator.
        gen_auxiliary_loss (dict | None, optional): Config for auxiliary loss
            to generator. Defaults to None.
        train_cfg (dict | None, optional): Config for training schedule.
            Defaults to None.
        test_cfg (dict | None, optional): Config for testing schedule. Defaults
            to None.
    """

    def _parse_train_cfg(self):
        super(MSPIEStyleGAN2, self)._parse_train_cfg()

        # set the number of upsampling blocks. This value will be used to
        # calculate the current result size according to the size of the input
        # feature map, e.g., positional encoding map
        self.num_upblocks = self.train_cfg.get('num_upblocks', 6)

        # multiple input scales (a list of int) that will be added to the
        # original starting scale.
        self.multi_input_scales = self.train_cfg.get('multi_input_scales')
        self.multi_scale_probability = self.train_cfg.get(
            'multi_scale_probability')

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   running_status=None):
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
        # get data from data_batch
        real_imgs = data_batch['real_img']
        # If you adopt ddp, this batch size is local batch size for each GPU.
        # If you adopt dp, this batch size is the global batch size as usual.
        batch_size = real_imgs.shape[0]

        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        if dist.is_initialized():
            # randomly sample a scale for current training iteration
            chosen_scale = np.random.choice(self.multi_input_scales, 1,
                                            self.multi_scale_probability)[0]

            chosen_scale = torch.tensor(chosen_scale, dtype=torch.int).cuda()
            dist.broadcast(chosen_scale, 0)
            chosen_scale = int(chosen_scale.item())

        else:
            mmcv.print_log(
                'Distributed training has not been initialized. Degrade to '
                'the standard stylegan2',
                logger='mmgen',
                level=logging.WARN)
            chosen_scale = 0

        curr_size = (4 + chosen_scale) * (2**self.num_upblocks)
        # adjust the shape of images
        if real_imgs.shape[-2:] != (curr_size, curr_size):
            real_imgs = F.interpolate(
                real_imgs,
                size=(curr_size, curr_size),
                mode='bilinear',
                align_corners=True)

        # disc training
        set_requires_grad(self.discriminator, True)
        optimizer['discriminator'].zero_grad()
        # TODO: add noise sampler to customize noise sampling
        with torch.no_grad():
            fake_imgs = self.generator(
                None, num_batches=batch_size, chosen_scale=chosen_scale)

        # disc pred for fake imgs and real_imgs
        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)
        # get data dict to compute losses for disc
        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            iteration=curr_iter,
            batch_size=batch_size,
            gen_partial=partial(self.generator, chosen_scale=chosen_scale))

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))
        loss_disc.backward()
        optimizer['discriminator'].step()

        # skip generator training if only train discriminator for current
        # iteration
        if (curr_iter + 1) % self.disc_steps != 0:
            results = dict(
                fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
            log_vars_disc['curr_size'] = curr_size
            outputs = dict(
                log_vars=log_vars_disc,
                num_samples=batch_size,
                results=results)
            if hasattr(self, 'iteration'):
                self.iteration += 1
            return outputs

        # generator training
        set_requires_grad(self.discriminator, False)
        optimizer['generator'].zero_grad()

        # TODO: add noise sampler to customize noise sampling
        fake_imgs = self.generator(
            None, num_batches=batch_size, chosen_scale=chosen_scale)
        disc_pred_fake_g = self.discriminator(fake_imgs)

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=fake_imgs,
            disc_pred_fake_g=disc_pred_fake_g,
            iteration=curr_iter,
            batch_size=batch_size,
            gen_partial=partial(self.generator, chosen_scale=chosen_scale))

        loss_gen, log_vars_g = self._get_gen_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_gen))

        loss_gen.backward()
        optimizer['generator'].step()

        log_vars = {}
        log_vars.update(log_vars_g)
        log_vars.update(log_vars_disc)
        log_vars['curr_size'] = curr_size

        results = dict(fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs
