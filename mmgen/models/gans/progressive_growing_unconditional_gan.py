# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import _find_tensors

from mmgen.core.optimizer import build_optimizers
from mmgen.models.builder import MODELS, build_module
from ..common import set_requires_grad
from .base_gan import BaseGAN


@MODELS.register_module('StyleGANV1')
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

    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 disc_auxiliary_loss,
                 gen_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self._gen_cfg = deepcopy(generator)
        self.generator = build_module(generator)

        # support no discriminator in testing
        if discriminator is not None:
            self.discriminator = build_module(discriminator)
        else:
            self.discriminator = None

        # support no gan_loss in testing
        if gan_loss is not None:
            self.gan_loss = build_module(gan_loss)
        else:
            self.gan_loss = None

        if disc_auxiliary_loss:
            self.disc_auxiliary_losses = build_module(disc_auxiliary_loss)
            if not isinstance(self.disc_auxiliary_losses, nn.ModuleList):
                self.disc_auxiliary_losses = nn.ModuleList(
                    [self.disc_auxiliary_losses])
        else:
            self.disc_auxiliary_losses = None

        if gen_auxiliary_loss:
            self.gen_auxiliary_losses = build_module(gen_auxiliary_loss)
            if not isinstance(self.gen_auxiliary_losses, nn.ModuleList):
                self.gen_auxiliary_losses = nn.ModuleList(
                    [self.gen_auxiliary_losses])
        else:
            self.gen_auxiliary_losses = None

        # register necessary training status
        self.register_buffer('shown_nkimg', torch.tensor(0.))
        self.register_buffer('_curr_transition_weight', torch.tensor(1.))

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        # this buffer is used to resume model easily
        self.register_buffer(
            '_next_scale_int',
            torch.tensor(self.scales[0][0], dtype=torch.int32))
        # TODO: init it with the same value as `_next_scale_int`
        # a dirty workaround for testing
        self.register_buffer(
            '_curr_scale_int',
            torch.tensor(self.scales[-1][0], dtype=torch.int32))
        if test_cfg is not None:
            self._parse_test_cfg()

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        # whether to use exponential moving average for training
        self.use_ema = self.train_cfg.get('use_ema', False)
        if self.use_ema:
            # use deepcopy to guarantee the consistency
            self.generator_ema = deepcopy(self.generator)

        # setup interpolation operation at the beginning of training iter
        interp_real_cfg = deepcopy(self.train_cfg.get('interp_real', None))
        if interp_real_cfg is None:
            interp_real_cfg = dict(mode='bilinear', align_corners=True)

        self.interp_real_to = partial(F.interpolate, **interp_real_cfg)
        # parsing the training schedule: scales : kimg
        assert isinstance(self.train_cfg['nkimgs_per_scale'],
                          dict), ('Please provide "nkimgs_per_'
                                  'scale" to schedule the training procedure.')
        nkimgs_per_scale = deepcopy(self.train_cfg['nkimgs_per_scale'])
        self.scales = []
        self.nkimgs = []
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
        self.prev_stage = 0
        # actually nkimgs shown at the end of per training stage
        self._actual_nkimgs = []
        # In each scale, transit from previous torgb layer to newer torgb layer
        # with `transition_kimgs` imgs
        self.transition_kimgs = self.train_cfg.get('transition_kimgs', 600)

        # setup optimizer
        self.optimizer = build_optimizers(
            self, deepcopy(self.train_cfg['optimizer_cfg']))
        # get lr schedule
        self.g_lr_base = self.train_cfg['g_lr_base']
        self.d_lr_base = self.train_cfg['d_lr_base']
        # example for lr schedule: {'32': 0.001, '64': 0.0001}
        self.g_lr_schedule = self.train_cfg.get('g_lr_schedule', dict())
        self.d_lr_schedule = self.train_cfg.get('d_lr_schedule', dict())
        # reset the states for optimizers, e.g. momentum in Adam
        self.reset_optim_for_new_scale = self.train_cfg.get(
            'reset_optim_for_new_scale', True)
        # dirty walkround for avoiding optimizer bug in resuming
        self.prev_stage = self.train_cfg.get('prev_stage', self.prev_stage)

    def _parse_test_cfg(self):
        """Parsing train config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

        # whether to use exponential moving average for testing
        self.use_ema = self.test_cfg.get('use_ema', False)
        # TODO: finish ema part

    def sample_from_noise(self,
                          noise,
                          num_batches=0,
                          curr_scale=None,
                          transition_weight=None,
                          sample_model='ema/orig',
                          **kwargs):
        """Sample images from noises by using the generator.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional):  The number of batch size.
                Defaults to 0.

        Returns:
            torch.Tensor | dict: The output may be the direct synthesized \
                images in ``torch.Tensor``. Otherwise, a dict with queried \
                data, including generated images, will be returned.
        """
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

        if sample_model == 'ema':
            assert self.use_ema
            _model = self.generator_ema
        elif sample_model == 'ema/orig' and self.use_ema:
            _model = self.generator_ema
        else:
            _model = self.generator

        outputs = _model(
            noise,
            num_batches=num_batches,
            curr_scale=curr_scale,
            transition_weight=transition_weight,
            **kwargs)

        if isinstance(outputs, dict) and 'noise_batch' in outputs:
            noise = outputs['noise_batch']

        if sample_model == 'ema/orig' and self.use_ema:
            _model = self.generator
            outputs_ = _model(
                noise,
                num_batches=num_batches,
                curr_scale=curr_scale,
                transition_weight=transition_weight,
                **kwargs)
            if isinstance(outputs_, dict):
                outputs['fake_img'] = torch.cat(
                    [outputs['fake_img'], outputs_['fake_img']], dim=0)
            else:
                outputs = torch.cat([outputs, outputs_], dim=0)
        return outputs

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
        batch_size = real_imgs.shape[0]

        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        # check if optimizer from model
        if hasattr(self, 'optimizer'):
            optimizer = self.optimizer

        # update current stage
        self.curr_stage = int(
            min(
                sum(self.cum_nkimgs <= self.shown_nkimg.item()),
                len(self.scales) - 1))
        self.curr_scale = self.scales[self.curr_stage]
        self._curr_scale_int = self._next_scale_int.clone()
        # add new scale and update training status
        if self.curr_stage != self.prev_stage:
            self.prev_stage = self.curr_stage
            self._actual_nkimgs.append(self.shown_nkimg.item())
            # reset optimizer
            if self.reset_optim_for_new_scale:
                optim_cfg = deepcopy(self.train_cfg['optimizer_cfg'])
                optim_cfg['generator']['lr'] = self.g_lr_schedule.get(
                    str(self.curr_scale[0]), self.g_lr_base)
                optim_cfg['discriminator']['lr'] = self.d_lr_schedule.get(
                    str(self.curr_scale[0]), self.d_lr_base)
                self.optimizer = build_optimizers(self, optim_cfg)
                optimizer = self.optimizer
                mmcv.print_log('Reset optimizer for new scale', logger='mmgen')

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

        # resize real image to target scale
        if real_imgs.shape[2:] == self.curr_scale:
            pass
        elif real_imgs.shape[2] >= self.curr_scale[0] and real_imgs.shape[
                3] >= self.curr_scale[1]:
            real_imgs = self.interp_real_to(real_imgs, size=self.curr_scale)
        else:
            raise RuntimeError(
                f'The scale of real image {real_imgs.shape[2:]} is smaller '
                f'than current scale {self.curr_scale}.')

        # disc training
        set_requires_grad(self.discriminator, True)
        optimizer['discriminator'].zero_grad()
        # TODO: add noise sampler to customize noise sampling
        with torch.no_grad():
            fake_imgs = self.generator(
                None,
                num_batches=batch_size,
                curr_scale=self.curr_scale[0],
                transition_weight=transition_weight)

        # disc pred for fake imgs and real_imgs
        disc_pred_fake = self.discriminator(
            fake_imgs,
            curr_scale=self.curr_scale[0],
            transition_weight=transition_weight)
        disc_pred_real = self.discriminator(
            real_imgs,
            curr_scale=self.curr_scale[0],
            transition_weight=transition_weight)
        # get data dict to compute losses for disc
        data_dict_ = dict(
            iteration=curr_iter,
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            curr_scale=self.curr_scale[0],
            transition_weight=transition_weight,
            gen_partial=partial(
                self.generator,
                curr_scale=self.curr_scale[0],
                transition_weight=transition_weight),
            disc_partial=partial(
                self.discriminator,
                curr_scale=self.curr_scale[0],
                transition_weight=transition_weight))

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))
        loss_disc.backward()
        optimizer['discriminator'].step()

        # update training log status
        if dist.is_initialized():
            _batch_size = batch_size * dist.get_world_size()
        else:
            if 'batch_size' not in running_status:
                raise RuntimeError(
                    'You should offer "batch_size" in running status for PGGAN'
                )
            _batch_size = running_status['batch_size']
        self.shown_nkimg += (_batch_size / 1000.)
        log_vars_disc.update(
            dict(
                shown_nkimg=self.shown_nkimg.item(),
                curr_scale=self.curr_scale[0],
                transition_weight=transition_weight))

        # skip generator training if only train discriminator for current
        # iteration
        if (curr_iter + 1) % self.disc_steps != 0:
            results = dict(
                fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
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
            None,
            num_batches=batch_size,
            curr_scale=self.curr_scale[0],
            transition_weight=transition_weight)
        disc_pred_fake_g = self.discriminator(
            fake_imgs,
            curr_scale=self.curr_scale[0],
            transition_weight=transition_weight)

        data_dict_ = dict(
            iteration=curr_iter,
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=fake_imgs,
            disc_pred_fake_g=disc_pred_fake_g)

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
        log_vars.update({'batch_size': batch_size})

        results = dict(fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1

        # check if a new scale will be added in the next iteration
        _curr_stage = int(
            min(
                sum(self.cum_nkimgs <= self.shown_nkimg.item()),
                len(self.scales) - 1))
        # in the next iteration, we will switch to a new scale
        if _curr_stage != self.curr_stage:
            # `self._next_scale_int` is updated at the end of `train_step`
            self._next_scale_int = self._next_scale_int * 2
        return outputs
