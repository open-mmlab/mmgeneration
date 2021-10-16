# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import _find_tensors

from ..builder import MODELS, build_module
from ..common import set_requires_grad
from .base_gan import BaseGAN


@MODELS.register_module('BasiccGAN')
@MODELS.register_module()
class BasicConditionalGAN(BaseGAN):
    """Basic conditional GANs.

    This is the conditional GAN model containing standard adversarial training
    schedule. To fulfill the requirements of various GAN algorithms,
    ``disc_auxiliary_loss`` and ``gen_auxiliary_loss`` are provided to
    customize auxiliary losses, e.g., gradient penalty loss, and discriminator
    shift loss. In addition, ``train_cfg`` and ``test_cfg`` aims at setuping
    training schedule.

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
        num_classes (int | None, optional): The number of conditional classes.
            Defaults to None.
    """

    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self._gen_cfg = deepcopy(generator)
        self.generator = build_module(
            generator, default_args=dict(num_classes=num_classes))

        # support no discriminator in testing
        if discriminator is not None:
            self.discriminator = build_module(
                discriminator, default_args=dict(num_classes=num_classes))
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
            self.disc_auxiliary_loss = None

        if gen_auxiliary_loss:
            self.gen_auxiliary_losses = build_module(gen_auxiliary_loss)
            if not isinstance(self.gen_auxiliary_losses, nn.ModuleList):
                self.gen_auxiliary_losses = nn.ModuleList(
                    [self.gen_auxiliary_losses])
        else:
            self.gen_auxiliary_losses = None

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parse_test_cfg()

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)
        self.gen_steps = self.train_cfg.get('gen_steps', 1)

        # add support for accumulating gradients within multiple steps. This
        # feature aims to simulate large `batch_sizes` (but may have some
        # detailed differences in BN). Note that `self.disc_steps` should be
        # set according to the batch accumulation strategy.
        # In addition, in the detailed implementation, there is a difference
        # between the batch accumulation in the generator and discriminator.
        self.batch_accumulation_steps = self.train_cfg.get(
            'batch_accumulation_steps', 1)

        # whether to use exponential moving average for training
        self.use_ema = self.train_cfg.get('use_ema', False)
        if self.use_ema:
            # use deepcopy to guarantee the consistency
            self.generator_ema = deepcopy(self.generator)

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

        # whether to use exponential moving average for testing
        self.use_ema = self.test_cfg.get('use_ema', False)
        # TODO: finish ema part

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
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
            loss_scaler (:obj:`torch.cuda.amp.GradScaler` | None, optional):
                The loss/gradient scaler used for auto mixed-precision
                training. Defaults to ``None``.
            use_apex_amp (bool, optional). Whether to use apex.amp. Defaults to
                ``False``.
            running_status (dict | None, optional): Contains necessary basic
                information for training, e.g., iteration number. Defaults to
                None.

        Returns:
            dict: Contains 'log_vars', 'num_samples', and 'results'.
        """
        # get data from data_batch
        real_imgs = data_batch['img']
        # get the ground-truth label, torch.Tensor (N, )
        gt_label = data_batch['gt_label']
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

        # disc training
        set_requires_grad(self.discriminator, True)

        # do not `zero_grad` during batch accumulation
        if curr_iter % self.batch_accumulation_steps == 0:
            optimizer['discriminator'].zero_grad()
        # TODO: add noise sampler to customize noise sampling
        with torch.no_grad():
            fake_data = self.generator(
                None, num_batches=batch_size, label=None, return_noise=True)
            # fake_label should be in the same data type with the gt_label
            fake_imgs, fake_label = fake_data['fake_img'], fake_data['label']

        # disc pred for fake imgs and real_imgs
        disc_pred_fake = self.discriminator(fake_imgs, label=fake_label)
        disc_pred_real = self.discriminator(real_imgs, label=gt_label)
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
            gt_label=gt_label,
            fake_label=fake_label,
            loss_scaler=loss_scaler)

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)
        loss_disc = loss_disc / float(self.batch_accumulation_steps)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))

        if loss_scaler:
            # add support for fp16
            loss_scaler.scale(loss_disc).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(
                    loss_disc, optimizer['discriminator'],
                    loss_id=0) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_disc.backward()

        if (curr_iter + 1) % self.batch_accumulation_steps == 0:
            if loss_scaler:
                loss_scaler.unscale_(optimizer['discriminator'])
                # note that we do not contain clip_grad procedure
                loss_scaler.step(optimizer['discriminator'])
                # loss_scaler.update will be called in runner.train()
            else:
                optimizer['discriminator'].step()

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
        # allow for training the generator with multiple steps
        for _ in range(self.gen_steps):
            optimizer['generator'].zero_grad()
            for _ in range(self.batch_accumulation_steps):
                # TODO: add noise sampler to customize noise sampling
                fake_data = self.generator(
                    None, num_batches=batch_size, return_noise=True)
                # fake_label should be in the same data type with the gt_label
                fake_imgs, fake_label = fake_data['fake_img'], fake_data[
                    'label']
                disc_pred_fake_g = self.discriminator(
                    fake_imgs, label=fake_label)

                data_dict_ = dict(
                    gen=self.generator,
                    disc=self.discriminator,
                    fake_imgs=fake_imgs,
                    disc_pred_fake_g=disc_pred_fake_g,
                    iteration=curr_iter,
                    batch_size=batch_size,
                    fake_label=fake_label,
                    loss_scaler=loss_scaler)

                loss_gen, log_vars_g = self._get_gen_loss(data_dict_)
                loss_gen = loss_gen / float(self.batch_accumulation_steps)

                # prepare for backward in ddp. If you do not call this function
                # before back propagation, the ddp will not dynamically find
                # the used params in current computation.
                if ddp_reducer is not None:
                    ddp_reducer.prepare_for_backward(_find_tensors(loss_gen))

                if loss_scaler:
                    loss_scaler.scale(loss_gen).backward()
                elif use_apex_amp:
                    from apex import amp
                    with amp.scale_loss(
                            loss_gen, optimizer['generator'],
                            loss_id=1) as scaled_loss_disc:
                        scaled_loss_disc.backward()
                else:
                    loss_gen.backward()

            if loss_scaler:
                loss_scaler.unscale_(optimizer['generator'])
                # note that we do not contain clip_grad procedure
                loss_scaler.step(optimizer['generator'])
                # loss_scaler.update will be called in runner.train()
            else:
                optimizer['generator'].step()

        log_vars = {}
        log_vars.update(log_vars_g)
        log_vars.update(log_vars_disc)

        results = dict(fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs

    def sample_from_noise(self,
                          noise,
                          num_batches=0,
                          sample_model='ema/orig',
                          label=None,
                          **kwargs):
        """Sample images from noises by using the generator.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            sampel_model (str, optional): Use which model to sample fake
                images. Defaults to `'ema/orig'`.
            label (torch.Tensor | None , optional): The conditional label.
                Defaults to None.

        Returns:
            torch.Tensor | dict: The output may be the direct synthesized
                images in ``torch.Tensor``. Otherwise, a dict with queried
                data, including generated images, will be returned.
        """
        if sample_model == 'ema':
            assert self.use_ema
            _model = self.generator_ema
        elif sample_model == 'ema/orig' and self.use_ema:
            _model = self.generator_ema
        else:
            _model = self.generator

        outputs = _model(noise, num_batches=num_batches, label=label, **kwargs)

        if isinstance(outputs, dict) and 'noise_batch' in outputs:
            noise = outputs['noise_batch']
            label = outputs['label']

        if sample_model == 'ema/orig' and self.use_ema:
            _model = self.generator
            outputs_ = _model(
                noise, num_batches=num_batches, label=label, **kwargs)

            if isinstance(outputs_, dict):
                outputs['fake_img'] = torch.cat(
                    [outputs['fake_img'], outputs_['fake_img']], dim=0)
            else:
                outputs = torch.cat([outputs, outputs_], dim=0)

        return outputs
