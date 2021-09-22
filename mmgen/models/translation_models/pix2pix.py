# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn.parallel.distributed import _find_tensors

from mmgen.models.builder import MODELS
from ..common import set_requires_grad
from .static_translation_gan import StaticTranslationGAN


@MODELS.register_module()
class Pix2Pix(StaticTranslationGAN):
    """Pix2Pix model for paired image-to-image translation.

    Ref:
     Image-to-Image Translation with Conditional Adversarial Networks
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_ema = False

    def forward_test(self, img, target_domain, **kwargs):
        """Forward function for testing.

        Args:
            img (tensor): Input image tensor.
            target_domain (str): Target domain of output image.
            kwargs (dict): Other arguments.

        Returns:
            dict: Forward results.
        """
        # This is a trick for Pix2Pix
        # ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e1bdf46198662b0f4d0b318e24568205ec4d7aee/test.py#L54  # noqa
        self.train()
        target = self.translation(img, target_domain=target_domain, **kwargs)
        results = dict(source=img.cpu(), target=target.cpu())
        return results

    def _get_disc_loss(self, outputs):
        # GAN loss for the discriminator
        losses = dict()

        discriminators = self.get_module(self.discriminators)
        target_domain = self._default_domain
        source_domain = self.get_other_domains(target_domain)[0]
        fake_ab = torch.cat((outputs[f'real_{source_domain}'],
                             outputs[f'fake_{target_domain}']), 1)
        fake_pred = discriminators[target_domain](fake_ab.detach())
        losses['loss_gan_d_fake'] = self.gan_loss(
            fake_pred, target_is_real=False, is_disc=True)
        real_ab = torch.cat((outputs[f'real_{source_domain}'],
                             outputs[f'real_{target_domain}']), 1)
        real_pred = discriminators[target_domain](real_ab)
        losses['loss_gan_d_real'] = self.gan_loss(
            real_pred, target_is_real=True, is_disc=True)

        loss_d, log_vars_d = self._parse_losses(losses)
        loss_d *= 0.5

        return loss_d, log_vars_d

    def _get_gen_loss(self, outputs):
        target_domain = self._default_domain
        source_domain = self.get_other_domains(target_domain)[0]
        losses = dict()

        discriminators = self.get_module(self.discriminators)
        # GAN loss for the generator
        fake_ab = torch.cat((outputs[f'real_{source_domain}'],
                             outputs[f'fake_{target_domain}']), 1)
        fake_pred = discriminators[target_domain](fake_ab)
        losses['loss_gan_g'] = self.gan_loss(
            fake_pred, target_is_real=True, is_disc=False)

        # gen auxiliary loss
        if self.with_gen_auxiliary_loss:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(outputs)
                if loss_ is None:
                    continue
                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses:
                    losses[loss_module.loss_name(
                    )] = losses[loss_module.loss_name()] + loss_
                else:
                    losses[loss_module.loss_name()] = loss_

        loss_g, log_vars_g = self._parse_losses(losses)
        return loss_g, log_vars_g

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   running_status=None):
        """Training step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            optimizer (dict[torch.optim.Optimizer]): Dict of optimizers for
                the generator and discriminator.
            ddp_reducer (:obj:`Reducer` | None, optional): Reducer from ddp.
                It is used to prepare for ``backward()`` in ddp. Defaults to
                None.
            running_status (dict | None, optional): Contains necessary basic
                information for training, e.g., iteration number. Defaults to
                None.

        Returns:
            dict: Dict of loss, information for logger, the number of samples\
                and results for visualization.
        """
        # data
        target_domain = self._default_domain
        source_domain = self.get_other_domains(self._default_domain)[0]
        source_image = data_batch[f'img_{source_domain}']
        target_image = data_batch[f'img_{target_domain}']

        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        # forward generator
        outputs = dict()
        results = self(
            source_image, target_domain=self._default_domain, test_mode=False)
        outputs[f'real_{source_domain}'] = results['source']
        outputs[f'fake_{target_domain}'] = results['target']
        outputs[f'real_{target_domain}'] = target_image
        log_vars = dict()

        # discriminator
        set_requires_grad(self.discriminators, True)
        # optimize
        optimizer['discriminators'].zero_grad()
        loss_d, log_vars_d = self._get_disc_loss(outputs)
        log_vars.update(log_vars_d)
        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_d))
        loss_d.backward()
        optimizer['discriminators'].step()

        # generator, no updates to discriminator parameters.
        if (curr_iter % self.disc_steps == 0
                and curr_iter >= self.disc_init_steps):
            set_requires_grad(self.discriminators, False)
            # optimize
            optimizer['generators'].zero_grad()
            loss_g, log_vars_g = self._get_gen_loss(outputs)
            log_vars.update(log_vars_g)
            # prepare for backward in ddp. If you do not call this function
            # before back propagation, the ddp will not dynamically find the
            # used params in current computation.
            if ddp_reducer is not None:
                ddp_reducer.prepare_for_backward(_find_tensors(loss_g))
            loss_g.backward()
            optimizer['generators'].step()

        if hasattr(self, 'iteration'):
            self.iteration += 1

        image_results = dict()
        image_results[f'real_{source_domain}'] = outputs[
            f'real_{source_domain}'].cpu()
        image_results[f'fake_{target_domain}'] = outputs[
            f'fake_{target_domain}'].cpu()
        image_results[f'real_{target_domain}'] = outputs[
            f'real_{target_domain}'].cpu()

        results = dict(
            log_vars=log_vars,
            num_samples=len(outputs[f'real_{source_domain}']),
            results=image_results)

        return results
