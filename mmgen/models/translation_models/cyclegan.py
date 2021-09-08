# Copyright (c) OpenMMLab. All rights reserved.
from torch.nn.parallel.distributed import _find_tensors

from mmgen.models.builder import MODELS
from ..common import GANImageBuffer, set_requires_grad
from .static_translation_gan import StaticTranslationGAN


@MODELS.register_module()
class CycleGAN(StaticTranslationGAN):
    """CycleGAN model for unpaired image-to-image translation.

    Ref:
    Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
    Networks
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GAN image buffers
        self.image_buffers = dict()
        self.buffer_size = (50 if self.train_cfg is None else
                            self.train_cfg.get('buffer_size', 50))
        for domain in self._reachable_domains:
            self.image_buffers[domain] = GANImageBuffer(self.buffer_size)

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
        # This is a trick for CycleGAN
        # ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e1bdf46198662b0f4d0b318e24568205ec4d7aee/test.py#L54 # noqa
        self.train()
        target = self.translation(img, target_domain=target_domain, **kwargs)
        results = dict(source=img.cpu(), target=target.cpu())
        return results

    def _get_disc_loss(self, outputs):
        """Backward function for the discriminators.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Discriminators' loss and loss dict.
        """
        discriminators = self.get_module(self.discriminators)

        log_vars_d = dict()
        loss_d = 0

        # GAN loss for discriminators['a']
        for domain in self._reachable_domains:
            losses = dict()
            fake_img = self.image_buffers[domain].query(
                outputs[f'fake_{domain}'])
            fake_pred = discriminators[domain](fake_img.detach())
            losses[f'loss_gan_d_{domain}_fake'] = self.gan_loss(
                fake_pred, target_is_real=False, is_disc=True)
            real_pred = discriminators[domain](outputs[f'real_{domain}'])
            losses[f'loss_gan_d_{domain}_real'] = self.gan_loss(
                real_pred, target_is_real=True, is_disc=True)

            _loss_d, _log_vars_d = self._parse_losses(losses)
            _loss_d *= 0.5
            loss_d += _loss_d
            log_vars_d[f'loss_gan_d_{domain}'] = _log_vars_d['loss'] * 0.5

        return loss_d, log_vars_d

    def _get_gen_loss(self, outputs):
        """Backward function for the generators.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Generators' loss and loss dict.
        """
        generators = self.get_module(self.generators)
        discriminators = self.get_module(self.discriminators)

        losses = dict()
        for domain in self._reachable_domains:
            # Identity reconstruction for generators
            outputs[f'identity_{domain}'] = generators[domain](
                outputs[f'real_{domain}'])
            # GAN loss for generators
            fake_pred = discriminators[domain](outputs[f'fake_{domain}'])
            losses[f'loss_gan_g_{domain}'] = self.gan_loss(
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

    def _get_opposite_domain(self, domain):
        for item in self._reachable_domains:
            if item != domain:
                return item
        return None

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   running_status=None):
        """Training step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            optimizer (dict[torch.optim.Optimizer]): Dict of optimizers for
                the generators and discriminators.
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
        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        # forward generators
        outputs = dict()
        for target_domain in self._reachable_domains:
            # fetch data by domain
            source_domain = self.get_other_domains(target_domain)[0]
            img = data_batch[f'img_{source_domain}']
            # translation process
            results = self(img, test_mode=False, target_domain=target_domain)
            outputs[f'real_{source_domain}'] = results['source']
            outputs[f'fake_{target_domain}'] = results['target']
            # cycle process
            results = self(
                results['target'],
                test_mode=False,
                target_domain=source_domain)
            outputs[f'cycle_{source_domain}'] = results['target']

        log_vars = dict()

        # discriminators
        set_requires_grad(self.discriminators, True)
        # optimize
        optimizer['discriminators'].zero_grad()
        loss_d, log_vars_d = self._get_disc_loss(outputs)
        log_vars.update(log_vars_d)
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_d))
        loss_d.backward()
        optimizer['discriminators'].step()

        # generators, no updates to discriminator parameters.
        if (curr_iter % self.disc_steps == 0
                and curr_iter >= self.disc_init_steps):
            set_requires_grad(self.discriminators, False)
            # optimize
            optimizer['generators'].zero_grad()
            loss_g, log_vars_g = self._get_gen_loss(outputs)
            log_vars.update(log_vars_g)
            if ddp_reducer is not None:
                ddp_reducer.prepare_for_backward(_find_tensors(loss_g))
            loss_g.backward()
            optimizer['generators'].step()

        if hasattr(self, 'iteration'):
            self.iteration += 1

        image_results = dict()
        for domain in self._reachable_domains:
            image_results[f'real_{domain}'] = outputs[f'real_{domain}'].cpu()
            image_results[f'fake_{domain}'] = outputs[f'fake_{domain}'].cpu()
        results = dict(
            log_vars=log_vars,
            num_samples=len(outputs[f'real_{domain}']),
            results=image_results)

        return results
