import torch
from torch.nn.parallel.distributed import _find_tensors

from mmgen.models.builder import MODELS
from ..common import set_requires_grad
from .base_translation_model import BaseTranslationModel


@MODELS.register_module()
class Pix2Pix(BaseTranslationModel):

    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 default_style,
                 reachable_styles=[],
                 related_styles=[],
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(
            generator,
            discriminator,
            gan_loss,
            default_style,
            reachable_styles=reachable_styles,
            related_styles=related_styles,
            disc_auxiliary_loss=disc_auxiliary_loss,
            gen_auxiliary_loss=gen_auxiliary_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

        self.init_weights(pretrained)
        self.use_ema = False

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        for style in self._reachable_styles:
            self.generators[style].init_weights(pretrained=pretrained)
            self.discriminators[style].init_weights(pretrained=pretrained)

    def _get_disc_loss(self, outputs):
        # GAN loss for the discriminator
        losses = dict()
        # conditional GAN
        dst_style = self._default_style
        src_style = self._get_opposite_style(dst_style)
        fake_ab = torch.cat(
            (outputs[f'src_{src_style}'], outputs[f'style_{dst_style}']), 1)
        fake_pred = self.discriminators[dst_style](fake_ab.detach())
        losses['loss_gan_d_fake'] = self.gan_loss(
            fake_pred, target_is_real=False, is_disc=True)
        real_ab = torch.cat(
            (outputs[f'src_{src_style}'], outputs[f'src_{dst_style}']), 1)
        real_pred = self.discriminators[dst_style](real_ab)
        losses['loss_gan_d_real'] = self.gan_loss(
            real_pred, target_is_real=True, is_disc=True)

        loss_d, log_vars_d = self._parse_losses(losses)
        loss_d *= 0.5

        return loss_d, log_vars_d

    def _get_gen_loss(self, outputs):
        dst_style = self._default_style
        src_style = self._get_opposite_style(dst_style)
        losses = dict()
        # GAN loss for the generator
        fake_ab = torch.cat(
            (outputs[f'src_{src_style}'], outputs[f'style_{dst_style}']), 1)
        fake_pred = self.discriminators[dst_style](fake_ab)
        losses['loss_gan_g'] = self.gan_loss(
            fake_pred, target_is_real=True, is_disc=False)

        # gen auxiliary loss
        if self.with_gen_auxiliary_loss:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(outputs)
                if loss_ is None:
                    continue

                # mmcv.print_log(f'get loss for {loss_module.name()}')
                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses:
                    losses[loss_module.loss_name(
                    )] = losses[loss_module.loss_name()] + loss_
                else:
                    losses[loss_module.loss_name()] = loss_

        loss_g, log_vars_g = self._parse_losses(losses)
        return loss_g, log_vars_g

    def _get_opposite_style(self, style):
        for item in self._related_styles:
            if item != style:
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
                the generator and discriminator.

        Returns:
            dict: Dict of loss, information for logger, the number of samples\
                and results for visualization.
        """
        # data
        src_style = self._get_opposite_style(self._default_style)
        dst_style = self._default_style
        src_img = data_batch[f'img_{src_style}']
        dst_img = data_batch[f'img_{dst_style}']

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
        results = self.forward(
            src_img, style=self._default_style, test_mode=False)
        outputs[f'src_{src_style}'] = results['orig_img']
        outputs[f'style_{dst_style}'] = results['styled_img']
        outputs[f'src_{dst_style}'] = dst_img
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
        if ((curr_iter + 1) % self.disc_steps == 0
                and (curr_iter + 1) >= self.disc_init_steps):
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

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        results = dict(
            log_vars=log_vars,
            num_samples=len(outputs[f'src_{src_style}']),
            results=dict(
                real_a=outputs[f'src_{src_style}'].cpu(),
                fake_b=outputs[f'style_{dst_style}'].cpu(),
                real_b=outputs[f'src_{dst_style}'].cpu()))

        return results
