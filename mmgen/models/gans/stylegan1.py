# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmgen.registry import MODELS
from mmgen.typing import NoiseVar
from .progressive_growing_unconditional_gan import ProgressiveGrowingGAN

ModelType = Union[Dict, nn.Module]
TrainInput = Union[dict, Tensor]


@MODELS.register_module('StyleGANV1')
@MODELS.register_module()
class StyleGANv1(ProgressiveGrowingGAN):

    def __init__(self,
                 generator,
                 discriminator,
                 data_preprocessor,
                 nkimgs_per_scale,
                 interp_real=None,
                 transition_kimgs: int = 600,
                 prev_stage: int = 0,
                 ema_config: Optional[Dict] = None):
        super().__init__(generator, discriminator, data_preprocessor, None,
                         nkimgs_per_scale, interp_real, transition_kimgs,
                         prev_stage, ema_config)

    def noise_fn(self, noise: NoiseVar = None, num_batches: int = 1):
        return super().noise_fn(noise, num_batches)

    def disc_loss(self, disc_pred_fake: Tensor, disc_pred_real: Tensor,
                  fake_data: Tensor, real_data: Tensor) -> Tuple[Tensor, dict]:
        r"""Get disc loss. StyleGANv1 use non-saturating gan loss and R1
        gradient penalty. loss to train the discriminator.

        .. math:
            L_{D} = \mathbb{E}_{z\sim{p_{z}}}D\left\(G\left\(z\right\)\right\)
                - \mathbb{E}_{x\sim{p_{data}}}D\left\(x\right\) + L_{GP} \\
            L_{GP} = \lambda\mathbb{E}(\Vert\nabla_{\tilde{x}}D(\tilde{x})
                \Vert_2-1)^2 \\
            \tilde{x} = \epsilon x + (1-\epsilon)G(z)
            L_{shift} =

        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.
            disc_pred_real (Tensor): Discriminator's prediction of the real
                images.
            fake_data (Tensor): Generated images, used to calculate gradient
                penalty.
            real_data (Tensor): Real images, used to calculate gradient
                penalty.

        Returns:
            Tuple[Tensor, dict]: Loss value and a dict of log variables.
        """

        losses_dict = dict()
        losses_dict['loss_disc_fake'] = F.softplus(disc_pred_fake).mean()
        losses_dict['loss_disc_real'] = F.softplus(disc_pred_real).mean()

        # R1 gradient penalty
        batch_size = real_data.size(0)
        real_data_ = real_data.clone().requires_grad_()
        disc_pred = self.discriminator(real_data_)
        gradients = autograd.grad(
            outputs=disc_pred,
            inputs=real_data_,
            grad_outputs=torch.ones_like(disc_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        # norm_mode is 'HWC'
        gradients_penalty = gradients.pow(2).reshape(batch_size,
                                                     -1).sum(1).mean()
        losses_dict['loss_r1_gp'] = 10 * gradients_penalty

        parsed_loss, log_vars = self.parse_losses(losses_dict)
        return parsed_loss, log_vars

    def gen_loss(self, disc_pred_fake: Tensor) -> Tuple[Tensor, dict]:
        r"""Generator loss for PGGAN. PGGAN use WGAN's loss to train the
        generator.

        .. math:
            L_{G} = -\mathbb{E}_{z\sim{p_{z}}}D\left\(G\left\(z\right\)\right\)
                + L_{MSE}

        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.
            recon_imgs (Tensor): Reconstructive images.

        Returns:
            Tuple[Tensor, dict]: Loss value and a dict of log variables.
        """
        losses_dict = dict()
        losses_dict['loss_gen'] = -disc_pred_fake.mean()
        loss, log_vars = self.parse_losses(losses_dict)
        return loss, log_vars
