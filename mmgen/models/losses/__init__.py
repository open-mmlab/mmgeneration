# Copyright (c) OpenMMLab. All rights reserved.
from .ddpm_loss import DDPMVLBLoss
from .disc_auxiliary_loss import (DiscShiftLoss, GradientPenaltyLoss,
                                  R1GradientPenalty, disc_shift_loss,
                                  gradient_penalty_loss,
                                  r1_gradient_penalty_loss)
from .gan_loss import GANLoss
from .gen_auxiliary_loss import (CLIPLoss, FaceIdLoss,
                                 GeneratorPathRegularizer,
                                 gen_path_regularizer)
from .pixelwise_loss import (DiscretizedGaussianLogLikelihoodLoss,
                             GaussianKLDLoss, L1Loss, MSELoss,
                             discretized_gaussian_log_likelihood, gaussian_kld)

__all__ = [
    'GANLoss', 'DiscShiftLoss', 'disc_shift_loss', 'gradient_penalty_loss',
    'GradientPenaltyLoss', 'R1GradientPenalty', 'r1_gradient_penalty_loss',
    'GeneratorPathRegularizer', 'gen_path_regularizer', 'MSELoss', 'L1Loss',
    'gaussian_kld', 'GaussianKLDLoss', 'DiscretizedGaussianLogLikelihoodLoss',
    'DDPMVLBLoss', 'discretized_gaussian_log_likelihood', 'FaceIdLoss',
    'CLIPLoss'
]
