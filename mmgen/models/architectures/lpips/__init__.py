# Copyright (c) OpenMMLab. All rights reserved.
r"""
    The lpips module was adapted from https://github.com/rosinality/stylegan2-pytorch/tree/master/lpips ,  # noqa
    and you can see the origin implementation in https://github.com/richzhang/PerceptualSimilarity/tree/master/lpips  # noqa
"""
from .perceptual_loss import PerceptualLoss

__all__ = ['PerceptualLoss']
