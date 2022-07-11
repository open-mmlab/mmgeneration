# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import BaseAveragedModel
from torch import Tensor

from mmgen.registry import MODELS


@MODELS.register_module()
class RampUpEMA(BaseAveragedModel):
    r"""Implements the exponential moving average with ramping up momentum.

    Ref: https://github.com/NVlabs/stylegan3/blob/master/training/training_loop.py # noqa

    Args:
        model (nn.Module): The model to be averaged.
        interval (int): Interval between two updates. Defaults to 1.
        ema_kimg (int, optional): EMA kimgs. Defaults to 10.
        ema_rampup (float, optional): Ramp up rate. Defaults to 0.05.
        batch_size (int, optional): Global batch size. Defaults to 32.
        eps (float, optional): Ramp up epsilon. Defaults to 1e-8.
        start_iter (int, optional): EMA start iter. Defaults to 0.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """  # noqa: W605

    def __init__(self,
                 model: nn.Module,
                 interval: int = 1,
                 ema_kimg: int = 10,
                 ema_rampup: float = 0.05,
                 batch_size: int = 32,
                 eps: float = 1e-8,
                 start_iter: int = 0,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        """_summary_"""
        super().__init__(model, interval, device, update_buffers)
        self.interval = interval
        self.ema_kimg = ema_kimg
        self.ema_rampup = ema_rampup
        self.batch_size = batch_size
        self.eps = eps

    @staticmethod
    def rampup(steps, ema_kimg=10, ema_rampup=0.05, batch_size=4, eps=1e-8):
        """Ramp up ema momentum.

        Ref: https://github.com/NVlabs/stylegan3/blob/a5a69f58294509598714d1e88c9646c3d7c6ec94/training/training_loop.py#L300-L308 # noqa

        Args:
            steps:
            ema_kimg (int, optional): Half-life of the exponential moving
                average of generator weights. Defaults to 10.
            ema_rampup (float, optional): EMA ramp-up coefficient.If set to
                None, then rampup will be disabled. Defaults to 0.05.
            batch_size (int, optional): Total batch size for one training
                iteration. Defaults to 4.
            eps (float, optional): Epsiolon to avoid ``batch_size`` divided by
                zero. Defaults to 1e-8.

        Returns:
            dict: Updated momentum.
        """
        cur_nimg = (steps + 1) * batch_size
        ema_nimg = ema_kimg * 1000
        if ema_rampup is not None:
            ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
        ema_beta = 0.5**(batch_size / max(ema_nimg, eps))
        return ema_beta

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using exponential
        moving average.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = self.rampup(self.steps, self.ema_kimg, self.ema_rampup,
                               self.batch_size, self.eps)
        averaged_param.mul_(1 - momentum).add_(source_param, alpha=momentum)
