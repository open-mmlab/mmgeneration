# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmengine.optim import DefaultOptimWrapperConstructor, OptimWrapperDict

from mmgen.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class GenOptimWrapperConstructor:
    """OptimizerConstructor for GAN models. This class construct optimizer for
    the submodules (generator and discriminator for GAN most models) of the
    model separately, and return a :class:~`mmengine.optim.OptimWrapperDict`.

    Example:
        >>> # build GAN model
        >>> model = dict(
        >>>     type='SAGAN',
        >>>     num_classes=10,
        >>>     generator=dict(type='SAGANGenerator'),
        >>>     discriminator=dict(type='ProjDiscriminator'))
        >>> gan_model = MODELS.build(model)
        >>> # build constructor
        >>> optim_wrapper = dict(
        >>>     constructor='GANOptimWrapperConstructor',
        >>>     generator=dict(
        >>>         type='OptimWrapper',
        >>>         accumulative_counts=1,
        >>>         optimizer=dict(type='Adam', lr=0.0002,
        >>>                        betas=(0.5, 0.999))),
        >>>     discriminator=dict(
        >>>         optimizer=dict(
        >>>             type='OptimWrapper',
        >>>             accumulative_counts=1,
        >>>             optimizer=dict(type='Adam', lr=0.0002,
        >>>                            betas=(0.5, 0.999)),
        >>>         )))
        >>> optim_wrapper_dict_builder = GenOptimConstructor(optim_wrapper)
        >>> # build optim wrapper dict
        >>> optim_wrapper_dict = optim_wrapper_dict_builder(gan_model)

    Args:
        optim_wrapper_cfg (dict): Config of the optimizer wrapper.
        paramwise_cfg (Optional[dict]): Parameter-wise options.
    """

    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):
        if not isinstance(optim_wrapper_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            f'but got {type(optim_wrapper_cfg)}')
        assert paramwise_cfg is None, (
            'parawise_cfg should be set in each optimizer separately')
        self.optim_cfg = optim_wrapper_cfg
        self.constructors = {}
        for key, cfg in self.optim_cfg.items():
            cfg_ = cfg.copy()
            paramwise_cfg_ = cfg_.pop('paramwise_cfg', None)
            self.constructors[key] = DefaultOptimWrapperConstructor(
                cfg_, paramwise_cfg_)

    def __call__(self, module: nn.Module) -> OptimWrapperDict:
        """Build optimizer and return a optimizerwrapperdict."""
        optimizers = {}
        if hasattr(module, 'module'):
            module = module.module
        for key, constructor in self.constructors.items():
            optimizers[key] = constructor(module._modules[key])
        return OptimWrapperDict(**optimizers)
