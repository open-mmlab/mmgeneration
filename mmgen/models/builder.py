# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

MODELS = Registry('model')
MODULES = Registry('module')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.
    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.ModuleList(modules)

    return build_from_cfg(cfg, registry, default_args)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model (GAN)."""
    return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_module(cfg, default_args=None):
    """Build a module or modules from a list."""
    return build(cfg, MODULES, default_args)
