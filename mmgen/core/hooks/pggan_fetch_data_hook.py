# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class PGGANFetchDataHook(Hook):
    """PGGAN Fetch Data Hook.

    Args:
        interval (int, optional):  The interval of calling this hook. If set
            to -1, the visualization hook will not be called. Defaults to 1.
    """

    def __init__(self, interval=1):
        super().__init__()
        self.interval = interval

    def before_fetch_train_data(self, runner):
        """The behavior before fetch train data.

        Args:
            runner (object): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        _module = runner.model.module if is_module_wrapper(
            runner.model) else runner.model

        _next_scale_int = _module._next_scale_int
        if isinstance(_next_scale_int, torch.Tensor):
            _next_scale_int = _next_scale_int.item()
        runner.data_loader.update_dataloader(_next_scale_int)
