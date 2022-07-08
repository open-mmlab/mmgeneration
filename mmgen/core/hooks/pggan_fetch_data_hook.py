# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import torch
from mmengine.data import pseudo_collate
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import IterBasedTrainLoop
from mmengine.runner.loops import _InfiniteDataloaderIterator
from torch.utils.data.dataloader import DataLoader

from mmgen.registry import HOOKS

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class PGGANFetchDataHook(Hook):
    """PGGAN Fetch Data Hook.

    Args:
        interval (int, optional):  The interval of calling this hook. If set
            to -1, the visualization hook will not be called. Defaults to 1.
    """

    def __init__(self):
        super().__init__()

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:

        _module = runner.model.module if is_model_wrapper(
            runner.model) else runner.model
        _next_scale_int = _module._next_scale_int
        if isinstance(_next_scale_int, torch.Tensor):
            _next_scale_int = _next_scale_int.item()

        dataloader_orig = runner.train_loop.dataloader
        dataloader = self.update_dataloader(dataloader_orig, _next_scale_int)
        runner.train_loop.dataloader = dataloader
        if isinstance(runner.train_loop, IterBasedTrainLoop):
            runner.train_loop.dataloader_iterator = \
                _InfiniteDataloaderIterator(dataloader)

    def update_dataloader(self, dataloader: DataLoader, curr_scale: int):

        if hasattr(dataloader.dataset, 'update_annotations'):
            update_flag = dataloader.dataset.update_annotations(curr_scale)
        else:
            update_flag = False

        if update_flag:
            dataset = dataloader.dataset
            sampler = dataloader.sampler
            num_workers = dataloader.num_workers
            worker_init_fn = dataloader.worker_init_fn

            dataloader = DataLoader(
                dataset,
                batch_size=dataloader.dataset.samples_per_gpu,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=pseudo_collate,
                shuffle=False,
                worker_init_fn=worker_init_fn)
        return dataloader
