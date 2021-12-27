# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
from functools import partial

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import collate, is_module_wrapper
from mmcv.runner import HOOKS, RUNNERS, IterBasedRunner, get_host_info
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    from torch.cuda.amp import GradScaler
except ImportError:
    pass


class IterLoader:
    """Iteration based dataloader.

    This wrapper for dataloader is to matching the iter-based training
    proceduer.

    Args:
        dataloader (object): Dataloader in PyTorch.
        runner (object): ``mmcv.Runner``
    """

    def __init__(self, dataloader, runner):
        self._dataloader = dataloader
        self.runner = runner
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        """The number of current epoch.

        Returns:
            int: Epoch number.
        """
        return self._epoch

    def update_dataloader(self, curr_scale):
        """Update dataloader.

        Update the dataloader according to the `curr_scale`. This functionality
        is very helpful in training progressive growing GANs in which the
        dataloader should be updated according to the scale of the models in
        training.

        Args:
            curr_scale (int): The scale in current stage.
        """
        # update dataset, sampler, and samples per gpu in dataloader
        if hasattr(self._dataloader.dataset, 'update_annotations'):
            update_flag = self._dataloader.dataset.update_annotations(
                curr_scale)
        else:
            update_flag = False
        if update_flag:
            # the sampler should be updated with the modified dataset
            assert hasattr(self._dataloader.sampler, 'update_sampler')
            samples_per_gpu = None if not hasattr(
                self._dataloader.dataset, 'samples_per_gpu'
            ) else self._dataloader.dataset.samples_per_gpu
            self._dataloader.sampler.update_sampler(self._dataloader.dataset,
                                                    samples_per_gpu)
            # update samples per gpu
            if samples_per_gpu is not None:
                if dist.is_initialized():
                    # samples = samples_per_gpu
                    # self._dataloader.collate_fn = partial(
                    #     collate, samples_per_gpu=samples)
                    self._dataloader = DataLoader(
                        self._dataloader.dataset,
                        batch_size=samples_per_gpu,
                        sampler=self._dataloader.sampler,
                        num_workers=self._dataloader.num_workers,
                        collate_fn=partial(
                            collate, samples_per_gpu=samples_per_gpu),
                        shuffle=False,
                        worker_init_fn=self._dataloader.worker_init_fn)

                    self.iter_loader = iter(self._dataloader)
                else:
                    raise NotImplementedError(
                        'Currently, we only support dynamic batch size in'
                        ' ddp, because the number of gpus in DataParallel '
                        'cannot be obtained easily.')

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


@RUNNERS.register_module()
class DynamicIterBasedRunner(IterBasedRunner):
    """Dynamic Iterbased Runner.

    In this Dynamic Iterbased Runner, we will pass the ``reducer`` to the
    ``train_step`` so that the models can be trained with dynamic architecture.
    More details and clarification can be found in this [tutorial](docs/en/tutorials/ddp_train_gans.md).  # noqa

    Args:
        is_dynamic_ddp (bool, optional): Whether to adopt the dynamic ddp.
            Defaults to False.
        pass_training_status (bool, optional): Whether to pass the training
            status. Defaults to False.
        fp16_loss_scaler (dict | None, optional): Config for fp16 GradScaler
            from ``torch.cuda.amp``. Defaults to None.
        use_apex_amp (bool, optional): Whether to use apex.amp to start mixed
            precision training. Defaults to False.
    """

    def __init__(self,
                 *args,
                 is_dynamic_ddp=False,
                 pass_training_status=False,
                 fp16_loss_scaler=None,
                 use_apex_amp=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if is_module_wrapper(self.model):
            _model = self.model.module
        else:
            _model = self.model

        self.is_dynamic_ddp = is_dynamic_ddp
        self.pass_training_status = pass_training_status

        # add a flag for checking if `self.optimizer` comes from `_model`
        self.optimizer_from_model = False
        # add support for optimizer is None.
        # sanity check for whether `_model` contains self-defined optimizer
        if hasattr(_model, 'optimizer'):
            assert self.optimizer is None, (
                'Runner and model cannot contain optimizer at the same time.')
            self.optimizer_from_model = True
            self.optimizer = _model.optimizer

        # add fp16 grad scaler, using pytorch official GradScaler
        self.with_fp16_grad_scaler = False
        if fp16_loss_scaler is not None:
            self.loss_scaler = GradScaler(**fp16_loss_scaler)
            self.with_fp16_grad_scaler = True
            mmcv.print_log('Use FP16 grad scaler in Training', 'mmgen')

        # flag to use amp in apex (NVIDIA)
        self.use_apex_amp = use_apex_amp

    def call_hook(self, fn_name):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self)

    def train(self, data_loader, **kwargs):
        if is_module_wrapper(self.model):
            _model = self.model.module
        else:
            _model = self.model
        self.model.train()
        self.mode = 'train'
        # check if self.optimizer from model and track it
        if self.optimizer_from_model:
            self.optimizer = _model.optimizer

        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        self.call_hook('before_fetch_train_data')
        data_batch = next(self.data_loader)
        self.call_hook('before_train_iter')

        # prepare input args for train_step
        # running status
        if self.pass_training_status:
            running_status = dict(iteration=self.iter, epoch=self.epoch)
            kwargs['running_status'] = running_status
        # ddp reducer for tracking dynamic computational graph
        if self.is_dynamic_ddp:
            kwargs.update(dict(ddp_reducer=self.model.reducer))

        if self.with_fp16_grad_scaler:
            kwargs.update(dict(loss_scaler=self.loss_scaler))

        if self.use_apex_amp:
            kwargs.update(dict(use_apex_amp=True))

        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)

        # the loss scaler should be updated after ``train_step``
        if self.with_fp16_grad_scaler:
            self.loss_scaler.update()

        # further check for the cases where the optimizer is built in
        # `train_step`.
        if self.optimizer is None:
            if hasattr(_model, 'optimizer'):
                self.optimizer_from_model = True
                self.optimizer = _model.optimizer

        # check if self.optimizer from model and track it
        if self.optimizer_from_model:
            self.optimizer = _model.optimizer
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x, self) for x in data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               resume_loss_scaler=True,
               map_location='default'):
        """Resume model from checkpoint.

        Args:
            checkpoint (str): Checkpoint to resume from.
            resume_optimizer (bool, optional): Whether resume the optimizer(s)
                if the checkpoint file includes optimizer(s). Default to True.
            resume_loss_scaler (bool, optional): Whether to resume the loss
                scaler (GradScaler) from ``torch.cuda.amp``. Defaults to True.
            map_location (str, optional): Same as :func:`torch.load`.
                Default to 'default'.
        """
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        self._inner_iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        if 'loss_scaler' in checkpoint and resume_loss_scaler:
            self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])

        if self.use_apex_amp:
            from apex import amp
            amp.load_state_dict(checkpoint['amp'])

        self.logger.info(f'resumed from epoch: {self.epoch}, iter {self.iter}')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='iter_{}.pth',
                        meta=None,
                        save_optimizer=True,
                        create_symlink=True):
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'iter_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
        """
        if meta is None:
            meta = dict(iter=self.iter + 1, epoch=self.epoch + 1)
        elif isinstance(meta, dict):
            meta.update(iter=self.iter + 1, epoch=self.epoch + 1)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        _loss_scaler = self.loss_scaler if self.with_fp16_grad_scaler else None
        save_checkpoint(
            self.model,
            filepath,
            optimizer=optimizer,
            loss_scaler=_loss_scaler,
            save_apex_amp=self.use_apex_amp,
            meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    def register_lr_hook(self, lr_config):
        if lr_config is None:
            return

        if isinstance(lr_config, dict):
            assert 'policy' in lr_config
            policy_type = lr_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of Lr updater.
            # Since this is not applicable for `
            # CosineAnnealingLrUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            hook = mmcv.build_from_cfg(lr_config, HOOKS)
        else:
            hook = lr_config
        self.register_hook(hook)
