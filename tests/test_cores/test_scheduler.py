import logging
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, call

import torch
import torch.nn as nn
from mmcv.runner import PaviLoggerHook, build_runner
from torch.utils.data import DataLoader


def _build_demo_runner(runner_type='EpochBasedRunner',
                       max_epochs=1,
                       max_iters=None):

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)

    log_config = dict(
        interval=1, hooks=[
            dict(type='TextLoggerHook'),
        ])

    tmp_dir = tempfile.mkdtemp()
    runner = build_runner(
        dict(type=runner_type),
        default_args=dict(
            model=model,
            work_dir=tmp_dir,
            optimizer=optimizer,
            logger=logging.getLogger(),
            max_epochs=max_epochs,
            max_iters=max_iters))
    runner.register_checkpoint_hook(dict(interval=1))
    runner.register_logger_hooks(log_config)
    return runner


def test_linear_lr_updater_scheduler():
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner()

    # add momentum LR scheduler
    lr_config = dict(
        policy='Linear', by_epoch=False, target_lr=0, start=0, interval=1)
    runner.register_lr_hook(lr_config)
    runner.register_hook_from_cfg(dict(type='IterTimerHook'))

    # add pavi hook
    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)

    # TODO: use a more elegant way to check values
    assert hasattr(hook, 'writer')
    calls = [
        call('train', {
            'learning_rate': 0.018000000000000002,
            'momentum': 0.95
        }, 2),
        call('train', {
            'learning_rate': 0.014,
            'momentum': 0.95
        }, 4),
        call('train', {
            'learning_rate': 0.01,
            'momentum': 0.95
        }, 6),
    ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)
