import os.path as osp
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from mmengine.data import DefaultSampler, pseudo_collate
from mmengine.runner import IterBasedTrainLoop
from torch.utils.data.dataloader import DataLoader

from mmgen.core import PGGANFetchDataHook
from mmgen.registry import DATASETS, MODELS
from mmgen.utils import register_all_modules

register_all_modules()


class TestPGGANFetchDataHook(TestCase):

    pggan_cfg = dict(
        type='ProgressiveGrowingGAN',
        data_preprocessor=dict(type='GANDataPreprocessor'),
        noise_size=512,
        generator=dict(type='PGGANGenerator', out_scale=8),
        discriminator=dict(type='PGGANDiscriminator', in_scale=8),
        nkimgs_per_scale={
            '4': 600,
            '8': 1200,
            '16': 1200,
            '32': 1200,
            '64': 1200,
            '128': 12000
        },
        transition_kimgs=600,
        ema_config=dict(interval=1))

    imgs_root = osp.join(osp.dirname(__file__), '..', '..', 'data/image')
    grow_scale_dataset_cfg = dict(
        type='GrowScaleImgDataset',
        data_roots={
            '4': imgs_root,
            '8': osp.join(imgs_root, 'img_root'),
            '32': osp.join(imgs_root, 'img_root', 'grass')
        },
        gpu_samples_base=4,
        gpu_samples_per_scale={
            '4': 64,
            '8': 32,
            '16': 16,
            '32': 8,
            '64': 4
        },
        len_per_stage=10,
        pipeline=[
            dict(type='LoadImageFromFile', io_backend='disk', key='img')
        ])

    def test_before_train_iter(self):
        runner = MagicMock()
        model = MODELS.build(self.pggan_cfg)
        dataset = DATASETS.build(self.grow_scale_dataset_cfg)
        dataloader = DataLoader(
            batch_size=64,
            dataset=dataset,
            sampler=DefaultSampler,
            collate_fn=pseudo_collate)

        runner.train_loop = MagicMock(spec=IterBasedTrainLoop)
        runner.train_loop.dataloader = dataloader
        runner.model = model

        hooks = PGGANFetchDataHook()
        hooks.before_train_iter(runner, 0, None)

        for scale, target_bz in self.grow_scale_dataset_cfg[
                'gpu_samples_per_scale'].items():

            model._next_scale_int = torch.tensor(int(scale), dtype=torch.int32)
            hooks.before_train_iter(runner, 0, None)
            self.assertEqual(runner.train_loop.dataloader.batch_size,
                             target_bz)

        # set `_next_scale_int` as int
        delattr(model, '_next_scale_int')
        setattr(model, '_next_scale_int', 128)
        hooks.before_train_iter(runner, 0, None)
        self.assertEqual(runner.train_loop.dataloader.batch_size, 4)

        # test do not update
        hooks.before_train_iter(runner, 1, None)

        # test not `IterBasedTrainLoop`
        runner.train_loop = MagicMock()
        runner.train_loop.dataloader = dataloader
        runner.model = model
        hooks.before_train_iter(runner, 0, None)
