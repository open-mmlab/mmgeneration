# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmgen.datasets.builder import build_dataloader, build_dataset


class TestPersistentWorker(object):

    @classmethod
    def setup_class(cls):
        imgs_root = osp.join(osp.dirname(__file__), '..', 'data/image')
        train_pipeline = [
            dict(type='LoadImageFromFile', io_backend='disk', key='real_img')
        ]
        cls.config = dict(
            samples_per_gpu=1,
            workers_per_gpu=4,
            drop_last=True,
            persistent_workers=True)

        cls.data_cfg = dict(
            type='UnconditionalImageDataset',
            imgs_root=imgs_root,
            pipeline=train_pipeline,
            test_mode=False)

    def test_persistent_worker(self):
        # test non-persistent-worker
        dataset = build_dataset(self.data_cfg)
        build_dataloader(dataset, **self.config)
