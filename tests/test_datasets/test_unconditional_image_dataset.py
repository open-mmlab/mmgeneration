# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmgen.datasets import UnconditionalImageDataset
from mmgen.utils import register_all_modules

register_all_modules()


class TestUnconditionalImageDataset(object):

    @classmethod
    def setup_class(cls):
        cls.imgs_root = osp.join(osp.dirname(__file__), '..', 'data/image')
        cls.default_pipeline = [
            dict(type='LoadImageFromFile', io_backend='disk', key='img')
        ]

    def test_unconditional_imgs_dataset(self):
        dataset = UnconditionalImageDataset(
            self.imgs_root, pipeline=self.default_pipeline)
        assert len(dataset) == 6
        img = dataset[2]['img']
        assert img.ndim == 3
        assert repr(dataset) == (
            f'dataset_name: {dataset.__class__}, '
            f'total {6} images in data_root: {self.imgs_root}')
