# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmgen.datasets import UnpairedImageDataset


class TestUnpairedImageDataset(object):

    @classmethod
    def setup_class(cls):
        cls.imgs_root = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/unpaired')
        cls.default_pipeline = [
            dict(
                type='mmgen.LoadImageFromFile',
                io_backend='disk',
                key='img_a',
                flag='color'),
            dict(
                type='mmgen.LoadImageFromFile',
                io_backend='disk',
                key='img_b',
                flag='color'),
            dict(
                type='TransformBroadcaster',
                mapping={'img': ['img_a', 'img_b']},
                auto_remap=True,
                share_random_params=True,
                transforms=[
                    dict(
                        type='mmgen.Resize',
                        scale=(286, 286),
                        interpolation='bicubic')
                ]),
            dict(
                type='mmgen.Crop',
                keys=['img_a', 'img_b'],
                crop_size=(256, 256),
                random_crop=True),
            dict(
                type='mmgen.Flip',
                direction='horizontal',
                keys=['img_a', 'img_b']),
            dict(
                type='mmgen.PackGenInputs',
                keys=['img_a', 'img_b'],
                meta_keys=['img_a_path', 'img_b_path']),
        ]

    def test_unpaired_image_dataset(self):
        dataset = UnpairedImageDataset(
            self.imgs_root,
            pipeline=self.default_pipeline,
            domain_a='a',
            domain_b='b')
        assert len(dataset) == 2
        img = dataset[0]['inputs']['img_a']
        assert img.ndim == 3
        img = dataset[0]['inputs']['img_b']
        assert img.ndim == 3
