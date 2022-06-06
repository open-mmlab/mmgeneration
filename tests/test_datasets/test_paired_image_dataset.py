# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmgen.datasets import PairedImageDataset


class TestPairedImageDataset(object):

    @classmethod
    def setup_class(cls):
        cls.imgs_root = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/paired')
        cls.default_pipeline = [
            dict(
                type='mmgen.LoadPairedImageFromFile',
                io_backend='disk',
                key='pair',
                domain_a='a',
                domain_b='b'),
            dict(
                type='TransformBroadcaster',
                mapping={'img': ['img_a', 'img_b']},
                auto_remap=True,
                share_random_params=True,
                transforms=[
                    dict(
                        type='mmgen.Resize',
                        scale=(286, 286),
                        interpolation='bicubic'),
                    dict(type='mmgen.FixedCrop', crop_size=(256, 256))
                ]),
            dict(
                type='mmgen.Flip',
                direction='horizontal',
                keys=['img_a', 'img_b']),
            dict(
                type='mmgen.PackGenInputs',
                keys=['img_a', 'img_b'],
                meta_keys=['img_a_path', 'img_b_path'])
        ]

    def test_paired_image_dataset(self):
        dataset = PairedImageDataset(
            self.imgs_root, pipeline=self.default_pipeline)
        assert len(dataset) == 2
        print(dataset)
        img = dataset[0]['inputs']['img_a']
        assert img.ndim == 3
        img = dataset[0]['inputs']['img_b']
        assert img.ndim == 3
