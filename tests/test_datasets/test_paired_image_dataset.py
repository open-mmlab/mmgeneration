import os.path as osp

from mmgen.datasets import PairedImageDataset


class TestPairedImageDataset(object):

    @classmethod
    def setup_class(cls):
        cls.imgs_root = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/paired')
        img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        cls.default_pipeline = [
            dict(
                type='LoadPairedImageFromFile', io_backend='disk', key='pair'),
            dict(
                type='Resize',
                keys=['img_a', 'img_b'],
                scale=(286, 286),
                interpolation='bicubic'),
            dict(
                type='FixedCrop',
                keys=['img_a', 'img_b'],
                crop_size=(256, 256)),
            dict(type='Flip', keys=['img_a', 'img_b'], direction='horizontal'),
            dict(type='RescaleToZeroOne', keys=['img_a', 'img_b']),
            dict(
                type='Normalize',
                keys=['img_a', 'img_b'],
                to_rgb=True,
                **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img_a', 'img_b']),
            dict(
                type='Collect',
                keys=['img_a', 'img_b'],
                meta_keys=['img_a_path', 'img_b_path'])
        ]

    def test_paired_image_dataset(self):
        dataset = PairedImageDataset(
            self.imgs_root, pipeline=self.default_pipeline)
        assert len(dataset) == 2
        img = dataset[0]['img_a']
        assert img.ndim == 3
        img = dataset[0]['img_b']
        assert img.ndim == 3
