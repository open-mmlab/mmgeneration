# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import pytest
import torch

from mmgen.datasets.transforms import (CenterCropLongEdge, Flip, NumpyPad,
                                       RandomCropLongEdge, Resize)


class TestAugmentations(object):

    @classmethod
    def setup_class(cls):
        cls.results = dict()
        cls.img_gt = np.random.rand(256, 128, 3).astype(np.float32)
        cls.img_lq = np.random.rand(64, 32, 3).astype(np.float32)

        cls.results = dict(
            lq=cls.img_lq,
            gt=cls.img_gt,
            scale=4,
            lq_path='fake_lq_path',
            gt_path='fake_gt_path')

        cls.results['img'] = np.random.rand(256, 256, 3).astype(np.float32)
        cls.results['mask'] = np.random.rand(256, 256, 1).astype(np.float32)
        cls.results['img_tensor'] = torch.rand((3, 256, 256))
        cls.results['mask_tensor'] = torch.zeros((1, 256, 256))
        cls.results['mask_tensor'][:, 50:150, 40:140] = 1.

    @staticmethod
    def assert_img_equal(img, ref_img, ratio_thr=0.999):
        """Check if img and ref_img are matched approximately."""
        assert img.shape == ref_img.shape
        assert img.dtype == ref_img.dtype
        area = ref_img.shape[-1] * ref_img.shape[-2]
        diff = np.abs(img.astype('int32') - ref_img.astype('int32'))
        assert np.sum(diff <= 1) / float(area) > ratio_thr

    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""
        return set(target_keys).issubset(set(result_keys))

    @staticmethod
    def check_flip(origin_img, result_img, flip_type):
        """Check if the origin_img are flipped correctly into result_img in
        different flip_types."""
        h, w, c = origin_img.shape
        if flip_type == 'horizontal':
            # yapf: disable
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        if result_img[i, j, k] != origin_img[i, w - 1 - j, k]:
                            return False
            # yapf: enable
        else:
            # yapf: disable
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        if result_img[i, j, k] != origin_img[h - 1 - i, j, k]:
                            return False
            # yapf: enable
        return True

    def test_flip(self):
        results = copy.deepcopy(self.results)

        with pytest.raises(ValueError):
            Flip(keys=['lq', 'gt'], direction='vertically')

        # horizontal
        np.random.seed(1)
        target_keys = ['lq', 'gt', 'flip', 'flip_direction']
        flip = Flip(keys=['lq', 'gt'], flip_ratio=1, direction='horizontal')
        results = flip(results)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert self.check_flip(self.img_lq, results['lq'],
                               results['flip_direction'])
        assert self.check_flip(self.img_gt, results['gt'],
                               results['flip_direction'])
        assert results['lq'].shape == self.img_lq.shape
        assert results['gt'].shape == self.img_gt.shape

        # vertical
        results = copy.deepcopy(self.results)
        flip = Flip(keys=['lq', 'gt'], flip_ratio=1, direction='vertical')
        results = flip(results)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert self.check_flip(self.img_lq, results['lq'],
                               results['flip_direction'])
        assert self.check_flip(self.img_gt, results['gt'],
                               results['flip_direction'])
        assert results['lq'].shape == self.img_lq.shape
        assert results['gt'].shape == self.img_gt.shape
        assert repr(flip) == flip.__class__.__name__ + (
            f"(keys={['lq', 'gt']}, flip_ratio=1, "
            f"direction={results['flip_direction']})")

        # flip a list
        # horizontal
        flip = Flip(keys=['lq', 'gt'], flip_ratio=1, direction='horizontal')
        results = dict(
            lq=[self.img_lq, np.copy(self.img_lq)],
            gt=[self.img_gt, np.copy(self.img_gt)],
            scale=4,
            lq_path='fake_lq_path',
            gt_path='fake_gt_path')
        flip_rlt = flip(copy.deepcopy(results))
        assert self.check_keys_contain(flip_rlt.keys(), target_keys)
        assert self.check_flip(self.img_lq, flip_rlt['lq'][0],
                               flip_rlt['flip_direction'])
        assert self.check_flip(self.img_gt, flip_rlt['gt'][0],
                               flip_rlt['flip_direction'])
        np.testing.assert_almost_equal(flip_rlt['gt'][0], flip_rlt['gt'][1])
        np.testing.assert_almost_equal(flip_rlt['lq'][0], flip_rlt['lq'][1])

        # vertical
        flip = Flip(keys=['lq', 'gt'], flip_ratio=1, direction='vertical')
        flip_rlt = flip(copy.deepcopy(results))
        assert self.check_keys_contain(flip_rlt.keys(), target_keys)
        assert self.check_flip(self.img_lq, flip_rlt['lq'][0],
                               flip_rlt['flip_direction'])
        assert self.check_flip(self.img_gt, flip_rlt['gt'][0],
                               flip_rlt['flip_direction'])
        np.testing.assert_almost_equal(flip_rlt['gt'][0], flip_rlt['gt'][1])
        np.testing.assert_almost_equal(flip_rlt['lq'][0], flip_rlt['lq'][1])

        # no flip
        flip = Flip(keys=['lq', 'gt'], flip_ratio=0, direction='vertical')
        results = flip(copy.deepcopy(results))
        assert self.check_keys_contain(results.keys(), target_keys)
        np.testing.assert_almost_equal(results['gt'][0], self.img_gt)
        np.testing.assert_almost_equal(results['lq'][0], self.img_lq)
        np.testing.assert_almost_equal(results['gt'][0], results['gt'][1])
        np.testing.assert_almost_equal(results['lq'][0], results['lq'][1])

    def test_resize(self):
        # test grayscale resize
        img = np.random.rand(16, 16).astype(np.float32)
        results = dict(img=img)
        resize = Resize(scale_factor=4.)
        resize_results = resize(results)
        assert resize_results['img'].shape == (64, 64, 1)

        img = np.random.rand(16, 16).astype(np.float32)
        results = dict(img=img)
        resize = Resize(scale=(8, 32))
        resize_results = resize(results)
        assert resize_results['img'].shape == (32, 8, 1)


def test_random_long_edge_crop():
    results = dict(img=np.random.rand(256, 128, 3).astype(np.float32))
    crop = RandomCropLongEdge(['img'])
    results = crop(results)
    assert results['img'].shape == (128, 128, 3)

    repr_str = crop.__class__.__name__
    repr_str += (f'(keys={crop.keys})')

    assert str(crop) == repr_str


def test_center_long_edge_crop():
    results = dict(img=np.random.rand(256, 128, 3).astype(np.float32))
    crop = CenterCropLongEdge(['img'])
    results = crop(results)
    assert results['img'].shape == (128, 128, 3)

    repr_str = crop.__class__.__name__
    repr_str += (f'(keys={crop.keys})')

    assert str(crop) == repr_str


def test_numpy_pad():
    results = dict(img=np.zeros((5, 5, 1)))

    pad = NumpyPad(((2, 2), (0, 0), (0, 0)))
    results = pad(results)
    assert results['img'].shape == (9, 5, 1)

    repr_str = pad.__class__.__name__
    repr_str += (f'(padding={pad.padding}, kwargs={pad.kwargs})')

    assert str(pad) == repr_str
