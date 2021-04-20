import copy

import numpy as np
import pytest
import torch

from mmgen.datasets.pipelines import Flip, NumpyPad, Resize


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
        with pytest.raises(AssertionError):
            Resize([], scale=0.5)
        with pytest.raises(AssertionError):
            Resize(['gt_img'], size_factor=32, scale=0.5)
        with pytest.raises(AssertionError):
            Resize(['gt_img'], size_factor=32, keep_ratio=True)
        with pytest.raises(AssertionError):
            Resize(['gt_img'], max_size=32, size_factor=None)
        with pytest.raises(ValueError):
            Resize(['gt_img'], scale=-0.5)
        with pytest.raises(TypeError):
            Resize(['gt_img'], (0.4, 0.2))
        with pytest.raises(TypeError):
            Resize(['gt_img'], dict(test=None))

        target_keys = ['alpha']

        alpha = np.random.rand(240, 320).astype(np.float32)
        results = dict(alpha=alpha)
        resize = Resize(keys=['alpha'], size_factor=32, max_size=None)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert resize_results['alpha'].shape == (224, 320, 1)
        resize = Resize(keys=['alpha'], size_factor=32, max_size=320)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert resize_results['alpha'].shape == (224, 320, 1)

        resize = Resize(keys=['alpha'], size_factor=32, max_size=200)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert resize_results['alpha'].shape == (192, 192, 1)

        resize = Resize(['gt_img'], (-1, 200))
        assert resize.scale == (np.inf, 200)

        results = dict(gt_img=self.results['img'].copy())
        resize_keep_ratio = Resize(['gt_img'], scale=0.5, keep_ratio=True)
        results = resize_keep_ratio(results)
        assert results['gt_img'].shape[:2] == (128, 128)
        assert results['scale_factor'] == 0.5

        results = dict(gt_img=self.results['img'].copy())
        resize_keep_ratio = Resize(['gt_img'],
                                   scale=(128, 128),
                                   keep_ratio=False)
        results = resize_keep_ratio(results)
        assert results['gt_img'].shape[:2] == (128, 128)

        # test input with shape (256, 256)
        results = dict(gt_img=self.results['img'][..., 0].copy())
        resize = Resize(['gt_img'], scale=(128, 128), keep_ratio=False)
        results = resize(results)
        assert results['gt_img'].shape == (128, 128, 1)

        name_ = str(resize_keep_ratio)
        assert name_ == resize_keep_ratio.__class__.__name__ + (
            f"(keys={['gt_img']}, scale=(128, 128), "
            f'keep_ratio={False}, size_factor=None, '
            'max_size=None,interpolation=bilinear)')


def test_numpy_pad():
    results = dict(img=np.zeros((5, 5, 1)))

    pad = NumpyPad(['img'], ((2, 2), (0, 0), (0, 0)))
    results = pad(results)
    assert results['img'].shape == (9, 5, 1)

    repr_str = pad.__class__.__name__
    repr_str += (
        f'(keys={pad.keys}, padding={pad.padding}, kwargs={pad.kwargs})')

    assert str(pad) == repr_str
