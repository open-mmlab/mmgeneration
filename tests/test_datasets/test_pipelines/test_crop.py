import copy

import numpy as np
import pytest

from mmgen.datasets.pipelines import Crop, FixedCrop


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

        cls.results['img_a'] = np.random.rand(286, 286, 3).astype(np.float32)
        cls.results['img_b'] = np.random.rand(286, 286, 3).astype(np.float32)

    @staticmethod
    def check_crop(result_img_shape, result_bbox):
        crop_w = result_bbox[2] - result_bbox[0]
        """Check if the result_bbox is in correspond to result_img_shape."""
        crop_h = result_bbox[3] - result_bbox[1]
        crop_shape = (crop_h, crop_w)
        return result_img_shape == crop_shape

    @staticmethod
    def check_crop_around_semi(alpha):
        return ((alpha > 0) & (alpha < 255)).any()

    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""
        return set(target_keys).issubset(set(result_keys))

    def test_crop(self):
        with pytest.raises(TypeError):
            Crop(['img'], (0.23, 0.1))

        # test center crop
        results = copy.deepcopy(self.results)
        center_crop = Crop(['img'], crop_size=(128, 128), random_crop=False)
        results = center_crop(results)
        assert results['img_crop_bbox'] == [64, 64, 128, 128]
        assert np.array_equal(self.results['img'][64:192, 64:192, :],
                              results['img'])

        # test random crop
        results = copy.deepcopy(self.results)
        random_crop = Crop(['img'], crop_size=(128, 128), random_crop=True)
        results = random_crop(results)
        assert 0 <= results['img_crop_bbox'][0] <= 128
        assert 0 <= results['img_crop_bbox'][1] <= 128
        assert results['img_crop_bbox'][2] == 128
        assert results['img_crop_bbox'][3] == 128

        # test random crop for lager size than the original shape
        results = copy.deepcopy(self.results)
        random_crop = Crop(['img'], crop_size=(512, 512), random_crop=True)
        results = random_crop(results)
        assert np.array_equal(self.results['img'], results['img'])
        assert str(random_crop) == (
            random_crop.__class__.__name__ +
            "keys=['img'], crop_size=(512, 512), random_crop=True")

    def test_fixed_crop(self):
        with pytest.raises(TypeError):
            FixedCrop(['img_a', 'img_b'], (0.23, 0.1))
        with pytest.raises(TypeError):
            FixedCrop(['img_a', 'img_b'], (256, 256), (0, 0.1))

        # test shape consistency
        results = copy.deepcopy(self.results)
        fixed_crop = FixedCrop(['img_a', 'img'], crop_size=(128, 128))
        with pytest.raises(ValueError):
            results = fixed_crop(results)

        # test given pos crop
        results = copy.deepcopy(self.results)
        given_pos_crop = FixedCrop(['img_a', 'img_b'],
                                   crop_size=(256, 256),
                                   crop_pos=(1, 1))
        results = given_pos_crop(results)
        assert results['img_a_crop_bbox'] == [1, 1, 256, 256]
        assert results['img_b_crop_bbox'] == [1, 1, 256, 256]
        assert np.array_equal(self.results['img_a'][1:257, 1:257, :],
                              results['img_a'])
        assert np.array_equal(self.results['img_b'][1:257, 1:257, :],
                              results['img_b'])

        # test given pos crop if pos > suitable pos
        results = copy.deepcopy(self.results)
        given_pos_crop = FixedCrop(['img_a', 'img_b'],
                                   crop_size=(256, 256),
                                   crop_pos=(280, 280))
        results = given_pos_crop(results)
        assert results['img_a_crop_bbox'] == [280, 280, 6, 6]
        assert results['img_b_crop_bbox'] == [280, 280, 6, 6]
        assert np.array_equal(self.results['img_a'][280:, 280:, :],
                              results['img_a'])
        assert np.array_equal(self.results['img_b'][280:, 280:, :],
                              results['img_b'])
        assert str(given_pos_crop) == (
            given_pos_crop.__class__.__name__ +
            "keys=['img_a', 'img_b'], crop_size=(256, 256), " +
            'crop_pos=(280, 280)')

        # test random initialized fixed crop
        results = copy.deepcopy(self.results)
        random_fixed_crop = FixedCrop(['img_a', 'img_b'],
                                      crop_size=(256, 256),
                                      crop_pos=None)
        results = random_fixed_crop(results)
        assert 0 <= results['img_a_crop_bbox'][0] <= 30
        assert 0 <= results['img_a_crop_bbox'][1] <= 30
        assert results['img_a_crop_bbox'][2] == 256
        assert results['img_a_crop_bbox'][3] == 256
        x_offset, y_offset, crop_w, crop_h = results['img_a_crop_bbox']
        assert x_offset == results['img_b_crop_bbox'][0]
        assert y_offset == results['img_b_crop_bbox'][1]
        assert crop_w == results['img_b_crop_bbox'][2]
        assert crop_h == results['img_b_crop_bbox'][3]
        assert np.array_equal(
            self.results['img_a'][y_offset:y_offset + crop_h,
                                  x_offset:x_offset + crop_w, :],
            results['img_a'])
        assert np.array_equal(
            self.results['img_b'][y_offset:y_offset + crop_h,
                                  x_offset:x_offset + crop_w, :],
            results['img_b'])

        # test given pos crop for lager size than the original shape
        results = copy.deepcopy(self.results)
        given_pos_crop = FixedCrop(['img_a', 'img_b'],
                                   crop_size=(512, 512),
                                   crop_pos=(1, 1))
        results = given_pos_crop(results)
        assert results['img_a_crop_bbox'] == [1, 1, 285, 285]
        assert results['img_b_crop_bbox'] == [1, 1, 285, 285]
        assert np.array_equal(self.results['img_a'][1:, 1:, :],
                              results['img_a'])
        assert np.array_equal(self.results['img_b'][1:, 1:, :],
                              results['img_b'])
        assert str(given_pos_crop) == (
            given_pos_crop.__class__.__name__ +
            "keys=['img_a', 'img_b'], crop_size=(512, 512), crop_pos=(1, 1)")

        # test random initialized fixed crop for lager size
        # than the original shape
        results = copy.deepcopy(self.results)
        random_fixed_crop = FixedCrop(['img_a', 'img_b'],
                                      crop_size=(512, 512),
                                      crop_pos=None)
        results = random_fixed_crop(results)
        assert results['img_a_crop_bbox'] == [0, 0, 286, 286]
        assert results['img_b_crop_bbox'] == [0, 0, 286, 286]
        assert np.array_equal(self.results['img_a'], results['img_a'])
        assert np.array_equal(self.results['img_b'], results['img_b'])
        assert str(random_fixed_crop) == (
            random_fixed_crop.__class__.__name__ +
            "keys=['img_a', 'img_b'], crop_size=(512, 512), crop_pos=None")
