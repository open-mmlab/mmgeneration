# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.data import LabelData

from mmgen.structures import GenDataSample


class TestGenDataSample(TestCase):

    def _test_set_label(self, key):
        data_sample = GenDataSample()
        method = getattr(data_sample, 'set_' + key)
        # Test number
        method(1)
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.LongTensor)

        # Test tensor with single number
        method(torch.tensor(2))
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.LongTensor)

        # Test array with single number
        method(np.array(3))
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.LongTensor)

        # Test tensor
        method(torch.tensor([1, 2, 3]))
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.Tensor)
        self.assertTrue((label.label == torch.tensor([1, 2, 3])).all())

        # Test array
        method(np.array([1, 2, 3]))
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertTrue((label.label == torch.tensor([1, 2, 3])).all())

        # Test Sequence
        method([1, 2, 3])
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertTrue((label.label == torch.tensor([1, 2, 3])).all())

        # Test Sequence with float number
        method([0.2, 0, 0.8])
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertTrue((label.label == torch.tensor([0.2, 0, 0.8])).all())

        # Test unavailable type
        with self.assertRaisesRegex(TypeError, "<class 'str'> is not"):
            method('hi')

        # Test set num_classes
        data_sample = GenDataSample(metainfo={'num_classes': 10})
        method = getattr(data_sample, 'set_' + key)
        method(5)
        self.assertIn(key, data_sample)
        label = getattr(data_sample, key)
        self.assertIsInstance(label, LabelData)
        self.assertIn('num_classes', label)
        self.assertEqual(label.num_classes, 10)

        # Test unavailable label
        with self.assertRaisesRegex(ValueError, r'data .*[15].* should '):
            method(15)

    def test_set_gt_label(self):
        self._test_set_label('gt_label')

    def test_del_gt_label(self):
        data_sample = GenDataSample()
        self.assertNotIn('gt_label', data_sample)
        data_sample.set_gt_label(1)
        self.assertIn('gt_label', data_sample)
        del data_sample.gt_label
        self.assertNotIn('gt_label', data_sample)
