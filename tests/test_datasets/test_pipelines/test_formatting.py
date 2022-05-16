# Copyright (c) OpenMMLab. All rights reserved.
from tkinter import Pack
import numpy as np
import pytest
import torch

from mmgen.datasets.pipelines import PackGenInputs
from unittest import TestCase

class TestPackGenInputs(TestCase):
    def test_unconditional_data_pack(self):
        # test without meta_keys
        img = np.random.rand(16, 16, 3).astype(np.float32)
        results = dict(img=img,img_shape=(16,16),num_classes=1000)
        packer = PackGenInputs(meta_keys=[])
        results = packer.transform(results)
        assert results['inputs'].shape == (3, 16, 16)
        assert results['data_sample'].metainfo_keys() == []

        # test with meta_keys
        img = np.random.rand(16, 16, 3).astype(np.float32)
        results = dict(img=img,img_shape=(16,16),num_classes=1000)
        packer = PackGenInputs(meta_keys=['img_shape', 'num_classes'])
        results = packer.transform(results)
        assert results['inputs'].shape == (3, 16, 16)
        assert set(results['data_sample'].metainfo_keys()) == set(['num_classes', 'img_shape'])
        assert results['data_sample'].img_shape == (16, 16)
        assert results['data_sample'].num_classes == 1000