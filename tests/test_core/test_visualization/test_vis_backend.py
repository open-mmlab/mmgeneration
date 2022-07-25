import os.path as osp
import shutil
import sys
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import torch
from mmengine import Config

from mmgen.core import GenVisBackend, PaviGenVisBackend


class TestGenVisBackend(TestCase):

    def test_vis_backend(self):
        data_root = 'tmp_dir'
        sys.modules['petrel_client'] = MagicMock()
        vis_backend = GenVisBackend(save_dir='tmp_dir', ceph_path='s3://xxx')

        self.assertEqual(vis_backend.experiment, vis_backend)

        # test with `delete_local` is True
        vis_backend.add_config(Config(dict(name='test')))
        vis_backend.add_image(
            name='add_img', image=np.random.random((8, 8, 3)).astype(np.uint8))
        vis_backend.add_scalars(
            scalar_dict=dict(lr=0.001, loss=torch.FloatTensor([0.693])),
            step=3)

        # test with `delete_local` is False
        vis_backend._delete_local = False
        vis_backend.add_config(Config(dict(name='test')))
        vis_backend.add_image(
            name='add_img', image=np.random.random((8, 8, 3)).astype(np.uint8))
        vis_backend.add_scalars(
            scalar_dict=dict(lr=0.001, loss=torch.FloatTensor([0.693])),
            step=3)
        vis_backend.add_scalars(
            scalar_dict=dict(lr=0.001, loss=torch.FloatTensor([0.693])),
            step=3,
            file_path='new_scalar.json')
        self.assertTrue(osp.exists(osp.join(data_root, 'config.py')))
        self.assertTrue(
            osp.exists(osp.join(data_root, 'vis_image', 'add_img_0.png')))
        self.assertTrue(osp.exists(osp.join(data_root, 'new_scalar.json')))
        self.assertTrue(osp.exists(osp.join(data_root, 'scalars.json')))

        # test with `ceph_path` is None
        vis_backend = GenVisBackend(save_dir='tmp_dir')
        vis_backend.add_config(Config(dict(name='test')))
        vis_backend.add_image(
            name='add_img', image=np.random.random((8, 8, 3)).astype(np.uint8))
        vis_backend.add_scalar(
            name='scalar_tensor', value=torch.FloatTensor([0.693]), step=3)
        vis_backend.add_scalar(name='scalar', value=0.693, step=3)
        vis_backend.add_scalars(
            scalar_dict=dict(lr=0.001, loss=torch.FloatTensor([0.693])),
            step=3)
        vis_backend.add_scalars(
            scalar_dict=dict(lr=0.001, loss=torch.FloatTensor([0.693])),
            step=3,
            file_path='new_scalar.json')

        # raise error
        with self.assertRaises(AssertionError):
            vis_backend.add_scalars(
                scalar_dict=dict(lr=0.001), step=3, file_path='scalars.json')
        with self.assertRaises(AssertionError):
            vis_backend.add_scalars(
                scalar_dict=dict(lr=0.001), step=3, file_path='new_scalars')

        shutil.rmtree('tmp_dir')


class TestPaviBackend(TestCase):

    def test_pavi(self):
        save_dir = 'tmp_dir'
        exp_name = 'unit test'
        vis_backend = PaviGenVisBackend(save_dir=save_dir, exp_name=exp_name)
        with self.assertRaises(ImportError):
            vis_backend._init_env()
        sys.modules['pavi'] = MagicMock()
        vis_backend._init_env()

        exp = vis_backend.experiment
        self.assertEqual(exp, vis_backend._pavi)

        # add image
        vis_backend.add_image(
            name='add_img', image=np.random.random((8, 8, 3)).astype(np.uint8))

        # add scalars
        vis_backend.add_scalars(
            scalar_dict=dict(lr=0.001, loss=torch.FloatTensor([0.693])),
            step=3)
