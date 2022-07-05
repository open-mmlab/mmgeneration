# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import torch
from mmengine.data import LabelData
from mmengine.testing import assert_allclose

from mmgen.core import GenDataSample
from mmgen.models import BaseConditionalGAN, GANDataPreprocessor

generator = dict(
    type='SAGANGenerator',
    output_scale=32,
    base_channels=32,
    attention_cfg=dict(type='SelfAttentionBlock'),
    attention_after_nth_block=2,
    with_spectral_norm=True)
discriminator = dict(
    type='ProjDiscriminator',
    input_scale=32,
    base_channels=32,
    attention_cfg=dict(type='SelfAttentionBlock'),
    attention_after_nth_block=1,
    with_spectral_norm=True)


class ToyCGAN(BaseConditionalGAN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_generator(self, inputs, data_samples, optimizer_wrapper):
        return dict(loss_gen=1)

    def train_discriminator(self, inputs, data_samples, optimizer_wrapper):
        return dict(loss_disc=2)


class TestBaseGAN(TestCase):

    def test_val_step_and_test_step(self):
        gan = ToyCGAN(
            noise_size=10,
            num_classes=10,
            generator=deepcopy(generator),
            data_preprocessor=GANDataPreprocessor())
        gan.eval()

        # no mode
        inputs = dict(num_batches=3)
        outputs_val = gan.val_step(inputs)
        outputs_test = gan.test_step(inputs)
        self.assertEqual(outputs_val.shape, (3, 3, 32, 32))
        self.assertEqual(outputs_test.shape, (3, 3, 32, 32))

        # set mode
        inputs = dict(num_batches=4, sample_model='orig')
        outputs_val = gan.val_step(inputs)
        outputs_test = gan.test_step(inputs)
        self.assertEqual(outputs_val.shape, (4, 3, 32, 32))
        self.assertEqual(outputs_test.shape, (4, 3, 32, 32))

        inputs = dict(num_batches=4, sample_model='orig/ema')
        self.assertRaises(AssertionError, gan.val_step, inputs)

        inputs = dict(num_batches=4, sample_model='ema')
        self.assertRaises(AssertionError, gan.val_step, inputs)

        # set noise and label input
        inputs = dict(
            inputs=dict(noise=torch.randn(10)),
            data_sample=GenDataSample(
                gt_label=LabelData(label=torch.randint(0, 10, (1, )))))
        outputs_val = gan.val_step([inputs])
        outputs_test = gan.test_step([inputs])
        assert_allclose(outputs_test, outputs_val)

    def test_forward(self):
        # set a gan w/o EMA
        gan = ToyCGAN(
            noise_size=10,
            num_classes=10,
            generator=deepcopy(generator),
            data_preprocessor=GANDataPreprocessor())
        gan.eval()
        inputs = dict(num_batches=3)
        outputs = gan(inputs, None)
        self.assertEqual(outputs.shape, (3, 3, 32, 32))

        outputs = gan(inputs)
        self.assertEqual(outputs.shape, (3, 3, 32, 32))

        outputs = gan(torch.randn(3, 10))
        self.assertEqual(outputs.shape, (3, 3, 32, 32))

        # set a gan w EMA
        gan = ToyCGAN(
            noise_size=10,
            num_classes=10,
            generator=deepcopy(generator),
            data_preprocessor=GANDataPreprocessor(),
            ema_config=dict(interval=1))
        gan.eval()
        inputs = dict(num_batches=3)
        outputs = gan(inputs)
        self.assertEqual(outputs.shape, (3, 3, 32, 32))

        inputs = dict(num_batches=3, sample_model='ema/orig')
        outputs = gan(inputs)
        self.assertEqual(set(outputs.keys()), set(['ema', 'orig']))
        self.assertEqual(outputs['ema'].shape, outputs['orig'].shape)

        inputs = dict(noise=torch.randn(4, 10))
        outputs = gan(inputs)
        self.assertEqual(outputs.shape, (4, 3, 32, 32))
