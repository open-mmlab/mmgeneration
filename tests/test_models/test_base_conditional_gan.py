# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import torch
from mmcls.core import ClsDataSample
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
        self.assertEqual(len(outputs_val), 3)
        self.assertEqual(len(outputs_test), 3)
        for out_val, out_test in zip(outputs_val, outputs_test):
            self.assertEqual(out_val.fake_img.data.shape, (3, 32, 32))
            self.assertEqual(out_test.fake_img.data.shape, (3, 32, 32))

        # set mode
        inputs = dict(num_batches=4, sample_model='orig')
        outputs_val = gan.val_step(inputs)
        outputs_test = gan.test_step(inputs)
        self.assertEqual(len(outputs_val), 4)
        self.assertEqual(len(outputs_test), 4)
        for out_val, out_test in zip(outputs_val, outputs_test):
            self.assertEqual(out_val.sample_model, 'orig')
            self.assertEqual(out_test.sample_model, 'orig')
            self.assertEqual(out_val.fake_img.data.shape, (3, 32, 32))
            self.assertEqual(out_test.fake_img.data.shape, (3, 32, 32))

        inputs = dict(num_batches=4, sample_model='orig/ema')
        self.assertRaises(AssertionError, gan.val_step, inputs)

        inputs = dict(num_batches=4, sample_model='ema')
        self.assertRaises(AssertionError, gan.val_step, inputs)

        # set noise and label input
        gt_label = torch.randint(0, 10, (1, ))
        inputs = dict(
            inputs=dict(noise=torch.randn(10)),
            data_sample=GenDataSample(gt_label=LabelData(label=gt_label)))
        outputs_val = gan.val_step([inputs])
        outputs_test = gan.test_step([inputs])
        self.assertEqual(len(outputs_val), 1)
        self.assertEqual(len(outputs_val), 1)
        for idx in range(1):
            test_fake_img = outputs_test[idx].fake_img.data
            val_fake_img = outputs_val[idx].fake_img.data
            test_label = outputs_test[idx].gt_label.label
            val_label = outputs_val[idx].gt_label.label
            self.assertEqual(test_label, gt_label)
            self.assertEqual(val_label, gt_label)
            assert_allclose(test_fake_img, val_fake_img)

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
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            self.assertEqual(out.fake_img.data.shape, (3, 32, 32))

        outputs = gan(inputs)
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            self.assertEqual(out.fake_img.data.shape, (3, 32, 32))

        outputs = gan(torch.randn(3, 10))
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            self.assertEqual(out.fake_img.data.shape, (3, 32, 32))

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
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            self.assertEqual(out.fake_img.data.shape, (3, 32, 32))

        inputs = dict(num_batches=3, sample_model='ema/orig')
        outputs = gan(inputs)
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            ema_img = out.ema
            orig_img = out.orig
            self.assertEqual(ema_img.fake_img.data.shape,
                             orig_img.fake_img.data.shape)
            self.assertTrue(out.sample_model, 'ema/orig')

        inputs = dict(noise=torch.randn(4, 10))
        outputs = gan(inputs)
        self.assertEqual(len(outputs), 4)
        for out in outputs:
            self.assertEqual(out.fake_img.data.shape, (3, 32, 32))

        # test data sample input
        inputs = dict(noise=torch.randn(3, 10))
        label = [torch.randint(0, 10, (1, )) for _ in range(3)]
        data_sample = [ClsDataSample() for _ in range(3)]
        for idx, sample in enumerate(data_sample):
            sample.set_gt_label(label[idx])
        outputs = gan(inputs, data_sample)
        self.assertEqual(len(outputs), 3)
        for idx, output in enumerate(outputs):
            self.assertEqual(output.gt_label.label, label[idx])
