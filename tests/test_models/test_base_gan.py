# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from mmengine import MessageHub
from mmengine.optim import OptimWrapper, OptimWrapperDict
from mmengine.testing import assert_allclose
from torch.optim import SGD

from mmgen.models import BaseGAN, GANDataPreprocessor
from mmgen.models.builder import build_module

generator = dict(type='DCGANGenerator', output_scale=8, base_channels=8)
discriminator = dict(
    type='DCGANDiscriminator',
    base_channels=8,
    input_scale=8,
    output_scale=4,
    out_channels=1)


class ToyGAN(BaseGAN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_generator(self, inputs, data_samples, optimizer_wrapper):
        return dict(loss_gen=1)

    def train_discriminator(self, inputs, data_samples, optimizer_wrapper):
        return dict(loss_disc=2)


class TestBaseGAN(TestCase):

    def test_init(self):
        gan = ToyGAN(
            noise_size=5,
            generator=deepcopy(generator),
            discriminator=deepcopy(discriminator),
            data_preprocessor=GANDataPreprocessor())
        self.assertIsInstance(gan, BaseGAN)
        self.assertIsInstance(gan.data_preprocessor, GANDataPreprocessor)

        # test only generator have noise size
        gen_cfg = deepcopy(generator)
        gen_cfg['noise_size'] = 10
        gan = ToyGAN(
            generator=gen_cfg,
            discriminator=discriminator,
            data_preprocessor=GANDataPreprocessor())
        self.assertEqual(gan.noise_size, 10)

        # test init with nn.Module
        gen_cfg = deepcopy(generator)
        gen_cfg['noise_size'] = 10
        disc_cfg = deepcopy(discriminator)
        gen = build_module(gen_cfg)
        disc = build_module(disc_cfg)
        gan = ToyGAN(
            generator=gen,
            discriminator=disc,
            data_preprocessor=GANDataPreprocessor())
        self.assertEqual(gan.generator, gen)
        self.assertEqual(gan.discriminator, disc)

        # test init without discriminator
        gan = ToyGAN(generator=gen, data_preprocessor=GANDataPreprocessor())
        self.assertEqual(gan.discriminator, None)

    def test_train_step(self):
        # prepare model
        accu_iter = 2
        n_disc = 2
        message_hub = MessageHub.get_instance('mmgen')
        gan = ToyGAN(
            noise_size=10,
            generator=generator,
            discriminator=discriminator,
            data_preprocessor=GANDataPreprocessor(),
            discriminator_steps=n_disc)
        ToyGAN.train_discriminator = MagicMock(
            return_value=dict(loss_disc=torch.Tensor(1), loss=torch.Tensor(1)))
        ToyGAN.train_generator = MagicMock(
            return_value=dict(loss_gen=torch.Tensor(2), loss=torch.Tensor(2)))
        # prepare messageHub
        message_hub.update_info('iter', 0)
        # prepare optimizer
        gen_optim = SGD(gan.generator.parameters(), lr=0.1)
        disc_optim = SGD(gan.discriminator.parameters(), lr=0.1)
        optim_wrapper_dict = OptimWrapperDict(
            generator=OptimWrapper(gen_optim, accumulative_counts=accu_iter),
            discriminator=OptimWrapper(
                disc_optim, accumulative_counts=accu_iter))
        # prepare inputs
        inputs = torch.randn(3, 4, 4)
        data = dict(inputs=inputs)

        # simulate train_loop here
        disc_update_times = 0
        for idx in range(n_disc * accu_iter):
            message_hub.update_info('iter', idx)
            log = gan.train_step([data], optim_wrapper_dict)
            if (idx + 1) == n_disc * accu_iter:
                # should update at after (n_disc * accu_iter)
                self.assertEqual(ToyGAN.train_generator.call_count, accu_iter)
                self.assertEqual(
                    set(log.keys()), set(['loss', 'loss_disc', 'loss_gen']))
            else:
                # should not update when discriminator's updating is unfinished
                self.assertEqual(ToyGAN.train_generator.call_count, 0)
                self.assertEqual(log.keys(), set(['loss', 'loss_disc']))

            # disc should update once for each iteration
            disc_update_times += 1
            self.assertEqual(ToyGAN.train_discriminator.call_count,
                             disc_update_times)

    def test_update_ema(self):
        # prepare model
        n_gen = 4
        n_disc = 2
        accu_iter = 2
        ema_interval = 3
        message_hub = MessageHub.get_instance('mmgen')
        gan = ToyGAN(
            noise_size=10,
            generator=generator,
            discriminator=discriminator,
            data_preprocessor=GANDataPreprocessor(),
            discriminator_steps=n_disc,
            generator_steps=n_gen,
            ema_config=dict(interval=ema_interval))
        gan.train_discriminator = MagicMock(
            return_value=dict(loss_disc=torch.Tensor(1), loss=torch.Tensor(1)))
        gan.train_generator = MagicMock(
            return_value=dict(loss_gen=torch.Tensor(2), loss=torch.Tensor(2)))

        self.assertTrue(gan.with_ema_gen)
        # mock generator_ema with MagicMock
        del gan.generator_ema
        setattr(gan, 'generator_ema', MagicMock())
        # prepare messageHub
        message_hub.update_info('iter', 0)
        # prepare optimizer
        gen_optim = SGD(gan.generator.parameters(), lr=0.1)
        disc_optim = SGD(gan.discriminator.parameters(), lr=0.1)
        optim_wrapper_dict = OptimWrapperDict(
            generator=OptimWrapper(gen_optim, accumulative_counts=accu_iter),
            discriminator=OptimWrapper(
                disc_optim, accumulative_counts=accu_iter))
        # prepare inputs
        inputs = torch.randn(3, 4, 4)
        data = dict(inputs=inputs)

        # simulate train_loop here
        ema_times = 0
        gen_update_times = 0
        disc_update_times = 0
        for idx in range(n_disc * accu_iter * ema_interval):
            message_hub.update_info('iter', idx)
            gan.train_step([data], optim_wrapper_dict)
            if (idx + 1) % (n_disc * accu_iter) == 0:
                ema_times += 1
                gen_update_times += accu_iter * n_gen

            disc_update_times += 1
            self.assertEqual(gan.generator_ema.update_parameters.call_count,
                             ema_times)
            self.assertEqual(gan.train_generator.call_count, gen_update_times)
            # disc should update once for each iteration
            self.assertEqual(gan.train_discriminator.call_count,
                             disc_update_times)

    def test_val_step_and_test_step(self):
        gan = ToyGAN(
            noise_size=10,
            generator=deepcopy(generator),
            data_preprocessor=GANDataPreprocessor())

        # no mode
        inputs = dict(num_batches=3)
        outputs_val = gan.val_step(inputs)
        outputs_test = gan.test_step(inputs)
        self.assertEqual(outputs_val.shape, (3, 3, 8, 8))
        self.assertEqual(outputs_test.shape, (3, 3, 8, 8))

        # set mode
        inputs = dict(num_batches=4, sample_model='orig')
        outputs_val = gan.val_step(inputs)
        outputs_test = gan.test_step(inputs)
        self.assertEqual(outputs_val.shape, (4, 3, 8, 8))
        self.assertEqual(outputs_test.shape, (4, 3, 8, 8))

        inputs = dict(num_batches=4, sample_model='orig/ema')
        self.assertRaises(AssertionError, gan.val_step, inputs)

        inputs = dict(num_batches=4, sample_model='ema')
        self.assertRaises(AssertionError, gan.val_step, inputs)

        # set noise input
        inputs = dict(noise=torch.randn(4, 10))
        outputs_val = gan.val_step(inputs)
        outputs_test = gan.test_step(inputs)
        assert_allclose(outputs_test, outputs_val)

    def test_forward(self):
        # set a gan w/o EMA
        gan = ToyGAN(
            noise_size=10,
            generator=deepcopy(generator),
            data_preprocessor=GANDataPreprocessor())
        inputs = dict(num_batches=3)
        outputs = gan(inputs, None)
        self.assertEqual(outputs.shape, (3, 3, 8, 8))

        outputs = gan(inputs)
        self.assertEqual(outputs.shape, (3, 3, 8, 8))

        outputs = gan(torch.randn(3, 10))
        self.assertEqual(outputs.shape, (3, 3, 8, 8))

        # set a gan w EMA
        gan = ToyGAN(
            noise_size=10,
            generator=deepcopy(generator),
            data_preprocessor=GANDataPreprocessor(),
            ema_config=dict(interval=1))
        inputs = dict(num_batches=3)
        outputs = gan(inputs)
        self.assertEqual(outputs.shape, (3, 3, 8, 8))

        inputs = dict(num_batches=3, sample_model='ema/orig')
        outputs = gan(inputs)
        self.assertEqual(set(outputs.keys()), set(['ema', 'orig']))
        self.assertEqual(outputs['ema'].shape, outputs['orig'].shape)

        inputs = dict(noise=torch.randn(4, 10))
        outputs = gan(inputs)
        self.assertEqual(outputs.shape, (4, 3, 8, 8))
