# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import pytest
import torch

from mmgen.models.builder import build_model
from mmgen.models.diffusions import (BasicGaussianDiffusion,
                                     UniformTimeStepSampler)
from mmgen.models.diffusions.utils import _get_label_batch, _get_noise_batch


class TestBasicGaussianDiffusion:

    @classmethod
    def setup_class(cls):
        cls.config = dict(
            type='BasicGaussianDiffusion',
            num_timesteps=10,
            betas_cfg=dict(type='cosine'),
            train_cfg=None,
            test_cfg=None)
        cls.denoising = dict(
            type='DenoisingUnet',
            image_size=32,
            in_channels=3,
            base_channels=128,
            resblocks_per_downsample=1,
            attention_res=[16, 8],
            use_scale_shift_norm=True,
            dropout=0.3,
            num_heads=1,
            use_rescale_timesteps=True,
            output_cfg=dict(mean='eps', var='learned_range'),
        )
        cls.sampler = dict(type='UniformTimeStepSampler')
        cls.ddpm_loss = [
            dict(
                type='DDPMVLBLoss',
                rescale_mode='constant',
                rescale_cfg=dict(scale=1),
                data_info=dict(
                    mean_pred='mean_pred',
                    mean_target='mean_posterior',
                    logvar_pred='logvar_pred',
                    logvar_target='logvar_posterior'),
                log_cfgs=[
                    dict(
                        type='quartile',
                        prefix_name='loss_vlb',
                        total_timesteps=1000),
                    dict(type='name')
                ]),
            dict(
                type='DDPMMSELoss',
                log_cfgs=dict(
                    type='quartile',
                    prefix_name='loss_mse',
                    total_timesteps=1000),
            )
        ]

    def test_diffusion(self):
        # test build model
        cfg = deepcopy(self.config)
        cfg['denoising'] = self.denoising
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        diffusion = build_model(cfg)
        assert isinstance(diffusion, BasicGaussianDiffusion)
        assert isinstance(diffusion.sampler, UniformTimeStepSampler)
        assert not diffusion.use_ema

        # test build model --> parse train_cfg with ema
        cfg = deepcopy(self.config)
        cfg['denoising'] = self.denoising
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        cfg['train_cfg'] = dict(use_ema=True)
        diffusion = build_model(cfg)
        assert isinstance(diffusion, BasicGaussianDiffusion)
        assert isinstance(diffusion.sampler, UniformTimeStepSampler)
        assert diffusion.use_ema

        # test build_model --> parse train_cfg, without ema
        cfg = deepcopy(self.config)
        cfg['denoising'] = self.denoising
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        cfg['train_cfg'] = dict(use_ema=False)
        diffusion = build_model(cfg)
        assert isinstance(diffusion, BasicGaussianDiffusion)
        assert isinstance(diffusion.sampler, UniformTimeStepSampler)
        assert not diffusion.use_ema

        # test build_model --> parse test_cfg, with ema
        cfg = deepcopy(self.config)
        cfg['denoising'] = self.denoising
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        cfg['test_cfg'] = dict(use_ema=True)
        diffusion = build_model(cfg)
        assert isinstance(diffusion, BasicGaussianDiffusion)
        assert isinstance(diffusion.sampler, UniformTimeStepSampler)
        assert diffusion.use_ema

        # test build_model --> parse test_cfg, without ema
        cfg = deepcopy(self.config)
        cfg['denoising'] = self.denoising
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        cfg['test_cfg'] = dict(use_ema=False)
        diffusion = build_model(cfg)
        assert isinstance(diffusion, BasicGaussianDiffusion)
        assert isinstance(diffusion.sampler, UniformTimeStepSampler)
        assert not diffusion.use_ema

        # test sampler is None
        cfg = deepcopy(self.config)
        cfg['denoising'] = self.denoising
        cfg['timestep_sampler'] = None
        cfg['ddpm_loss'] = None
        diffusion = build_model(cfg)
        assert diffusion.sampler is None

        # test build model --> betas type = linear
        cfg = deepcopy(self.config)
        cfg['betas_cfg'] = dict(type='linear')
        cfg['denoising'] = self.denoising
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        diffusion = build_model(cfg)
        assert isinstance(diffusion, BasicGaussianDiffusion)
        assert isinstance(diffusion.sampler, UniformTimeStepSampler)
        assert not diffusion.use_ema

        # test build model --> wrong beta cfgs
        cfg = deepcopy(self.config)
        cfg['betas_cfg'] = dict(type='sine')
        cfg['denoising'] = self.denoising
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        with pytest.raises(AttributeError):
            diffusion = build_model(cfg)

        # test forward train --> raise error
        with pytest.raises(NotImplementedError):
            diffusion(None, return_loss=True)

        # test forward test
        imgs = diffusion(
            None, return_loss=False, mode='sampling', num_batches=2)
        assert imgs.shape == (2, 3, 32, 32)

        # test forward test --> wrong mode
        with pytest.raises(NotImplementedError):
            diffusion(
                None, return_loss=False, mode='generation', num_batches=2)

        # test reconstruction step --> given timestep
        cfg = deepcopy(self.config)
        cfg['denoising'] = self.denoising
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        diffusion = build_model(cfg)
        data_batch = dict(real_img=torch.randn(2, 3, 32, 32))
        fake_imgs = diffusion(
            data_batch, timesteps=[0, 5], mode='reconstruction')
        assert fake_imgs.shape == (2, 3, 32, 32)

        # test reconstruction step --> timestep = None
        output_dict = diffusion(
            data_batch, mode='reconstruction', return_noise=True)
        assert output_dict['fake_img'].shape == (20, 3, 32, 32)
        timestep = torch.cat([torch.LongTensor([i, i]) for i in range(10)])
        assert (output_dict['timesteps'] == timestep).all()

        # test reconstruction step --> noise in input
        noise = torch.randn(2, 3, 32, 32)
        output_dict = diffusion(
            data_batch, noise=noise, mode='reconstruction', return_noise=True)
        assert output_dict['noise'].shape == (20, 3, 32, 32)
        assert (output_dict['noise'] == torch.cat([noise for _ in range(10)],
                                                  dim=0)).all()

        # test reconstruction step --> noise in data_batch
        data_batch = dict(real_img=torch.randn(2, 3, 32, 32), noise=noise)
        output_dict = diffusion(
            data_batch, mode='reconstruction', return_noise=True)
        assert output_dict['noise'].shape == (20, 3, 32, 32)
        assert (output_dict['noise'] == torch.cat([noise for _ in range(10)],
                                                  dim=0)).all()

        # test reconstruction step --> noise in data_batch and input (error)
        data_batch = dict(
            real_img=torch.randn(2, 3, 32, 32),
            noise=torch.randn(2, 3, 32, 32))
        with pytest.raises(AssertionError):
            output_dict = diffusion(
                data_batch,
                noise=torch.randn(2, 3, 32, 32),
                mode='reconstruction')

        # test sample from noise
        fake_imgs = diffusion.sample_from_noise(None, num_batches=2)
        assert fake_imgs.shape == (2, 3, 32, 32)

        # test sample from noise --> save_intermedia
        output_dict = diffusion.sample_from_noise(
            None, num_batches=2, save_intermedia=True)
        assert list(output_dict.keys()) == [i for i in range(10, -1, -1)]

        # test sample from noise -->
        # sample model == ema/orig and use_ema == False
        output_dict = diffusion.sample_from_noise(
            None, num_batches=2, save_intermedia=True)

        # test sample from noise --> sample model == ema but use_ema = False
        with pytest.raises(AssertionError):
            diffusion.sample_from_noise(
                None, num_batches=2, sample_model='ema')

        # test sample from noise --> wrong sample method
        diffusion.sample_method = 'dk_method'
        with pytest.raises(AttributeError):
            diffusion.sample_from_noise(
                None, num_batches=2, save_intermedia=True)

        # test sample from noise --> sample model == orig/ema
        cfg = deepcopy(self.config)
        denoising_cfg = deepcopy(self.denoising)
        denoising_cfg['output_cfg'] = dict(mean='start_x', var='learned')
        cfg['train_cfg'] = dict(use_ema=True)
        cfg['denoising'] = denoising_cfg
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        diffusion = build_model(cfg)
        output = diffusion.sample_from_noise(None, num_batches=2)
        assert output.shape == (4, 3, 32, 32)

        # test sample from noise --> ema
        fake_img = diffusion.sample_from_noise(
            None, num_batches=2, sample_model='ema')
        assert fake_img.shape == (2, 3, 32, 32)

        # test sample from noise --> orig/ema + save_intermedia
        output_dict = diffusion.sample_from_noise(
            None, num_batches=2, save_intermedia=True)
        assert list(output_dict.keys()) == [i for i in range(10, -1, -1)]
        assert all([v.shape == (4, 3, 32, 32) for v in output_dict.values()])

        # test sample from noise -->
        # sample model == ema/orig return noise = True
        output_dict = diffusion.sample_from_noise(
            None,
            num_batches=2,
            save_intermedia=True,
            return_noise=True,
            sample_model='ema/orig')
        assert list(output_dict.keys()) == [i for i in range(10, -1, -1)]
        assert all([v.shape == (4, 3, 32, 32) for v in output_dict.values()])

        # test sample from noise -->
        # sample model == ema/orig return noise = True, timesteps_noise
        output_dict = diffusion.sample_from_noise(
            None,
            num_batches=2,
            save_intermedia=True,
            return_noise=True,
            timesteps_noise=torch.randn(10, 3, 32, 32),
            sample_model='ema/orig')
        assert list(output_dict.keys()) == [i for i in range(10, -1, -1)]
        assert all([v.shape == (4, 3, 32, 32) for v in output_dict.values()])

        # test denoising_var_mode = 'LEARNED' & denoising_mean_mode = 'start_x'
        cfg = deepcopy(self.config)
        denoising_cfg = deepcopy(self.denoising)
        denoising_cfg['output_cfg'] = dict(mean='start_x', var='learned')
        cfg['denoising'] = denoising_cfg
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        diffusion = build_model(cfg)
        output_dict = diffusion.sample_from_noise(None, num_batches=2)

        # test denoising_var_mode = 'fixed_large' &
        # denoising_mean_mode = 'previous_x'
        cfg = deepcopy(self.config)
        denoising_cfg = deepcopy(self.denoising)
        denoising_cfg['output_cfg'] = dict(
            mean='previous_x', var='fixed_large')
        cfg['denoising'] = denoising_cfg
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        diffusion = build_model(cfg)
        output_dict = diffusion.sample_from_noise(None, num_batches=2)

        # test denoising_var_mode = 'fixed_small' &
        # denoising_mean_mode = 'previous_x'
        cfg = deepcopy(self.config)
        denoising_cfg = deepcopy(self.denoising)
        denoising_cfg['output_cfg'] = dict(
            mean='previous_x', var='fixed_small')
        cfg['denoising'] = denoising_cfg
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        diffusion = build_model(cfg)
        output_dict = diffusion.sample_from_noise(None, num_batches=2)

        # test output_cfg --> error denoising_mean_mode
        cfg = deepcopy(self.config)
        denoising_cfg = deepcopy(self.denoising)
        denoising_cfg['output_cfg'] = dict(mean='x_0', var='fixed_small')
        cfg['denoising'] = denoising_cfg
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        diffusion = build_model(cfg)
        with pytest.raises(AttributeError):
            output_dict = diffusion.sample_from_noise(None, num_batches=2)

        # test output_cfg --> error denoising_var_mode
        cfg = deepcopy(self.config)
        denoising_cfg = deepcopy(self.denoising)
        denoising_cfg['output_cfg'] = dict(mean='previous_x', var='fixex')
        cfg['denoising'] = denoising_cfg
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = None
        diffusion = build_model(cfg)
        with pytest.raises(AttributeError):
            output_dict = diffusion.sample_from_noise(None, num_batches=2)

        # test train step --> no running status but have diffusion.iteration
        cfg = deepcopy(self.config)
        cfg['denoising'] = deepcopy(self.denoising)
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = deepcopy(self.ddpm_loss)
        diffusion = build_model(cfg)
        setattr(diffusion, 'iteration', 1)
        data = dict(real_img=torch.randn(2, 3, 32, 32))
        optimizer = dict(
            denoising=torch.optim.SGD(
                diffusion.denoising.parameters(), lr=0.01))
        model_outputs = diffusion.train_step(data, optimizer)
        assert 'log_vars' in model_outputs
        assert 'results' in model_outputs

        # test train step --> running status
        cfg = deepcopy(self.config)
        cfg['denoising'] = deepcopy(self.denoising)
        cfg['timestep_sampler'] = self.sampler
        cfg['ddpm_loss'] = deepcopy(self.ddpm_loss)
        diffusion = build_model(cfg)
        data = dict(real_img=torch.randn(2, 3, 32, 32))
        optimizer = dict(
            denoising=torch.optim.SGD(
                diffusion.denoising.parameters(), lr=0.01))
        model_outputs = diffusion.train_step(
            data, optimizer, running_status=dict(iteration=1))
        assert 'log_vars' in model_outputs
        assert 'results' in model_outputs


def test_ddpm_noise_batch_utils():
    image_shape = (3, 32, 32)
    num_batches = 2

    # noise is None, timestep is False
    noise_out = _get_noise_batch(None, image_shape, num_batches=num_batches)
    assert noise_out.shape == (2, 3, 32, 32)
    print(noise_out.shape)

    # noise is None, timestep is True
    noise_out = _get_noise_batch(None, image_shape, 4, num_batches, True)
    assert noise_out.shape == (4, 2, 3, 32, 32)
    print(noise_out.shape)

    # noise is callable, timestep is False
    noise_out = _get_noise_batch(
        lambda shape: torch.randn(*shape),
        image_shape,
        num_batches=num_batches)
    print(noise_out.shape)
    assert noise_out.shape == (2, 3, 32, 32)

    # noise is callable, timestep is True
    noise_out = _get_noise_batch(lambda shape: torch.randn(*shape),
                                 image_shape, 4, num_batches, True)
    print(noise_out.shape)
    assert noise_out.shape == (4, 2, 3, 32, 32)

    # noise is Tensor, timestep is False, noise dim = 3
    noise_inp = torch.randn(3, 32, 32)
    noise_out = _get_noise_batch(noise_inp, image_shape)
    print(noise_out.shape)
    assert noise_out.shape == (1, 3, 32, 32)

    # noise is Tensor, timestep is False, noise dim = 4
    noise_inp = torch.randn(2, 3, 32, 32)
    noise_out = _get_noise_batch(noise_inp, image_shape)
    print(noise_out.shape)
    assert noise_out.shape == (2, 3, 32, 32)

    # noise is Tensor, timestep is False, noise dim = 5
    noise_inp = torch.randn(1, 2, 3, 32, 32)
    with pytest.raises(ValueError):
        _get_noise_batch(noise_inp, image_shape)

    # noise is Tensor, timestep is True, noise dim = 4
    # noise.size(0) == num_batches
    noise_inp = torch.randn(4, 3, 32, 32)
    noise_out = _get_noise_batch(
        noise_inp,
        image_shape,
        num_timesteps=6,
        num_batches=4,
        timesteps_noise=True)
    print(noise_out.shape)
    assert noise_out.shape == (6, 4, 3, 32, 32)
    assert all([(noise_inp == noise).all() for noise in noise_out])

    # noise is Tensor, timestep is True, noise dim = 4
    # noise.size(0) == num_timesteps
    noise_inp = torch.randn(6, 3, 32, 32)
    noise_out = _get_noise_batch(
        noise_inp,
        image_shape,
        num_timesteps=6,
        num_batches=4,
        timesteps_noise=True)
    print(noise_out.shape)
    assert noise_out.shape == (6, 4, 3, 32, 32)
    assert all([(noise_inp == noise_out[:, idx, ...]).all()
                for idx in range(4)])

    # noise is Tensor, timestep is True, noise dim = 4
    # noise.size(0) == num_timesteps * num_batches
    noise_inp = torch.randn(24, 3, 32, 32)
    noise_out = _get_noise_batch(
        noise_inp,
        image_shape,
        num_timesteps=6,
        num_batches=4,
        timesteps_noise=True)
    print(noise_out.shape)
    assert noise_out.shape == (6, 4, 3, 32, 32)
    assert all([(noise_inp[idx] == noise_out[idx // 4][idx % 4]).all()
                for idx in range(24)])

    # noise is Tensor, timestep is True, noise dim = 4
    # noise_out.size(0) != num_batches * num_timesteps
    noise_inp = torch.randn(25, 3, 32, 32)
    with pytest.raises(ValueError):
        _get_noise_batch(
            noise_inp,
            image_shape,
            num_timesteps=6,
            num_batches=4,
            timesteps_noise=True)

    # noise is Tensor, timestep is True, noise dim = 5
    noise_inp = torch.randn(6, 4, 3, 32, 32)
    noise_out = _get_noise_batch(
        noise_inp,
        image_shape,
        num_timesteps=6,
        num_batches=4,
        timesteps_noise=True)
    print(noise_out.shape)
    assert noise_out.shape == (6, 4, 3, 32, 32)
    assert (noise_out == noise_inp).all()

    # noise is Tensor, timestep is True, noise dim = 6
    noise_inp = torch.randn(1, 6, 4, 3, 32, 32)
    with pytest.raises(ValueError):
        noise_out = _get_noise_batch(
            noise_inp,
            image_shape,
            num_timesteps=6,
            num_batches=4,
            timesteps_noise=True)


def test_ddpm_label_batch_utils():
    # num_classes = 0
    label_out = _get_label_batch(
        label=None, num_timesteps=2, num_classes=0, num_batches=2)
    assert label_out is None

    # num_classes = 0, label is not None
    with pytest.raises(AssertionError):
        label_out = _get_label_batch(
            label=torch.randint(0, 10, (2, )),
            num_timesteps=2,
            num_classes=0,
            num_batches=2)

    # label is None, timestep is False
    label_out = _get_label_batch(None, num_classes=10, num_batches=2)
    assert label_out.shape == (2, )
    assert torch.logical_and(label_out >= 0, label_out < 10).all()

    # label is None, timestep is True
    label_out = _get_label_batch(
        None,
        num_classes=10,
        num_batches=2,
        num_timesteps=4,
        timesteps_noise=True)
    assert label_out.shape == (4, 2)
    assert torch.logical_and(label_out >= 0, label_out < 10).all()

    # label is callable, timestep is False
    label_out = _get_label_batch(
        lambda shape: torch.randint(0, 10, shape),
        num_classes=10,
        num_batches=2)
    assert label_out.shape == (2, )
    assert torch.logical_and(label_out >= 0, label_out < 10).all()

    # label is callable, timestep is True
    label_out = _get_label_batch(
        lambda shape: torch.randint(0, 10, shape),
        num_classes=10,
        num_timesteps=4,
        num_batches=2,
        timesteps_noise=True)
    assert label_out.shape == (4, 2)
    assert torch.logical_and(label_out >= 0, label_out < 10).all()

    # label is tensor, timestep is False, label dim = 1
    label_inp = torch.LongTensor([4, 3])
    label_out = _get_label_batch(label_inp, num_classes=10)
    assert label_out.shape == (2, )
    assert (label_out == label_inp).all()

    # label is tensor, timestep is False, label dim = 0
    label_inp = torch.from_numpy(np.array(10))
    label_out = _get_label_batch(label_inp, num_classes=10)
    assert label_out.shape == (1, )

    # label is tensor, timestep is False, label dim = 2
    label_inp = torch.randint(0, 10, (4, 2))
    with pytest.raises(ValueError):
        _get_label_batch(label_inp, num_classes=10, num_batches=2)

    # label is tensor, timestep is True, label dim = 1
    # label.size(0) == num_batches
    label_inp = torch.randint(0, 10, (2, ))
    label_out = _get_label_batch(
        label_inp,
        num_timesteps=4,
        num_batches=2,
        num_classes=10,
        timesteps_noise=True)
    assert label_out.shape == (4, 2)
    assert all([(label_out[idx] == label_inp).all() for idx in range(4)])

    # label is tensor, timestep is True, label dim = 1
    # label.size(0) == num_timesteps
    label_inp = torch.randint(0, 10, (4, ))
    label_out = _get_label_batch(
        label_inp,
        num_timesteps=4,
        num_batches=2,
        num_classes=10,
        timesteps_noise=True)
    assert label_out.shape == (4, 2)
    assert all([(label_inp == label_out[:, idx]).all() for idx in range(2)])

    # label is tensor, timestep is True, label dim = 1
    # label.size(0) == num_timesteps * num_batches
    label_inp = torch.randint(0, 10, (8, ))
    label_out = _get_label_batch(
        label_inp,
        num_timesteps=4,
        num_batches=2,
        num_classes=10,
        timesteps_noise=True)
    assert label_out.shape == (4, 2)
    assert all([(label_inp[idx] == label_out[idx // 2][idx % 2]).all()
                for idx in range(8)])

    # label is tensor, timestep is True, label dim = 1
    # label.size(0) != num_timesteps * num_batches
    label_inp = torch.randint(0, 10, (9, ))
    with pytest.raises(ValueError):
        _get_label_batch(
            label_inp,
            num_timesteps=4,
            num_batches=2,
            num_classes=10,
            timesteps_noise=True)

    # label is tensor, timestep is True, label dim = 2
    label_inp = torch.randint(0, 10, (4, 2))
    label_out = _get_label_batch(
        label_inp,
        num_timesteps=4,
        num_batches=2,
        num_classes=10,
        timesteps_noise=True)
    assert label_out.shape == (4, 2)
    assert (label_out == label_inp).all()

    # label is tensor, timestep is True, label dim = 3
    label_inp = torch.randint(0, 10, (4, 2, 1))
    with pytest.raises(ValueError):
        _get_label_batch(
            label_inp,
            num_timesteps=4,
            num_batches=2,
            num_classes=10,
            timesteps_noise=True)
