# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch

from mmgen.models import DenoisingUnet, build_module


class TestDDPM:

    @classmethod
    def setup_class(cls):
        cls.denoising_cfg = dict(
            type='DenoisingUnet',
            image_size=32,
            in_channels=3,
            base_channels=128,
            resblocks_per_downsample=3,
            attention_res=[16, 8],
            use_scale_shift_norm=True,
            dropout=0,
            num_heads=4,
            use_rescale_timesteps=True,
            output_cfg=dict(mean='eps', var='learned_range'),
            num_timesteps=2000)
        cls.x_t = torch.randn(2, 3, 32, 32)
        cls.timesteps = torch.LongTensor([999, 1999])
        cls.label = torch.randint(0, 10, (2, ))

    def test_denoising_cpu(self):
        # test default config
        denoising = build_module(self.denoising_cfg)
        assert isinstance(denoising, DenoisingUnet)
        output_dict = denoising(self.x_t, self.timesteps, return_noise=True)
        assert 'eps_t_pred' in output_dict
        assert 'factor' in output_dict
        assert 'x_t' in output_dict
        assert 't_rescaled' in output_dict
        assert (output_dict['x_t'] == self.x_t).all()
        assert (output_dict['t_rescaled'] < 1000).all()
        assert (output_dict['factor'] < 1).all()
        assert (output_dict['factor'] > 0).all()

        # test image size --> list input
        config = deepcopy(self.denoising_cfg)
        config['image_size'] = [32, 32]
        output_dict = denoising(self.x_t, self.timesteps)
        assert 'eps_t_pred' in output_dict
        assert 'factor' in output_dict
        assert output_dict['eps_t_pred'].shape == (2, 3, 32, 32)

        # test image size --> raise type error
        config = deepcopy(self.denoising_cfg)
        config['image_size'] = '32'
        with pytest.raises(TypeError):
            build_module(config)

        # test image size --> wrong list length
        config = deepcopy(self.denoising_cfg)
        config['image_size'] = [32, 32, 32]
        with pytest.raises(AssertionError):
            build_module(config)

        # test image size --> wrong list element
        config = deepcopy(self.denoising_cfg)
        config['image_size'] = [32, 64]
        with pytest.raises(AssertionError):
            build_module(config)

        # test channels_cfg --> list
        config = deepcopy(self.denoising_cfg)
        config['channels_cfg'] = [1, 2, 2, 2]
        denoising = build_module(config)
        assert isinstance(denoising, DenoisingUnet)
        output_dict = denoising(self.x_t, self.timesteps)

        # test channels_cfg --> dict
        config = deepcopy(self.denoising_cfg)
        config['channels_cfg'] = {32: [1, 2, 2, 2, 2]}
        denoising = build_module(config)
        output_dict = denoising(self.x_t, self.timesteps)
        assert 'eps_t_pred' in output_dict
        assert 'factor' in output_dict
        assert (output_dict['factor'] < 1).all()
        assert (output_dict['factor'] > 0).all()

        # test channels_cfg --> no image size error
        config = deepcopy(self.denoising_cfg)
        config['image_size'] = 194
        with pytest.raises(KeyError):
            denoising = build_module(config)

        # test channels_cfg --> wrong type error
        config = deepcopy(self.denoising_cfg)
        config['channels_cfg'] = '1, 2, 2, 2'
        with pytest.raises(ValueError):
            denoising = build_module(config)

        # test use rescale timesteps
        config = deepcopy(self.denoising_cfg)
        config['use_rescale_timesteps'] = False
        denoising = build_module(config)
        output_dict = denoising(self.x_t, self.timesteps, return_noise=True)
        assert (output_dict['t_rescaled'] == self.timesteps).all()

        # test var_mode --> LEARNED
        config = deepcopy(self.denoising_cfg)
        config['output_cfg']['var'] = 'LEARNED'
        denoising = build_module(config)
        output_dict = denoising(self.x_t, self.timesteps, return_noise=True)
        assert 'logvar' in output_dict

        # test var_mode --> FIXED
        config = deepcopy(self.denoising_cfg)
        config['output_cfg']['var'] = 'FIXED_SMALL'
        denoising = build_module(config)
        output_dict = denoising(self.x_t, self.timesteps, return_noise=True)
        assert 'factor' not in output_dict and 'logvar' not in output_dict

        # test var_mode --> raise error
        config = deepcopy(self.denoising_cfg)
        config['output_cfg']['var'] = 'ERROR'
        denoising = build_module(config)
        with pytest.raises(AttributeError):
            output_dict = denoising(
                self.x_t, self.timesteps, return_noise=True)

        # test mean_mode --> START_X
        config = deepcopy(self.denoising_cfg)
        config['output_cfg']['mean'] = 'START_X'
        denoising = build_module(config)
        output_dict = denoising(self.x_t, self.timesteps, return_noise=True)
        assert 'x_0_pred' in output_dict

        # test mean_mode --> START_X
        config = deepcopy(self.denoising_cfg)
        config['output_cfg']['mean'] = 'PREVIOUS_X'
        denoising = build_module(config)
        output_dict = denoising(self.x_t, self.timesteps, return_noise=True)
        # print(output_dict.keys())
        assert 'x_tm1_pred' in output_dict

        # test var_mode --> raise error
        config = deepcopy(self.denoising_cfg)
        config['output_cfg']['mean'] = 'ERROR'
        denoising = build_module(config)
        with pytest.raises(AttributeError):
            output_dict = denoising(
                self.x_t, self.timesteps, return_noise=True)

        # test timestep embedding --> raise error
        config = deepcopy(self.denoising_cfg)
        config['time_embedding_mode'] = 'cos'
        with pytest.raises(ValueError):
            denoising = build_module(config)

        # test timestep embedding --> new config
        config = deepcopy(self.denoising_cfg)
        config['time_embedding_cfg'] = dict(max_period=1000)
        denoising = build_module(config)

        # test class-conditional denoising
        config = deepcopy(self.denoising_cfg)
        config['num_classes'] = 10
        denoising = build_module(config)
        output_dict = denoising(
            self.x_t, self.timesteps, self.label, return_noise=True)
        assert 'label' in output_dict
        assert (output_dict['label'] == self.label).all()
