# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import pytest
import torch

from mmgen.models.builder import build_module
from mmgen.models.losses.pixelwise_loss import (
    DiscretizedGaussianLogLikelihoodLoss, GaussianKLDLoss, MSELoss)


class TestDDPMVLBLoss:

    @classmethod
    def setup_class(cls):
        cls.gaussian_kld_data_info = dict(
            mean_pred='mean_pred',
            mean_target='mean_posterior',
            logvar_pred='logvar_pred',
            logvar_target='logvar_posterior')
        cls.disc_log_likelihood_data_info = dict(
            x='real_imgs', mean='mean_pred', logvar='logvar_pred')
        cls.config = dict(
            type='DDPMVLBLoss',
            rescale_mode='constant',
            rescale_cfg=dict(scale=4),
            data_info=cls.gaussian_kld_data_info,
            data_info_t_0=cls.disc_log_likelihood_data_info,
            log_cfgs=[
                dict(
                    type='quartile', prefix_name='loss_vlb',
                    total_timesteps=4),
                dict(type='name')
            ])
        cls.t = torch.LongTensor([0, 1, 2, 3])
        cls.tar_shape = [4, 2, 4, 4]

        cls.mean_pred = torch.randn(cls.tar_shape)
        cls.logvar_pred = torch.randn(cls.tar_shape)
        cls.mean_posterior = torch.randn(cls.tar_shape)
        cls.logvar_posterior = torch.randn(cls.tar_shape)
        cls.real_imgs = torch.randn(cls.tar_shape)
        cls.label = [0, 18, 1, 5]

        cls.output_dict = dict(
            mean_pred=cls.mean_pred,
            logvar_pred=cls.logvar_pred,
            mean_posterior=cls.mean_posterior,
            logvar_posterior=cls.logvar_posterior,
            real_imgs=cls.real_imgs,
            label=cls.label,
            meta_info=None,
            timesteps=cls.t)

        # calculate loss manually
        cls.loss_gaussian_kld = GaussianKLDLoss(
            data_info=cls.gaussian_kld_data_info,
            reduction='flatmean',
            base='2')(
                cls.output_dict)
        cls.loss_disc_likelihood = DiscretizedGaussianLogLikelihoodLoss(
            data_info=cls.disc_log_likelihood_data_info,
            reduction='flatmean',
            base='2')(
                cls.output_dict)

        cls.loss_manually = (-cls.loss_disc_likelihood[0] +
                             cls.loss_gaussian_kld[1:].sum()) / 4

        # TODO: unit test for sampler would be add later
        cls.weight = torch.rand(4, )

    def test_vlb_loss(self):
        # test forward
        config = deepcopy(self.config)
        loss_fn = build_module(config)
        loss = loss_fn(self.output_dict)
        np.allclose(loss, self.loss_manually * 4)

        # test log_cfgs --> dict input
        config = deepcopy(self.config)
        config['log_cfgs'] = dict(type='name')
        loss_fn = build_module(config)
        assert isinstance(loss_fn.log_fn_list, list)

        # test log_cfgs --> no log_cfgs
        config = deepcopy(self.config)
        config['log_cfgs'] = None
        loss_fn = build_module(config)
        loss = loss_fn(self.output_dict)
        assert not loss_fn.log_vars

        # test rescale_cfg --> rescale is None
        config = deepcopy(self.config)
        config['rescale_mode'] = None
        loss_fn = build_module(config)
        loss = loss_fn(self.output_dict)
        np.allclose(loss, self.loss_manually)

        # TODO: test rescale_cfg --> test sampler

        # test rescale_cfg --> test weight
        config = deepcopy(self.config)
        config['rescale_mode'] = 'timestep_weight'
        weight = self.weight.clone()
        loss_fn = build_module(config, default_args=dict(weight=weight))
        loss = loss_fn(self.output_dict)
        loss_weighted_manually = (
            -(self.loss_disc_likelihood * weight)[0] +
            (self.loss_gaussian_kld * weight)[1:].sum()) / 4
        np.allclose(loss, loss_weighted_manually)

        # test rescale_cfg --> change weight
        weight[0] += 1
        loss = loss_fn(self.output_dict)
        loss_weighted_manually = (
            -(self.loss_disc_likelihood * weight)[0] +
            (self.loss_gaussian_kld * weight)[1:].sum()) / 4
        np.allclose(loss, loss_weighted_manually)

        # test t = 0
        config = deepcopy(self.config)
        output_dict = deepcopy(self.output_dict)
        output_dict['timesteps'][0] = 1
        loss_fn = build_module(config)
        loss = loss_fn(output_dict)
        assert loss_fn.log_vars['loss_vlb_quartile_0'] == 0
        assert loss_fn.log_vars['loss_DiscGaussianLogLikelihood'] == 0


class TestDDPMMSELoss:

    @classmethod
    def setup_class(cls):
        cls.data_info = dict(pred='eps_t_pred', target='noise')
        cls.config = dict(
            type='DDPMMSELoss',
            data_info=cls.data_info,
            log_cfgs=dict(
                type='quartile', prefix_name='loss_mse', total_timesteps=4))

        cls.t = torch.LongTensor([0, 1, 2, 3])
        cls.tar_shape = [4, 2, 4, 4]

        cls.eps_t_pred = torch.randn(cls.tar_shape)
        cls.noise = torch.randn(cls.tar_shape)

        cls.output_dict = dict(
            eps_t_pred=cls.eps_t_pred,
            noise=cls.noise,
            meta_info=None,
            timesteps=cls.t)

        cls.weight = torch.rand(4, )

        # calculate loss manually
        cls.loss_manually = 0
        for idx in range(cls.tar_shape[0]):
            t = cls.t[idx]
            weight = cls.weight[t]
            output_dict_ = dict(
                eps_t_pred=cls.eps_t_pred[t], noise=cls.noise[t])
            cls.loss_manually += MSELoss(
                data_info=cls.data_info)(output_dict_) * weight
        cls.loss_manually /= 4

    def test_mse_loss(self):
        # test forward
        config = deepcopy(self.config)
        config['rescale_mode'] = 'timestep_weight'
        loss_fn = build_module(config, default_args=dict(weight=self.weight))
        loss = loss_fn(self.output_dict)
        np.allclose(loss, self.loss_manually)

        # test reduction raise error
        config = deepcopy(self.config)
        config['reduction'] = 'reduction'
        with pytest.raises(ValueError):
            loss_fn = build_module(config)

        # test return loss name
        config = deepcopy(self.config)
        config['loss_name'] = 'loss_name'
        loss_fn = build_module(config)
        assert loss_fn.loss_name() == 'loss_name'
