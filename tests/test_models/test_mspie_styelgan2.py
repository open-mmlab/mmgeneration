from copy import deepcopy

import torch

from mmgen.models.gans.mspie_stylegan2 import MSPIEStyleGAN2


class TestMSStyleGAN2:

    @classmethod
    def setup_class(cls):
        cls.generator_cfg = dict(
            type='MSStyleGANv2Generator', out_size=32, style_channels=16)
        cls.disc_cfg = dict(
            type='MSStyleGAN2Discriminator',
            in_size=32,
            with_adaptive_pool=True)
        cls.gan_loss = dict(type='GANLoss', gan_type='vanilla')
        cls.disc_auxiliary_loss = dict(
            type='R1GradientPenalty',
            loss_weight=10. / 2.,
            interval=1,
            norm_mode='HWC',
            data_info=dict(real_data='real_imgs', discriminator='disc'))

        cls.train_cfg = dict(
            use_ema=True,
            num_upblocks=3,
            multi_input_scales=[0, 2, 4],
            multi_scale_probability=[0.5, 0.25, 0.25])

    def test_msstylegan2_cpu(self):
        stylegan2 = MSPIEStyleGAN2(
            self.generator_cfg,
            self.disc_cfg,
            self.gan_loss,
            self.disc_auxiliary_loss,
            None,
            train_cfg=self.train_cfg,
            test_cfg=None)

        optimizer_g = torch.optim.SGD(
            stylegan2.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            stylegan2.discriminator.parameters(), lr=0.01)
        optim_dict = dict(generator=optimizer_g, discriminator=optimizer_d)

        data = torch.randn((2, 3, 16, 16))
        data_input = dict(real_img=data)

        model_outputs = stylegan2.train_step(data_input, optim_dict)
        assert 'results' in model_outputs
        assert 'log_vars' in model_outputs
        assert model_outputs['num_samples'] == 2

        cfg_ = deepcopy(self.train_cfg)
        cfg_['disc_steps'] = 2

        stylegan2 = MSPIEStyleGAN2(
            self.generator_cfg,
            self.disc_cfg,
            self.gan_loss,
            self.disc_auxiliary_loss,
            None,
            train_cfg=cfg_,
            test_cfg=None)

        optimizer_g = torch.optim.SGD(
            stylegan2.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            stylegan2.discriminator.parameters(), lr=0.01)
        optim_dict = dict(generator=optimizer_g, discriminator=optimizer_d)

        data = torch.randn((2, 3, 16, 16))
        data_input = dict(real_img=data)

        model_outputs = stylegan2.train_step(data_input, optim_dict)
        assert 'results' in model_outputs
        assert 'log_vars' in model_outputs
        assert model_outputs['num_samples'] == 2
