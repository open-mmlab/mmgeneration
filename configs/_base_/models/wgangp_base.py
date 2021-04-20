model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(type='WGANGPGenerator', noise_size=128, out_scale=128),
    discriminator=dict(
        type='WGANGPDiscriminator',
        in_channel=3,
        in_scale=128,
        conv_module_cfg=dict(
            conv_cfg=None,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
            norm_cfg=dict(type='GN'),
            order=('conv', 'norm', 'act'))),
    gan_loss=dict(type='GANLoss', gan_type='wgan'),
    disc_auxiliary_loss=[
        dict(
            type='GradientPenaltyLoss',
            loss_weight=10,
            norm_mode='HWC',
            data_info=dict(
                discriminator='disc',
                real_data='real_imgs',
                fake_data='fake_imgs'))
    ])

train_cfg = dict(disc_steps=5)

test_cfg = None

optimizer = dict(
    generator=dict(type='Adam', lr=0.0001, betas=(0.5, 0.9)),
    discriminator=dict(type='Adam', lr=0.0001, betas=(0.5, 0.9)))
