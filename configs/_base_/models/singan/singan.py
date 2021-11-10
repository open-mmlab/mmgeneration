model = dict(
    type='SinGAN',
    generator=dict(
        type='SinGANMultiScaleGenerator',
        in_channels=3,
        out_channels=3,
        num_scales=None,  # need to be specified
    ),
    discriminator=dict(
        type='SinGANMultiScaleDiscriminator',
        in_channels=3,
        num_scales=None,  # need to be specified
    ),
    gan_loss=dict(type='GANLoss', gan_type='wgan', loss_weight=1),
    disc_auxiliary_loss=[
        dict(
            type='GradientPenaltyLoss',
            loss_weight=0.1,
            norm_mode='pixel',
            data_info=dict(
                discriminator='disc_partial',
                real_data='real_imgs',
                fake_data='fake_imgs'))
    ],
    gen_auxiliary_loss=dict(
        type='MSELoss',
        loss_weight=10,
        data_info=dict(pred='recon_imgs', target='real_imgs'),
    ))

train_cfg = dict(
    noise_weight_init=0.1,
    iters_per_scale=2000,
    curr_scale=-1,
    disc_steps=3,
    generator_steps=3,
    lr_d=0.0005,
    lr_g=0.0005,
    lr_scheduler_args=dict(milestones=[1600], gamma=0.1))

test_cfg = None
