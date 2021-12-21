# define GAN model
model = dict(
    type='ProgressiveGrowingGAN',
    generator=dict(type='PGGANGenerator', out_scale=1024, noise_size=512),
    discriminator=dict(type='PGGANDiscriminator', in_scale=1024),
    gan_loss=dict(type='GANLoss', gan_type='wgan'),
    disc_auxiliary_loss=[
        dict(
            type='DiscShiftLoss',
            loss_weight=0.001 * 0.5,
            data_info=dict(pred='disc_pred_fake')),
        dict(
            type='DiscShiftLoss',
            loss_weight=0.001 * 0.5,
            data_info=dict(pred='disc_pred_real')),
        dict(
            type='GradientPenaltyLoss',
            loss_weight=10,
            norm_mode='HWC',
            data_info=dict(
                discriminator='disc_partial',
                real_data='real_imgs',
                fake_data='fake_imgs'))
    ])

train_cfg = dict(
    use_ema=True,
    nkimgs_per_scale={
        '4': 600,
        '8': 1200,
        '16': 1200,
        '32': 1200,
        '64': 1200,
        '128': 1200,
        '256': 1200,
        '512': 1200,
        '1024': 12000,
    },
    transition_kimgs=600,
    optimizer_cfg=dict(
        generator=dict(type='Adam', lr=0.001, betas=(0., 0.99)),
        discriminator=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
    g_lr_base=0.001,
    d_lr_base=0.001,
    g_lr_schedule={
        '128': 0.0015,
        '256': 0.002,
        '512': 0.003,
        '1024': 0.003
    },
    d_lr_schedule={
        '128': 0.0015,
        '256': 0.002,
        '512': 0.003,
        '1024': 0.003
    })

test_cfg = None
