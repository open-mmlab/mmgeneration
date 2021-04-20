model = dict(
    type='StyleGANV1',
    generator=dict(
        type='StyleGANv1Generator', out_size=None, style_channels=512),
    discriminator=dict(type='StyleGAN1Discriminator', in_size=None),
    gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
    disc_auxiliary_loss=[
        dict(
            type='R1GradientPenalty',
            loss_weight=10,
            norm_mode='HWC',
            data_info=dict(
                discriminator='disc_partial', real_data='real_imgs'))
    ])

train_cfg = dict(
    use_ema=True,
    transition_kimgs=600,
    optimizer_cfg=dict(
        generator=dict(type='Adam', lr=0.001, betas=(0.0, 0.99)),
        discriminator=dict(type='Adam', lr=0.001, betas=(0.0, 0.99))),
    g_lr_base=0.001,
    d_lr_base=0.001,
    g_lr_schedule=dict({
        '128': 0.0015,
        '256': 0.002,
        '512': 0.003,
        '1024': 0.003
    }),
    d_lr_schedule=dict({
        '128': 0.0015,
        '256': 0.002,
        '512': 0.003,
        '1024': 0.003
    }))

test_cfg = None
optimizer = None
