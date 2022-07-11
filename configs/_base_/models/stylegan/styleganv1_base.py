model = dict(
    type='StyleGANV1',
    data_preprocessor=dict(type='GANDataPreprocessor'),
    style_channels=512,
    generator=dict(
        type='StyleGANv1Generator', out_size=None, style_channels=512),
    discriminator=dict(type='StyleGAN1Discriminator', in_size=None))

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
