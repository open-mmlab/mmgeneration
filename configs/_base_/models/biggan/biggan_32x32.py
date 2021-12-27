model = dict(
    type='BasiccGAN',
    num_classes=10,
    generator=dict(
        type='BigGANGenerator',
        output_scale=32,
        noise_size=128,
        num_classes=10,
        base_channels=64,
        with_shared_embedding=False,
        sn_eps=1e-8,
        sn_style='torch',
        init_type='N02',
        split_noise=False,
        auto_sync_bn=False),
    discriminator=dict(
        type='BigGANDiscriminator',
        input_scale=32,
        num_classes=10,
        base_channels=64,
        sn_eps=1e-8,
        sn_style='torch',
        init_type='N02',
        with_spectral_norm=True),
    gan_loss=dict(type='GANLoss', gan_type='hinge'))

train_cfg = dict(
    disc_steps=4, gen_steps=1, batch_accumulation_steps=1, use_ema=True)
test_cfg = None
optimizer = dict(
    generator=dict(type='Adam', lr=0.0002, betas=(0.0, 0.999)),
    discriminator=dict(type='Adam', lr=0.0002, betas=(0.0, 0.999)))
