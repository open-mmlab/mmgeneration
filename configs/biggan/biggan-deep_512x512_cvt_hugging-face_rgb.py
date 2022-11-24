model = dict(
    type='BasiccGAN',
    generator=dict(
        type='BigGANDeepGenerator',
        output_scale=512,
        noise_size=128,
        num_classes=1000,
        base_channels=128,
        shared_dim=128,
        with_shared_embedding=True,
        sn_eps=1e-6,
        sn_style='torch',
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        concat_noise=True,
        auto_sync_bn=False,
        rgb2bgr=True),
    discriminator=dict(
        type='BigGANDeepDiscriminator',
        input_scale=512,
        num_classes=1000,
        base_channels=128,
        sn_eps=1e-6,
        sn_style='torch',
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        with_spectral_norm=True),
    gan_loss=dict(type='GANLoss', gan_type='hinge'))

train_cfg = dict(
    disc_steps=8, gen_steps=1, batch_accumulation_steps=8, use_ema=True)
test_cfg = None
optimizer = dict(
    generator=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999), eps=1e-6),
    discriminator=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-6))
