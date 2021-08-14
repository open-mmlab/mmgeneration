# define GAN model
model = dict(
    type='BasiccGAN',
    generator=dict(
        type='SAGANGenerator',
        output_scale=32,
        base_channels=256,
        attention_cfg=dict(type='SelfAttentionBlock'),
        attention_after_nth_block=2,
        with_spectral_norm=True),
    discriminator=dict(
        type='ProjDiscriminator',
        input_scale=32,
        base_channels=128,
        attention_cfg=dict(type='SelfAttentionBlock'),
        attention_after_nth_block=1,
        with_spectral_norm=True),
    gan_loss=dict(type='GANLoss', gan_type='hinge'))

train_cfg = dict(disc_steps=5)
test_cfg = None

# define optimizer
optimizer = dict(
    generator=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
    discriminator=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))
