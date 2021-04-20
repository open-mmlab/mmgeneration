# define GAN model
model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(type='DCGANGenerator', output_scale=64, base_channels=1024),
    discriminator=dict(
        type='DCGANDiscriminator',
        input_scale=64,
        output_scale=4,
        out_channels=1),
    gan_loss=dict(type='GANLoss', gan_type='vanilla'))

train_cfg = dict(disc_steps=1)
test_cfg = None

# define optimizer
optimizer = dict(
    generator=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
    discriminator=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))
