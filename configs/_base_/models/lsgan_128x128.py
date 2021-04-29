# define GAN model
model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(type='LSGANGenerator'),
    discriminator=dict(type='LSGANDiscriminator'),
    gan_loss=dict(type='GANLoss', gan_type='lsgan'))

train_cfg = dict(disc_steps=1)
test_cfg = None

# define optimizer
optimizer = dict(
    generator=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
    discriminator=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))
