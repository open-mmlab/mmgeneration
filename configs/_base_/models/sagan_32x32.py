# define GAN model
model = dict(
    type='BasiccGAN',
    generator=dict(type='SAGANGenerator', output_scale=128, base_channels=256),
    discriminator=dict(
        type='ProjDiscriminator', input_scale=128, base_channels=128),
    gan_loss=dict(type='GANLoss', gan_type='hinge'))

# hinge loss
train_cfg = dict(disc_steps=1)
test_cfg = None

# define optimizer
optimizer = dict(
    generator=dict(type='Adam', lr=0.0002, betas=(0.0, 0.999)),
    discriminator=dict(type='Adam', lr=0.0002, betas=(0.0, 0.999)))
