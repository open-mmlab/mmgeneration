# define GAN model
model = dict(
    type='BasiccGAN',
    generator=dict(type='SNGANGenerator', output_scale=128, base_channels=64),
    discriminator=dict(
        type='ProjDiscriminator', input_scale=128, base_channels=64),
    gan_loss=dict(type='GANLoss', gan_type='hinge'))

train_cfg = dict(disc_steps=2)
test_cfg = None

# define optimizer
optimizer = dict(
    generator=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
    discriminator=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))
