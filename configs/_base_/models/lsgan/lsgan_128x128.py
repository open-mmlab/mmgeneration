# define GAN model
model = dict(
    type='LSGAN',
    noise_size=1024,
    data_preprocessor=dict(type='GANDataPreprocessor'),
    generator=dict(
        type='LSGANGenerator',
        output_scale=128,
        base_channels=256,
        noise_size=1024),
    discriminator=dict(
        type='LSGANDiscriminator', input_scale=128, base_channels=64))
