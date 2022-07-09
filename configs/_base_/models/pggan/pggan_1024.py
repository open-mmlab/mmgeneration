# define GAN model
model = dict(
    type='ProgressiveGrowingGAN',
    data_preprocessor=dict(type='GANDataPreprocessor'),
    noise_size=512,
    generator=dict(type='PGGANGenerator', out_scale=1024, noise_size=512),
    discriminator=dict(type='PGGANDiscriminator', in_scale=1024),
    nkimgs_per_scale={
        '4': 600,
        '8': 1200,
        '16': 1200,
        '32': 1200,
        '64': 1200,
        '128': 1200,
        '256': 1200,
        '512': 1200,
        '1024': 12000,
    },
    transition_kimgs=600,
    ema_config=dict(interval=1))
