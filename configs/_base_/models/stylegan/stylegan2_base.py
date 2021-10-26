# define GAN model

d_reg_interval = 16
g_reg_interval = 4

g_reg_ratio = g_reg_interval / (g_reg_interval + 1)
d_reg_ratio = d_reg_interval / (d_reg_interval + 1)

model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(
        type='StyleGANv2Generator',
        out_size=None,  # Need to be set.
        style_channels=512,
    ),
    discriminator=dict(
        type='StyleGAN2Discriminator',
        in_size=None,  # Need to be set.
    ),
    gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
    disc_auxiliary_loss=dict(
        type='R1GradientPenalty',
        loss_weight=10. / 2. * d_reg_interval,
        interval=d_reg_interval,
        norm_mode='HWC',
        data_info=dict(real_data='real_imgs', discriminator='disc')),
    gen_auxiliary_loss=dict(
        type='GeneratorPathRegularizer',
        loss_weight=2. * g_reg_interval,
        pl_batch_shrink=2,
        interval=g_reg_interval,
        data_info=dict(generator='gen', num_batches='batch_size')))

train_cfg = dict(use_ema=True)
test_cfg = None

# define optimizer
optimizer = dict(
    generator=dict(
        type='Adam', lr=0.002 * g_reg_ratio, betas=(0, 0.99**g_reg_ratio)),
    discriminator=dict(
        type='Adam', lr=0.002 * d_reg_ratio, betas=(0, 0.99**d_reg_ratio)))
