_base_ = [
    '../_base_/models/stylegan/stylegan3_base.py',
    '../_base_/datasets/unconditional_imgs_flip_512x512.py',
    '../_base_/default_runtime.py'
]

synthesis_cfg = {
    'type': 'SynthesisNetwork',
    'channel_base': 65536,
    'channel_max': 1024,
    'magnitude_ema_beta': 0.999,
    'conv_kernel': 1,
    'use_radial_filters': True
}
model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(
        type='StyleGANv3Generator',
        noise_size=512,
        style_channels=512,
        out_size=512,
        img_channels=3,
        rgb2bgr=True,
        synthesis_cfg=synthesis_cfg),
    discriminator=dict(type='StyleGAN2Discriminator', in_size=512),
    gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
    disc_auxiliary_loss=dict(loss_weight=16.4))

imgs_root = None  # set by user

data = dict(
    samples_per_gpu=4,
    train=dict(dataset=dict(imgs_root=imgs_root)),
    val=dict(imgs_root=imgs_root))

ema_half_life = 10.  # G_smoothing_kimg

custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=5000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        interp_cfg=dict(momentum=0.5**(32. / (ema_half_life * 1000.))),
        priority='VERY_HIGH')
]

inception_pkl = None
metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl=inception_pkl,
        inception_args=dict(type='StyleGAN'),
        bgr2rgb=True))

evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=dict(
        type='FID',
        num_images=50000,
        inception_pkl=inception_pkl,
        bgr2rgb=True),
    sample_kwargs=dict(sample_model='ema'))

checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=30)
lr_config = None

total_iters = 800002
