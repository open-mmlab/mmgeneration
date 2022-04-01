_base_ = [
    '../_base_/models/stylegan/stylegan3_base.py',
    '../_base_/datasets/unconditional_imgs_flip_lanczos_resize_256x256.py',
    '../_base_/default_runtime.py'
]

synthesis_cfg = {
    'type': 'SynthesisNetwork',
    'channel_base': 16384,
    'channel_max': 512,
    'magnitude_ema_beta': 0.999
}
r1_gamma = 2.  # set by user
d_reg_interval = 16

model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(out_size=256, img_channels=3, synthesis_cfg=synthesis_cfg),
    discriminator=dict(in_size=256, channel_multiplier=1),
    gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
    disc_auxiliary_loss=dict(loss_weight=r1_gamma / 2.0 * d_reg_interval))

imgs_root = 'data/ffhq/images'
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
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='rampup',
        momentum_cfg=dict(
            ema_kimg=10, ema_rampup=0.05, batch_size=32, eps=1e-8),
        priority='VERY_HIGH')
]

inception_pkl = 'work_dirs/inception_pkl/ffhq-lanczos-256x256.pkl'
metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl=inception_pkl,
        inception_args=dict(type='StyleGAN'),
        bgr2rgb=True))

inception_path = None
evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=dict(
        type='FID',
        num_images=50000,
        inception_pkl=inception_pkl,
        inception_args=dict(type='StyleGAN', inception_path=inception_path),
        bgr2rgb=True),
    sample_kwargs=dict(sample_model='ema'))

checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=30)

lr_config = None

total_iters = 800002
