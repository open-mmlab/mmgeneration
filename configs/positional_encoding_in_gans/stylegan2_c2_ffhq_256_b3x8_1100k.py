"""Config for the `config-f` setting in StyleGAN2."""

_base_ = [
    '../_base_/datasets/ffhq_flip.py',
    '../_base_/models/stylegan/stylegan2_base.py',
    '../_base_/default_runtime.py'
]

model = dict(generator=dict(out_size=256), discriminator=dict(in_size=256))

data = dict(
    samples_per_gpu=3,
    train=dict(dataset=dict(imgs_root='./data/ffhq/ffhq_imgs/ffhq_256')))

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

metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl='work_dirs/inception_pkl/ffhq-256-50k-rgb.pkl',
        bgr2rgb=True),
    pr10k3=dict(type='PR', num_images=10000, k=3))

checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=30)
lr_config = None

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

total_iters = 1100002
