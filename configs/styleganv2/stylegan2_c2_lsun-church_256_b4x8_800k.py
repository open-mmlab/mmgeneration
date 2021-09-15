"""Note that this config is just for testing."""

_base_ = [
    '../_base_/datasets/lsun_stylegan.py',
    '../_base_/models/stylegan/stylegan2_base.py',
    '../_base_/default_runtime.py'
]

model = dict(generator=dict(out_size=256), discriminator=dict(in_size=256))

data = dict(
    samples_per_gpu=4,
    train=dict(dataset=dict(imgs_root='./data/lsun-church')))

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

checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=30)
lr_config = None

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

total_iters = 800002  # need to modify

metrics = dict(
    fid50k=dict(
        type='FID', num_images=50000, inception_pkl=None, bgr2rgb=True),
    pr50k3=dict(type='PR', num_images=50000, k=3),
    ppl_wend=dict(type='PPL', space='W', sampling='end', num_images=50000))
