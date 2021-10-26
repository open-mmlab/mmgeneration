"""Config for the `config-f` setting in StyleGAN2."""

_base_ = [
    '../_base_/datasets/ffhq_flip.py',
    '../_base_/models/stylegan/stylegan2_base.py',
    '../_base_/default_runtime.py'
]

ema_half_life = 10.  # G_smoothing_kimg

model = dict(generator=dict(out_size=1024), discriminator=dict(in_size=1024))

data = dict(
    samples_per_gpu=4,
    train=dict(dataset=dict(imgs_root='./data/ffhq/images')),
    val=dict(imgs_root='./data/ffhq/images'))

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
        inception_pkl='work_dirs/inception_pkl/ffhq-1024-50k-rgb.pkl',
        bgr2rgb=True),
    pr50k3=dict(type='PR', num_images=50000, k=3),
    ppl_wend=dict(type='PPL', space='W', sampling='end', num_images=50000))

evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=dict(
        type='FID',
        num_images=50000,
        inception_pkl='work_dirs/inception_pkl/ffhq-1024-50k-rgb.pkl',
        bgr2rgb=True),
    sample_kwargs=dict(sample_model='ema'))

checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=30)
lr_config = None

total_iters = 800002
