_base_ = [
    '../_base_/models/stylegan/styleganv1_base.py',
    '../_base_/datasets/grow_scale_imgs_ffhq_styleganv1.py',
    '../_base_/default_runtime.py',
]

model = dict(generator=dict(out_size=1024), discriminator=dict(in_size=1024))

train_cfg = dict(
    nkimgs_per_scale={
        '8': 1200,
        '16': 1200,
        '32': 1200,
        '64': 1200,
        '128': 1200,
        '256': 1200,
        '512': 1200,
        '1024': 166000
    })

checkpoint_config = dict(interval=5000, by_epoch=False, max_keep_ckpts=20)
lr_config = None

ema_half_life = 10.  # G_smoothing_kimg

custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=5000),
    dict(type='PGGANFetchDataHook', interval=1),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        interp_cfg=dict(momentum=0.5**(32. / (ema_half_life * 1000.))),
        priority='VERY_HIGH')
]

total_iters = 670000

metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl='work_dirs/inception_pkl/ffhq-1024-50k-rgb.pkl',
        bgr2rgb=True),
    pr50k3=dict(type='PR', num_images=50000, k=3),
    ppl_wend=dict(type='PPL', space='W', sampling='end', num_images=50000))
