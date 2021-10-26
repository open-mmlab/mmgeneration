_base_ = [
    '../_base_/datasets/ffhq_flip.py',
    '../_base_/models/stylegan/stylegan2_base.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='MSPIEStyleGAN2',
    generator=dict(
        type='MSStyleGANv2Generator',
        head_pos_encoding=dict(
            type='SPE',
            embedding_dim=256,
            padding_idx=0,
            init_size=256,
            center_shift=100),
        deconv2conv=True,
        up_after_conv=True,
        up_config=dict(scale_factor=2, mode='bilinear', align_corners=True),
        out_size=256),
    discriminator=dict(
        type='MSStyleGAN2Discriminator', in_size=256, with_adaptive_pool=True))

train_cfg = dict(
    num_upblocks=6,
    multi_input_scales=[0, 2, 4],
    multi_scale_probability=[0.5, 0.25, 0.25])

data = dict(
    samples_per_gpu=3,
    train=dict(dataset=dict(imgs_root='./data/ffhq/ffhq_imgs/ffhq_512')))

ema_half_life = 10.
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

checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=40)
lr_config = None

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

cudnn_benchmark = False
total_iters = 1100002

metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl='work_dirs/inception_pkl/ffhq-256-50k-rgb.pkl',
        bgr2rgb=True),
    pr10k3=dict(type='PR', num_images=10000, k=3))
