_base_ = [
    '../_base_/models/stylegan/stylegan3_base.py',
    '../_base_/datasets/ffhq_flip.py', '../_base_/default_runtime.py'
]

synthesis_kwargs = {
    'channel_base': 65536,
    'channel_max': 1024,
    'magnitude_ema_beta': 0.9998613801725043,
    'conv_kernel': 1,
    'use_radial_filters': True
}
model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(
        type='StyleGANv3Generator',
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=512,
        img_channels=3,
        rgb2bgr=True,
        **synthesis_kwargs),
    discriminator=dict(type='StyleGAN2Discriminator', in_size=512),
    gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
    disc_auxiliary_loss=[
        dict(
            type='R1GradientPenalty',
            loss_weight=10,
            norm_mode='HWC',
            data_info=dict(
                discriminator='disc_partial', real_data='real_imgs'))
    ])

data = dict(
    samples_per_gpu=4,
    train=dict(dataset=dict(imgs_root='data/afhq_v2/train/total')),
    val=dict(imgs_root='data/afhq_v2/train/total'))

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
        num_images=8,
        inception_pkl='work_dirs/inception_pkl/afhqv2-512-whole-rgb.pkl',
        inception_args=dict(type='StyleGAN'),
        # inception_args=dict(type='pytorch'),
        bgr2rgb=True))

evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=dict(
        type='FID',
        num_images=50000,
        inception_pkl='work_dirs/inception_pkl/afhqv2-512-whole-rgb.pkl',
        bgr2rgb=True),
    sample_kwargs=dict(sample_model='ema'))

checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=30)
lr_config = None

total_iters = 800002  # TODO: fix it
