_base_ = [
    '../_base_/models/stylegan/stylegan3_base.py',
    '../_base_/datasets/ffhq_flip.py', '../_base_/default_runtime.py'
]

synthesis_cfg = {
    'type': 'SynthesisNetwork',
    'channel_base': 32768,
    'channel_max': 512,
    'magnitude_ema_beta': 0.999
}
r1_gamma = 6.6  # set by user
d_reg_interval = 16

load_from = 'https://download.openmmlab.com/mmgen/stylegan3/stylegan3_t_ffhq_1024_b4x8_cvt_official_rgb_20220329_235113-db6c6580.pth'  # noqa
# ada settings
aug_kwargs = {
    'xflip': 1,
    'rotate90': 1,
    'xint': 1,
    'scale': 1,
    'rotate': 1,
    'aniso': 1,
    'xfrac': 1,
    'brightness': 1,
    'contrast': 1,
    'lumaflip': 1,
    'hue': 1,
    'saturation': 1
}

model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(
        out_size=1024,
        img_channels=3,
        rgb2bgr=True,
        synthesis_cfg=synthesis_cfg),
    discriminator=dict(
        type='ADAStyleGAN2Discriminator',
        in_size=1024,
        input_bgr2rgb=True,
        data_aug=dict(type='ADAAug', aug_pipeline=aug_kwargs, ada_kimg=100)),
    gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
    disc_auxiliary_loss=dict(loss_weight=r1_gamma / 2.0 * d_reg_interval))

imgs_root = 'data/metfaces/images/'
data = dict(
    samples_per_gpu=4,
    train=dict(dataset=dict(imgs_root=imgs_root)),
    val=dict(imgs_root=imgs_root))

ema_half_life = 10.  # G_smoothing_kimg

ema_kimg = 10
ema_nimg = ema_kimg * 1000
ema_beta = 0.5**(32 / max(ema_nimg, 1e-8))

custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=5000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interp_mode='lerp',
        interp_cfg=dict(momentum=ema_beta),
        interval=1,
        start_iter=0,
        priority='VERY_HIGH')
]

inception_pkl = 'work_dirs/inception_pkl/metface_1024x1024_noflip.pkl'
metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl=inception_pkl,
        inception_args=dict(type='StyleGAN'),
        bgr2rgb=True))

evaluation = dict(
    type='GenerativeEvalHook',
    interval=dict(milestones=[80000], interval=[10000, 5000]),
    metrics=dict(
        type='FID',
        num_images=50000,
        inception_pkl=inception_pkl,
        inception_args=dict(type='StyleGAN'),
        bgr2rgb=True),
    sample_kwargs=dict(sample_model='ema'))

lr_config = None

total_iters = 160000
