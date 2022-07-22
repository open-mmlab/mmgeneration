_base_ = ['./stylegan3_base.py']

synthesis_cfg = {
    'type': 'SynthesisNetwork',
    'channel_base': 65536,
    'channel_max': 1024,
    'magnitude_ema_beta': 0.999,
    'conv_kernel': 1,
    'use_radial_filters': True
}

r1_gamma = 32.8
d_reg_interval = 16

model = dict(
    generator=dict(
        out_size=1024,
        img_channels=3,
        synthesis_cfg=synthesis_cfg,
        rgb2bgr=True),
    discriminator=dict(type='StyleGAN2Discriminator', in_size=1024))

train_cfg = train_dataloader = optim_wrapper = None

metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema')
]
val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
