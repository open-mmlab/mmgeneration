_base_ = ['./stylegan3_base.py']

synthesis_cfg = {
    'type': 'SynthesisNetwork',
    'channel_base': 32768,
    'channel_max': 1024,
    'magnitude_ema_beta': 0.999,
    'conv_kernel': 1,
    'use_radial_filters': True
}
model = dict(
    generator=dict(
        out_size=256,
        img_channels=3,
        rgb2bgr=True,
        synthesis_cfg=synthesis_cfg),
    discriminator=dict(in_size=256, channel_multiplier=1))

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
