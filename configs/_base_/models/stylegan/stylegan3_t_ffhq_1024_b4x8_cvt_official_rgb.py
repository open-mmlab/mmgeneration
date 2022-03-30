_base_ = ['./stylegan3_base.py']

synthesis_cfg = {
    'type': 'SynthesisNetwork',
    'channel_base': 32768,
    'channel_max': 512,
    'magnitude_ema_beta': 0.999
}

model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(
        out_size=1024,
        img_channels=3,
        synthesis_cfg=synthesis_cfg,
        rgb2bgr=True),
    discriminator=dict(in_size=1024))
