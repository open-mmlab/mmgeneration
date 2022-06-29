_base_ = ['./stylegan3_base.py']

synthesis_cfg = {
    'type': 'SynthesisNetwork',
    'channel_base': 65536,
    'channel_max': 1024,
    'magnitude_ema_beta': 0.999,
    'conv_kernel': 1,
    'use_radial_filters': True
}
model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(
        type='StyleGANv3Generator',
        noise_size=512,
        style_channels=512,
        out_size=512,
        img_channels=3,
        rgb2bgr=True,
        synthesis_cfg=synthesis_cfg),
    discriminator=dict(type='StyleGAN2Discriminator', in_size=512))
