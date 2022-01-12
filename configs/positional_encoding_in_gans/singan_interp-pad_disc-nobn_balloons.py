_base_ = ['../singan/singan_balloons.py']

model = dict(
    type='PESinGAN',
    generator=dict(
        type='SinGANMSGeneratorPE', interp_pad=True, noise_with_pad=True),
    discriminator=dict(norm_cfg=None))

train_cfg = dict(fixed_noise_with_pad=True)

dist_params = dict(backend='nccl')
