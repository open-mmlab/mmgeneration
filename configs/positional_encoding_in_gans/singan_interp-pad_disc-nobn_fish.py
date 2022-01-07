_base_ = ['../singan/singan_fish.py']

model = dict(
    type='PESinGAN',
    generator=dict(
        type='SinGANMSGeneratorPE', interp_pad=True, noise_with_pad=True),
    discriminator=dict(norm_cfg=None))

train_cfg = dict(fixed_noise_with_pad=True)

data = dict(
    train=dict(
        img_path='./data/singan/fish-crop.jpg',
        min_size=25,
        max_size=300,
    ))

dist_params = dict(backend='nccl')
