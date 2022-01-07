_base_ = ['../singan/singan_fish.py']

num_scales = 10  # start from zero
model = dict(
    type='PESinGAN',
    generator=dict(
        type='SinGANMSGeneratorPE',
        num_scales=num_scales,
        padding=1,
        pad_at_head=False,
        first_stage_in_channels=2,
        positional_encoding=dict(type='CSG')),
    discriminator=dict(num_scales=num_scales))

train_cfg = dict(first_fixed_noises_ch=2)

data = dict(
    train=dict(
        img_path='./data/singan/fish-crop.jpg',
        min_size=25,
        max_size=300,
    ))

dist_params = dict(backend='nccl')
total_iters = 22000
