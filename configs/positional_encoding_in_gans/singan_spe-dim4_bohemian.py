_base_ = ['../singan/singan_bohemian.py']

embedding_dim = 4
num_scales = 10  # start from zero
model = dict(
    type='PESinGAN',
    generator=dict(
        type='SinGANMSGeneratorPE',
        num_scales=num_scales,
        padding=1,
        pad_at_head=False,
        first_stage_in_channels=embedding_dim * 2,
        positional_encoding=dict(
            type='SPE',
            embedding_dim=embedding_dim,
            padding_idx=0,
            init_size=512,
            div_half_dim=False,
            center_shift=200)),
    discriminator=dict(num_scales=num_scales))

train_cfg = dict(first_fixed_noises_ch=embedding_dim * 2)

data = dict(
    train=dict(
        img_path='./data/singan/bohemian.png',
        min_size=25,
        max_size=500,
    ))

dist_params = dict(backend='nccl')
total_iters = 22000
