_base_ = [
    '../_base_/models/singan/singan.py', '../_base_/datasets/singan.py',
    '../_base_/default_runtime.py'
]

num_scales = 10  # start from zero
model = dict(
    generator=dict(num_scales=num_scales),
    discriminator=dict(num_scales=num_scales))

train_cfg = dict(
    noise_weight_init=0.1,
    iters_per_scale=2000,
)

# test_cfg = dict(
#     _delete_ = True
#     pkl_data = 'path to pkl data'
# )

data = dict(
    train=dict(
        img_path='./data/singan/bohemian.png', min_size=25, max_size=500))

optimizer = None
lr_config = None
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=3)

custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='visual',
        interval=500,
        bgr2rgb=True,
        res_name_list=['fake_imgs', 'recon_imgs', 'real_imgs']),
    dict(
        type='PickleDataHook',
        output_dir='pickle',
        interval=-1,
        after_run=True,
        data_name_list=['noise_weights', 'fixed_noises', 'curr_stage'])
]

total_iters = 22000
