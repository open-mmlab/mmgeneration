_base_ = ['./singan_fish.py']

# MODEL
num_scales = 8  # start from zero
generator_steps = 3
discriminator_steps = 3
iters_per_scale = 2000

# NOTE: add by user, e.g.:
test_pkl_data = './work_dirs/singan_pkl/singan_balloons_20210406_191047-8fcd94cf.pkl'  # noqa
# test_pkl_data = None
model = dict(
    num_scales=num_scales,
    generator=dict(num_scales=num_scales),
    discriminator=dict(num_scales=num_scales),
    test_pkl_data=test_pkl_data)

# DATA
data_root = './data/singan/balloons.png'
train_dataloader = dict(dataset=dict(data_root=data_root))

# TRAINING
total_iters = (num_scales + 1) * iters_per_scale * discriminator_steps
train_cfg = dict(max_iters=total_iters)
