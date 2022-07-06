_base_ = [
    '../_base_/models/sagan/sagan_128x128.py',
    '../_base_/datasets/imagenet_128.py', '../_base_/default_runtime.py'
]

# MODEL
init_cfg = dict(type='studio')
model = dict(
    num_classes=1000,
    generator=dict(num_classes=1000, init_cfg=init_cfg),
    discriminator=dict(num_classes=1000, init_cfg=init_cfg))

# TRAIN
train_cfg = dict(max_iters=1000000)
train_dataloader = dict(batch_size=64)  # train on 4 gpus

optim_wrapper = dict(
    generator=dict(optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999))))

# METRICS
inception_pkl = './work_dirs/inception_pkl/imagenet-full.pkl'
metrics = [
    dict(
        type='InceptionScore',
        prefix='IS-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema'),
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        inception_pkl=inception_pkl,
        sample_model='ema')
]
default_hooks = dict(checkpoint=dict(save_best='FID-Full-50k/fid'))

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
