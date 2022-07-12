_base_ = [
    '../_base_/models/improved_ddpm/ddpm_64x64.py',
    '../_base_/datasets/imagenet_noaug_64.py', '../_base_/default_runtime.py'
]

# TRAIN
train_cfg = dict(max_iters=1500000)  # 1500k
train_dataloader = dict(batch_size=16)  # 16 * 8gpus = 128

optim_wrapper = dict(
    denoising=dict(optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0)))

# VISUALIZATION
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        sample_kwargs_list=dict(type='DDPMDenoising'))
]

# METRICS
inception_pkl = './work_dirs/inception_pkl/imagenet-full.pkl'
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema')
]

# NOTE: do not evaluation in the training process because evaluation take too
# much time.
val_cfg = val_evaluator = val_dataloader = None
test_evaluator = dict(metrics=metrics)
