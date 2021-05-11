"""Config for the `config-f` setting in StyleGAN2."""

_base_ = ['./stylegan2_c2_ffhq_256_b4x8_800k.py']

model = dict(
    generator=dict(out_size=256, fp16_enabled=True),
    discriminator=dict(in_size=256, fp16_enabled=True),
    disc_auxiliary_loss=dict(data_info=dict(loss_scaler='loss_scaler')),
    # gen_auxiliary_loss=dict(data_info=dict(loss_scaler='loss_scaler')),
)

dataset_type = 'QuickTestImageDataset'
data = dict(
    samples_per_gpu=2,
    train=dict(type=dataset_type, size=(256, 256)),
    val=dict(type=dataset_type, size=(256, 256)))

log_config = dict(interval=1)

total_iters = 800002

runner = dict(fp16_loss_scaler=dict(init_scale=512))

evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=dict(
        type='FID', num_images=50000, inception_pkl=None, bgr2rgb=True),
    sample_kwargs=dict(sample_model='ema'))
