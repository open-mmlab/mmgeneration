"""Config for the `config-f` setting in StyleGAN2."""

_base_ = ['./stylegan2_c2_ffhq_256_b4x8_800k.py']

model = dict(
    generator=dict(out_size=256),
    discriminator=dict(in_size=256, convert_input_fp32=False),
    # disc_auxiliary_loss=dict(use_apex_amp=True),
    # gen_auxiliary_loss=dict(use_apex_amp=True),
)

dataset_type = 'QuickTestImageDataset'
data = dict(
    samples_per_gpu=2,
    train=dict(type=dataset_type, size=(256, 256)),
    val=dict(type=dataset_type, size=(256, 256)))

log_config = dict(interval=1)

total_iters = 800002

apex_amp = dict(
    mode='gan', init_args=dict(opt_level='O1', num_losses=2, loss_scale=512.))

evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=dict(
        type='FID', num_images=50000, inception_pkl=None, bgr2rgb=True),
    sample_kwargs=dict(sample_model='ema'))
