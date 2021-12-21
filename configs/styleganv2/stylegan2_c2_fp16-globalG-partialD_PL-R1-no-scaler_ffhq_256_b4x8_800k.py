"""Config for the `config-f` setting in StyleGAN2."""

_base_ = ['./stylegan2_c2_ffhq_256_b4x8_800k.py']

model = dict(
    generator=dict(out_size=256, fp16_enabled=True),
    discriminator=dict(in_size=256, fp16_enabled=False, num_fp16_scales=4),
)

total_iters = 800000

# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    fp16_loss_scaler=dict(init_scale=512),
    is_dynamic_ddp=  # noqa
    False,  # Note that this flag should be False to use DDP wrapper.
)
