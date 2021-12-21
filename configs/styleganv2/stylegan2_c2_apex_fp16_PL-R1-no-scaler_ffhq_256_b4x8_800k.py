"""Config for the `config-f` setting in StyleGAN2."""

_base_ = ['./stylegan2_c2_ffhq_256_b4x8_800k.py']

model = dict(
    disc_auxiliary_loss=dict(use_apex_amp=False),
    gen_auxiliary_loss=dict(use_apex_amp=False),
)

total_iters = 800002

apex_amp = dict(mode='gan', init_args=dict(opt_level='O1', num_losses=2))
resume_from = None
