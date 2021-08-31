# MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"  # noqa
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
# TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

model = dict(
    type='BasicGaussianDiffusion',
    num_timesteps=4000,
    # betas_cfg=dict(type='linear', beta_0=1e-4, beta_T=2e-2),
    betas_cfg=dict(type='cosine'),
    denoising=dict(
        type='DenoisingUnet',
        image_size=32,
        in_channels=3,
        base_channels=128,
        resblocks_per_downsample=3,
        attention_res=[16, 8],
        use_scale_shift_norm=True,
        dropout=0.3,
        num_heads=4,
        rescale_timesteps=True,
        output_cfg=dict(mean='eps', var='learned_range')))
