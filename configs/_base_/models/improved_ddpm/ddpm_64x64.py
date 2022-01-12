model = dict(
    type='BasicGaussianDiffusion',
    num_timesteps=4000,
    betas_cfg=dict(type='cosine'),
    denoising=dict(
        type='DenoisingUnet',
        image_size=64,
        in_channels=3,
        base_channels=128,
        resblocks_per_downsample=3,
        attention_res=[16, 8],
        use_scale_shift_norm=True,
        dropout=0,
        num_heads=4,
        use_rescale_timesteps=True,
        output_cfg=dict(mean='eps', var='learned_range')),
    timestep_sampler=dict(type='UniformTimeStepSampler'),
    ddpm_loss=[
        dict(
            type='DDPMVLBLoss',
            rescale_mode='constant',
            rescale_cfg=dict(scale=4000 / 1000),
            data_info=dict(
                mean_pred='mean_pred',
                mean_target='mean_posterior',
                logvar_pred='logvar_pred',
                logvar_target='logvar_posterior'),
            log_cfgs=[
                dict(
                    type='quartile',
                    prefix_name='loss_vlb',
                    total_timesteps=4000),
                dict(type='name')
            ]),
        dict(
            type='DDPMMSELoss',
            log_cfgs=dict(
                type='quartile', prefix_name='loss_mse', total_timesteps=4000),
        )
    ],
)

train_cfg = dict(use_ema=True, real_img_key='img')
test_cfg = None
optimizer = dict(denoising=dict(type='AdamW', lr=1e-4, weight_decay=0))
