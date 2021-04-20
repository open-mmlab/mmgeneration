_base_ = ['./wgangp_GN_celeba-cropped_128_b64x1_160kiter.py']

model = dict(disc_auxiliary_loss=[
    dict(
        type='GradientPenaltyLoss',
        loss_weight=50,
        norm_mode='HWC',
        data_info=dict(
            discriminator='disc', real_data='real_imgs',
            fake_data='fake_imgs'))
])

data = dict(
    samples_per_gpu=64, train=dict(imgs_root='./data/lsun/bedroom_train'))
