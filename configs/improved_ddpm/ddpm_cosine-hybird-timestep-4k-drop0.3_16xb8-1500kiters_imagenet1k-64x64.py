_base_ = ['./ddpm_cosine-hybird-timestep-4k_16xb8-1500kiters_imagenet1k-64x64.py']

# MODEL
# set dropout prob as 0.3
model = dict(denoising=dict(dropout=0.3))
