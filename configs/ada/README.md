# ADA

> [Training Generative Adversarial Networks with Limited Data](https://arxiv.org/pdf/2006.06676.pdf)

<!-- [ALGORITHM] -->

## Abstract

Training generative adversarial networks (GAN) using too little data typically leads to discriminator overfitting, causing training to diverge. We propose an adaptive discriminator augmentation mechanism that significantly stabilizes training in limited data regimes. The approach does not require changes to loss functions or network architectures, and is applicable both when training from scratch and when fine-tuning an existing GAN on another dataset. We demonstrate, on several datasets, that good results are now possible using only a few thousand training images, often matching StyleGAN2 results with an order of magnitude fewer images. We expect this to open up new application domains for GANs. We also find that the widely used CIFAR-10 is, in fact, a limited data benchmark, and improve the record FID from 5.59 to 2.42.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/22982797/165902671-ee835ca5-3957-451e-8e7d-e3741d90e0b1.png"/>
</div>

## Results and Models

<div align="center">
  <b> Results (compressed) from StyleGAN3-ada trained by MMGeneration</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/165905181-66d6b4e7-6d40-48db-8281-50ebd2705f64.png" width="800"/>
</div>

|      Model      |      Dataset      |  Iter  | FID50k |                        Config                        |                        Log                        |                        Download                         |
| :-------------: | :---------------: | :----: | :----: | :--------------------------------------------------: | :-----------------------------------------------: | :-----------------------------------------------------: |
| stylegan3-t-ada | metface 1024x1024 | 130000 | 15.09  | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv3/stylegan3_t_ada_fp16_gamma6.6_metfaces_1024_b4x8.py) | [log](https://download.openmmlab.com/mmgen/stylegan3/stylegan3_t_ada_fp16_gamma6.6_metfaces_1024_b4x8_20220328_142211.log.json) | [model](https://download.openmmlab.com/mmgen/stylegan3/stylegan3_t_ada_fp16_gamma6.6_metfaces_1024_b4x8_best_fid_iter_130000_20220401_115101-f2ef498e.pth) |

## Usage

Currently we only implement ada for StyleGANv2/v3. To use this training trick. You should use `ADAStyleGAN2Discriminator` as your discriminator.

An example:

```python
model = dict(
    xxx,
    discriminator=dict(
        type='ADAStyleGAN2Discriminator',
        in_size=1024,
        data_aug=dict(type='ADAAug', aug_pipeline=aug_kwargs, ada_kimg=100)),
    xxx
)
```

Here, you can adjust `ada_kimg` to change the magnitude of augmentation(The smaller the value, the greater the magnitude).

`aug_kwargs` is usually set as follows:

```python
aug_kwargs = {
    'xflip': 1,
    'rotate90': 1,
    'xint': 1,
    'scale': 1,
    'rotate': 1,
    'aniso': 1,
    'xfrac': 1,
    'brightness': 1,
    'contrast': 1,
    'lumaflip': 1,
    'hue': 1,
    'saturation': 1
}
```

Here, the number is Probability multiplier for each operation. For details, you can refer to [augment](https://github.com/open-mmlab/mmgeneration/tree/master/mmgen/models/architectures/stylegan/ada/augment.py).

## Citation

```latex
@inproceedings{Karras2020ada,
  title     = {Training Generative Adversarial Networks with Limited Data},
  author    = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
  booktitle = {Proc. NeurIPS},
  year      = {2020}
}
```
