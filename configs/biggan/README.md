# LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS

## Introduction
<!-- [ALGORITHM] -->
```latex
@inproceedings{
brock2018large,
title={Large Scale {GAN} Training for High Fidelity Natural Image Synthesis},
author={Andrew Brock and Jeff Donahue and Karen Simonyan},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=B1xsqj09Fm},
}
```

The BigGAN/BigGAN-Deep is a conditional generation model that can generate both high-resolution and high-quality images by scaling up the batch size and number of model parameters. 

We have conducted training BigGAN with CIFAR10(3x32x32) and ImageNet1k(3x128x128) dataset, and the sampling results are showed below
<div align="center">
  <b> Results from our BigGAN trained in CIFAR10</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126476913-3ce8e2c8-f189-4caa-90ed-b44e279cb669.png" width="800"/>
</div>

Evaluation of our trained BIgGAN.
|    Models    | Dataset |   FID (Iter) | IS (Iter) | Config |  Download  |
|:------------:|:-------:|:--------------:|:---------------:|:------:|:----------:|
| BigGAN 32x32 | CIFAR10 |     9.78(390000)           |       8.70(390000)          | [config](https://github.com/open-mmlab/mmgeneration/blob/master/configs/biggan/biggan_cifar10_32x32_b25x2_500k.py) | [model](https://download.openmmlab.com/mmgen/biggan/biggan_cifar10_32x32_b25x2_500k_20210728_110906-08b61a44.pth)\|[log](https://download.openmmlab.com/mmgen/biggan/biggan_cifar10_32_b25x2_500k_20210706_171051.log.json) |
| BigGAN 128x128 | ImageNet1k |       12.32(1150000)         |         72.7(1150000)        | [config](https://github.com/open-mmlab/mmgeneration/blob/master/) | [model](https://download.openmmlab.com/mmgen/biggan/biggan_imagenet1k_128x128_b32x8_1150k_20210730_124753-b14026b7.pth)\|[log](https://download.openmmlab.com/mmgen/biggan/biggan_imagenet1k_128x128_b32x8_1500k_20210726_224316.log.json) |

Note: This is an earlier version(1150k iter) of BigGAN trained on ImageNet1k, the model with best performance is still on the way.

## converted weights
Since we havn't finished training our models, we provide you with several pre-trained weights which has already be evaluated. Here, we refer to [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) and [pytorch-pretrained-BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN).

Evaluation results and download links are provided below.

|        Models       |   Dataset  | FID | IS | Config | Download | Original Download link |
|:-------------------:|:----------:|:--:|:---:|:------:|:--------:|:----------------------:|
|    BigGAN 128x128   | ImageNet1k | 10.1414   |  96.728   | [config](https://github.com/open-mmlab/mmgeneration/blob/master/) |   [model](	
https://download.openmmlab.com/mmgen/biggan/biggan_imagenet1k_128x128_cvt_BigGAN-PyTorch_rgb_20210730_125223-3e353fef.pth)  |          [link](https://drive.google.com/open?id=1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW)          |
| BigGAN-Deep 128x128 | ImageNet1k |  5.9471  |  107.161   | [config](https://github.com/open-mmlab/mmgeneration/blob/master/) |   [model](https://download.openmmlab.com/mmgen/biggan/biggan-deep_imagenet1k_128x128_cvt_hugging-face_rgb_20210728_111659-099e96f9.pth)  |          [link](https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-pytorch_model.bin)          |
| BigGAN-Deep 256x256 | ImageNet1k | 11.3151   | 135.107    | [config](https://github.com/open-mmlab/mmgeneration/blob/master/) |   [model](https://download.openmmlab.com/mmgen/biggan/biggan-deep_imagenet1k_256x256_cvt_hugging-face_rgb_20210728_111735-28651569.pth)  |          [link](https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin)          |
| BigGAN-Deep 512x512 | ImageNet1k | 16.8728   | 124.368    | [config]() |   [model](https://download.openmmlab.com/mmgen/biggan/biggan-deep_imagenet1k_512x512_cvt_hugging-face_rgb_20210728_112346-a42585f2.pth)  |          [link](https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-pytorch_model.bin)          |

Sampling results are showed below.
<div align="center">
  <b> Results from our BigGAN-Deep trained in ImageNet 128x128 with truncation factor 0.4</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126481730-8da7180b-7b1b-42f0-9bec-78d879b6265b.png" width="800"/>
</div>

<div align="center">
  <b> Results from our BigGAN-Deep with Pre-trained weights in ImageNet 256x256 with truncation factor 0.4</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126486040-64effa29-959e-4e43-bcae-15925a2e0599.png" width="800"/>
</div>

<div align="center">
  <b> Results from our BigGAN-Deep with Pre-trained weights in ImageNet 512x512 truncation factor 0.4</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126487428-50101454-59cb-469d-a1f1-36ffb6291582.png" width="800"/>
</div>
Above sampling with truncation trick can be performed by command below

```python
python demo/conditional_demo.py CONFIG_PATH CKPT_PATH --sample-cfg truncation=0.4 # set truncation value as you want
```

## Interpolation

We will also provide script for image interpolation of conditional models soon.
<div align="center">
  <b> Image interpolating Results of our BigGAN-Deep</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126580403-2baa987b-ff55-4fb5-a53a-b08e8a6a72a2.png" width="800"/>
</div>