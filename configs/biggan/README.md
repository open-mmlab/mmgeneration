# Large Scale GAN Training for High Fidelity Natural Image Synthesis
## Introduction
<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://openreview.net/forum?id=B1xsqj09Fm">BigGAN (ICLR'2019)</a></summary>

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

</details>

The `BigGAN/BigGAN-Deep` is a conditional generation model that can generate both high-resolution and high-quality images by scaling up the batch size and the number of model parameters.

We have finished training `BigGAN` in `Cifar10` (32x32) and are aligning training performance in `ImageNet1k` (128x128). Some sampled results are shown below for your reference.
<div align="center">
  <b> Results from our BigGAN trained in CIFAR10</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126476913-3ce8e2c8-f189-4caa-90ed-b44e279cb669.png" width="800"/>
</div>

<div align="center">
  <b> Results from our BigGAN trained in ImageNet</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/127615534-6278ce1b-5cff-4189-83c6-9ecc8de08dfc.png" width="800"/>
</div>

Evaluation of our trained BIgGAN.
|    Models    | Dataset | Total Iters|Iter|   FID  | IS  | Config |  Download  |
|:------------:|:-------:|:--------------:|:---------------:|:------:|:----------:|----------:|----------:|
| BigGAN 32x32 | CIFAR10 | 500000|390000|    9.78          |       8.70          | [config](https://github.com/open-mmlab/mmgeneration/blob/master/configs/biggan/biggan_cifar10_32x32_b25x2_500k.py) | [model](https://download.openmmlab.com/mmgen/biggan/biggan_cifar10_32x32_b25x2_500k_20210728_110906-08b61a44.pth)\|[log](https://download.openmmlab.com/mmgen/biggan/biggan_cifar10_32_b25x2_500k_20210706_171051.log.json) |
| BigGAN 128x128 Best FID | ImageNet1k |  1500000|1232000|    8.69          |      101.15           | [config](https://github.com/open-mmlab/mmgeneration/blob/master/configs/biggan/biggan_ajbrock-sn_imagenet1k_128x128_b32x8_1500k.py) | [model](https://download.openmmlab.com/mmgen/biggan/biggan_imagenet1k_128x128_b32x8_best_fid_iter_1232000_20211111_122548-5315b13d.pth?versionId=CAEQHhiBgIDi1ti16BciIDIzMDlmNzE1MzU1NTQ1YWFiMmJlMDVmOGM1MDM3ZmQ5)\|[log](https://download.openmmlab.com/mmgen/biggan/biggan_imagenet1k_128x128_b32x8_1500k_20211111_122548-5315b13d.log.json?versionId=CAEQHhiBgMDsn9226BciIGFmNTNkOTI2NGE0NDQxNzdhNjJmNTY5MWQzZDJjM2M0) |
| BigGAN 128x128 Best IS | ImageNet1k |  1500000|1328000|      13.51        |      129.07           | [config](https://github.com/open-mmlab/mmgeneration/blob/master/configs/biggan/biggan_ajbrock-sn_imagenet1k_128x128_b32x8_1500k.py) | [model](https://download.openmmlab.com/mmgen/biggan/biggan_imagenet1k_128x128_b32x8_best_is_iter_1328000_20211111_122911-28c688bc.pth?versionId=CAEQHhiBgMD209i16BciIDBiOTU0N2Y3YTdjMjQ5Njk5ZTZlMTcyYTU3NGNmNDM3)\|[log](https://download.openmmlab.com/mmgen/biggan/biggan_imagenet1k_128x128_b32x8_1500k_20211111_122548-5315b13d.log.json?versionId=CAEQHhiBgMDsn9226BciIGFmNTNkOTI2NGE0NDQxNzdhNjJmNTY5MWQzZDJjM2M0) |

Note: `BigGAN-Deep` trained on `ImageNet1k` will come later.

## Converted weights
Since we haven't finished training our models, we provide you with several pre-trained weights which have been evaluated. Here, we refer to [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) and [pytorch-pretrained-BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN).

Evaluation results and download links are provided below.

|        Models       |   Dataset  | FID | IS | Config | Download | Original Download link |
|:-------------------:|:----------:|:--:|:---:|:------:|:--------:|:----------------------:|
|    BigGAN 128x128   | ImageNet1k | 10.1414   |  96.728   | [config](https://github.com/open-mmlab/mmgeneration/blob/master/configs/_base_/models/biggan/biggan_128x128_cvt_BigGAN-PyTorch_rgb.py) |   [model](https://download.openmmlab.com/mmgen/biggan/biggan_imagenet1k_128x128_cvt_BigGAN-PyTorch_rgb_20210730_125223-3e353fef.pth)  |          [link](https://drive.google.com/open?id=1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW)          |
| BigGAN-Deep 128x128 | ImageNet1k |  5.9471  |  107.161   | [config](https://github.com/open-mmlab/mmgeneration/blob/master/configs/_base_/models/biggan/biggan-deep_128x128_cvt_hugging-face_rgb.py) |   [model](https://download.openmmlab.com/mmgen/biggan/biggan-deep_imagenet1k_128x128_cvt_hugging-face_rgb_20210728_111659-099e96f9.pth)  |          [link](https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-pytorch_model.bin)          |
| BigGAN-Deep 256x256 | ImageNet1k | 11.3151   | 135.107    | [config](https://github.com/open-mmlab/mmgeneration/blob/master/configs/_base_/models/biggan/biggan-deep_256x256_cvt_hugging-face_rgb.py) |   [model](https://download.openmmlab.com/mmgen/biggan/biggan-deep_imagenet1k_256x256_cvt_hugging-face_rgb_20210728_111735-28651569.pth)  |          [link](https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin)          |
| BigGAN-Deep 512x512 | ImageNet1k | 16.8728   | 124.368    | [config](https://github.com/open-mmlab/mmgeneration/blob/master/configs/_base_/models/biggan/biggan-deep_512x512_cvt_hugging-face_rgb.py) |   [model](https://download.openmmlab.com/mmgen/biggan/biggan-deep_imagenet1k_512x512_cvt_hugging-face_rgb_20210728_112346-a42585f2.pth)  |          [link](https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-pytorch_model.bin)          |

Sampling results are shown below.
<div align="center">
  <b> Results from our BigGAN-Deep with Pre-trained weights in ImageNet 128x128 with truncation factor 0.4</b>
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
Sampling with truncation trick above can be performed by command below.

```bash
python demo/conditional_demo.py CONFIG_PATH CKPT_PATH --sample-cfg truncation=0.4 # set truncation value as you want
```
For converted weights, we provide model configs under `configs/_base_/models` listed as follows:
```bash
# biggan_128x128_cvt_BigGAN-PyTorch_rgb.py
# biggan-deep_128x128_cvt_hugging-face_rgb.py
# biggan-deep_256x256_cvt_hugging-face_rgb.py
# biggan-deep_512x512_cvt_hugging-face_rgb.py
```
## Interpolation

To perform image Interpolation on BigGAN(or other conditional models), run
```bash
python apps/conditional_interpolate.py CONFIG_PATH  CKPT_PATH  --samples-path SAMPLES_PATH
```
<div align="center">
  <b> Image interpolating Results of our BigGAN-Deep</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126580403-2baa987b-ff55-4fb5-a53a-b08e8a6a72a2.png" width="800"/>
</div>

To perform image Interpolation on BigGAN with fixed noise, run
```bash
python apps/conditional_interpolate.py CONFIG_PATH  CKPT_PATH  --samples-path SAMPLES_PATH --fix-z
```
<div align="center">
  <b> Image interpolating Results of our BigGAN-Deep with fixed noise</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/128123804-6df1dfca-1057-4b96-8428-787a86f81ef1.png" width="800"/>
</div>
To perform image Interpolation on BigGAN with fixed label, run

```bash
python apps/conditional_interpolate.py CONFIG_PATH  CKPT_PATH  --samples-path SAMPLES_PATH --fix-y
```

<div align="center">
  <b> Image interpolating Results of our BigGAN-Deep with fixed label</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/128124596-421396f1-3f23-4098-b629-b00d29d710a9.png" width="800"/>
</div>
