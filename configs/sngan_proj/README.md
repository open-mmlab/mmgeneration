# Spectral Normalization for Generative Adversarial Networks

## Introduction
<!-- [ALGORITHM] -->
```latex
@inproceedings{miyato2018spectral,
  title={Spectral Normalization for Generative Adversarial Networks},
  author={Miyato, Takeru and Kataoka, Toshiki and Koyama, Masanori and Yoshida, Yuichi},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```
<div align="center">
  <b> Results from our SNGAN-PROJ trained in CIFAR10</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/28132635/125151484-14220b80-e179-11eb-81f7-9391ccaeb841.png" width="400"/>
</div>


|              Models              | Details |                         Best IS (Iter)                              |                         Best FID (Iter)                              |                                                            Config                                                             | Log |
|:--------------------------------:|:-------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------:| :-:|
| sngan_proj_32 (w/o inplace ReLU) | CIFAR10 | [9.6919 (400000)](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_cifar10_32_lr-2e-4_b64x1_woReLUinplace_is-iter400000_20210709_163823-902ce1ae.pth) | [8.1158 (490000)](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_cifar10_32_lr-2e-4_b64x1_woReLUinplace_fid-iter490000_20210709_163329-ba0862a0.pth) | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sngan_proj_cifar10_32_lr-2e-4_b64x1_woReLUinplace.py) | [Log](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_cifar10_32_lr-2e-4_b64x1_woReLUinplace_20210624_065306_fid-ba0862a0_is-902ce1ae.json)
|  sngan_proj_32 (w inplace ReLU)  | CIFAR10 | [9.5564 (490000)](https://download.openmmlab.com/mmgen/sngan_projsngan_proj_cifar10_32_lr-2e-4_b64x1_wReLUinplace_is-iter490000_20210709_202230-cd863c74.pth/) | [8.3462 (490000)](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_cifar10_32_lr-2e-4-b64x1_wReLUinplace_fid-iter490000_20210709_203038-191b2648.pth) | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sngan_proj_cifar10_32_lr-2e-4_b64x1_wReLUinplace.py)  | [Log](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_cifar10_32_lr-2e-4_b64x1_wReLUinplace_20210624_063454_is-cd863c74_fid-191b2648.json) |


We also provide converted pretrain model from [Pytorch Studio GAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN).
To be noted that, in Pytorch Studio GAN, inplace ReLU is used in generator and discriminator.

|     Models     |  Details   |  IS\*  |  FID\*  | IS\*\* | FID\*\* |                                                                                                                     Download |                                    Original Download link                                     |
|:--------------:|:----------:|:------:|:-------:|:------:|:-------:|-----------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------:|
| sngan_proj_32  |  CIFAR10   | 9.372  |  9.205  | 8.677  | 13.248  |    [Download](https://download.openmmlab.com/mmgen/sngan_proj/sngan_cifar10_convert-studio-rgb_20210709_111346-2979202d.pth) | [Download](https://drive.google.com/drive/folders/16s5Cr-V-NlfLyy_uyXEkoNxLBt-8wYSM) |
| sngan_proj_128 | ImageNet1k | 30.218 | 29.8199 | 32.247 | 26.792  | [Download](https://download.openmmlab.com/mmgen/sngan_proj/sngan_imagenet1k_convert-studio-rgb_20210709_111406-877b1130.pth) | [Download](https://drive.google.com/drive/folders/1Ek2wAMlxpajL_M8aub4DKQ9B313K8XhS) |


'\*' denote results evaluated with our pipeline.
'\*\*' denote results released by Pytorch Studio GAN.

For IS evaluation,
1. We use [Tero's Inception](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) for feature extraction.
2. We use bicubic interpolation with PIL backend to resize image before feed them to Inception.

For FID evaluation, difference between two repositories mainly on the selection of images for ground truth activation (inception state) calculation. In ours, we follow the pipeline of BigGAN for more convincing results. Detailly, we use the entire training set (50k and 1281167 images for CIFAR10 and ImageNet1k) without shuffle for GT inception state extracting. We also use [Tero's Inception](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) for feature extraction.
You can use following scripts to reproduce those inception states.

```
# For CIFAR10
python tools/utils/inception_stat.py --data-cfg configs/_base_/datasets/cifar10_rgb.py --pklname cifar10.pkl --no-shuffle --inception-style stylegan --num-samples -1 --subset train

# For ImageNet1k
python tools/utils/inception_stat.py --data-cfg configs/_base_/datasets/imagenet_rgb.py --pklname imagenet.pkl --no-shuffle --inception-style stylegan --num-samples -1 --subset train
```
