# Spectral Normalization for Generative Adversarial Networks

## Introduction
<!-- [ALGORITHM] -->
```latex
@inproceedings{miyato2018spectral,
  title={Spectral Normalization for Generative Adversarial Networks},
  author={Miyato, Takeru and Kataoka, Toshiki and Koyama, Masanori and Yoshida, Yuichi},
  booktitle={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=B1QRgziT-},
}
```
<div align="center">
  <b> Results from our SNGAN-PROJ trained in CIFAR10 and ImageNet</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/28132635/125151484-14220b80-e179-11eb-81f7-9391ccaeb841.png" width="400"/> &nbsp;&nbsp;
  <img src="https://user-images.githubusercontent.com/28132635/127621152-7b7a0f2c-c743-485a-bf2e-2beca849a6e6.png" width="400"/>
</div>


|     Models     | Dataset  | Inplace ReLU | disc_step | Total Iters\* |                                                                                    Best IS (Iter)                                                                                    |                                                                                    Best FID (Iter)                                                                                    |                                                                             Config                                                                             |                                                                                    Log                                                                                     |
|:--------------:|:--------:|:------------:|:---------:|:-------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| sngan_proj_32  | CIFAR10  |     w/o      |     5     |    500000     |           [9.6919 (400000)](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_cifar10_32_lr-2e-4_b64x1_woReLUinplace_is-iter400000_20210709_163823-902ce1ae.pth)            |           [8.1158 (490000)](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_cifar10_32_lr-2e-4_b64x1_woReLUinplace_fid-iter490000_20210709_163329-ba0862a0.pth)            |        [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sngan_proj/sngan_proj_32_woReLUinplace_lr-2e-4_ndisc5_cifar10_b64x1.py)         |           [Log](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_cifar10_32_lr-2e-4_b64x1_woReLUinplace_20210624_065306_fid-ba0862a0_is-902ce1ae.json)           |
| sngan_proj_32  | CIFAR10  |      w       |     5     |    500000     |            [9.5564 (490000)](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_cifar10_32_lr-2e-4_b64x1_wReLUinplace_is-iter490000_20210709_202230-cd863c74.pth)            |            [8.3462 (490000)](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_cifar10_32_lr-2e-4-b64x1_wReLUinplace_fid-iter490000_20210709_203038-191b2648.pth)            |         [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sngan_proj/sngan_proj_32_wReLUinplace_lr-2e-4_ndisc5_cifar10_b64x1.py)         |           [Log](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_cifar10_32_lr-2e-4_b64x1_wReLUinplace_20210624_063454_is-cd863c74_fid-191b2648.json)            |
| sngan_proj_128 | ImageNet |     w/o      |     5     |    1000000    | [30.0651 (952000)](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_imagenet1k_128_Glr2e-4_Dlr5e-5_ndisc5_b128x2_woReLUinplace_is-iter952000_20210730_132027-9c884a21.pth) | [32.6193 (988000)](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_imagenet1k_128_Glr2e-4_Dlr5e-5_ndisc5_b128x2_woReLUinplace_fid-iter988000_20210730_131424-061bf803.pth) | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sngan_proj/sngan_proj_128_woReLUinplace_Glr-2e-4_Dlr-5e-5_ndisc5_imagenet1k_b128x2.py) | [Log](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_imagenet1k_128_Glr2e-4_Dlr5e-5_ndisc5_b128x2_woReLUinplace_20210730_131424_fid-061bf803_is-9c884a21.json) |
| sngan_proj_128 | ImageNet |      w       |     5     |    1000000    | [28.1799 (944000)](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_imagenet1k_128_Glr2e-4_Dlr5e-5_ndisc5_b128x2_wReLUinplace_is-iter944000_20210730_132714-ca0ccd07.pth)  | [33.4821 (988000)](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_imagenet1k_128_Glr2e-4_Dlr5e-5_ndisc5_b128x2_wReLUinplace_fid-iter988000_20210730_132401-9a682411.pth)  | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sngan_proj/sngan_proj_128_wReLUinplace_Glr-2e-4_Dlr-5e-5_ndisc5_imagenet1k_b128x2.py)  | [Log](https://download.openmmlab.com/mmgen/sngan_proj/sngan_proj_imagenet1k_128_Glr2e-4_Dlr5e-5_ndisc5_b128x2_wReLUinplace_20210730_132401_fid-9a682411_is-ca0ccd07.json)  |
| sngan_proj_128 | ImageNet |     w/o      |     2     |    2000000    |                                                                                     Coming soon                                                                                      |                                                                                     Comming soon                                                                                      |                                                                          Comming Soon                                                                          |                                                                                Comming Soon                                                                                |
| sngan_proj_128 | ImageNet |      w       |     2     |    2000000    |                                                                                     Coming soon                                                                                      |                                                                                     Comming soon                                                                                      |                                                                          Comming Soon                                                                          |                                                                                Comming Soon                                                                                |

'\*' Iteration counting rule in our implementation is different from others. If you want to align with other codebases, you can use the following conversion formula:
```
total_iters (biggan/pytorch studio gan) = our_total_iters / disc_step
```

We also provide converted pre-train models from [Pytorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN).
To be noted that, in Pytorch Studio GAN, **inplace ReLU** is used in generator and discriminator.

|     Models     | Dataset  | Inplace ReLU | disc_step | Total Iters | IS (Our Pipeline) | FID (Our Pipeline) | IS (StudioGAN) | FID (StudioGAN) |                                                           Download                                                           |                                Original Download link                                |
|:--------------:|:--------:|:------------:|:---------:|:-----------:|:-----------------:|:------------------:|:--------------:|:---------------:|:----------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------:|
| sngan_proj_32  | CIFAR10  |      w       |     5     |   100000    |       9.372       |      10.2011       |     8.677      |     13.248      |  [Download](https://download.openmmlab.com/mmgen/sngan_proj/sngan_cifar10_convert-studio-rgb_20210709_111346-2979202d.pth)   | [Download](https://drive.google.com/drive/folders/16s5Cr-V-NlfLyy_uyXEkoNxLBt-8wYSM) |
| sngan_proj_128 | ImageNet |      w       |     2     |   1000000   |      30.218       |      29.8199       |     32.247     |     26.792      | [Download](https://download.openmmlab.com/mmgen/sngan_proj/sngan_imagenet1k_convert-studio-rgb_20210709_111406-877b1130.pth) | [Download](https://drive.google.com/drive/folders/1Ek2wAMlxpajL_M8aub4DKQ9B313K8XhS) |


* `Our Pipeline` denote results evaluated with our pipeline.
* `StudioGAN` denote results released by Pytorch-StudioGAN.

For IS metric, our implementation is different from PyTorch-Studio GAN in the following aspects:
1. We use [Tero's Inception](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) for feature extraction.
2. We use bicubic interpolation with PIL backend to resize image before feed them to Inception.

For FID evaluation, differences between PyTorch Studio GAN and ours are mainly on the selection of real samples. In MMGen, we follow the pipeline of BigGAN, where the whole training set is adopted to extract inception statistics. Besides, we also use [Tero's Inception](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) for feature extraction.

You can download the preprocessed inception state by the following url: [CIFAR10](https://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/cifar10.pkl) and [ImageNet1k](https://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/imagenet.pkl).

You can use following commands to extract those inception states by yourself.
```
# For CIFAR10
python tools/utils/inception_stat.py --data-cfg configs/_base_/datasets/cifar10_inception_stat.py --pklname cifar10.pkl --no-shuffle --inception-style stylegan --num-samples -1 --subset train

# For ImageNet1k
python tools/utils/inception_stat.py --data-cfg configs/_base_/datasets/imagenet_128x128_inception_stat.py --pklname imagenet.pkl --no-shuffle --inception-style stylegan --num-samples -1 --subset train
```
