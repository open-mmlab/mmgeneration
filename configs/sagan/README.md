# Self-attention generative adversarial networks

## Introduction
<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://proceedings.mlr.press/v97/zhang19d.html">SAGAN (ICML'2019)</a></summary>
```latex
@inproceedings{zhang2019self,
  title={Self-attention generative adversarial networks},
  author={Zhang, Han and Goodfellow, Ian and Metaxas, Dimitris and Odena, Augustus},
  booktitle={International conference on machine learning},
  pages={7354--7363},
  year={2019},
  organization={PMLR},
  url={https://proceedings.mlr.press/v97/zhang19d.html},
}
```
<div align="center">
  <b> Results from our SAGAN trained in CIFAR10</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/28132635/127619657-67f2e62d-52e4-43d2-931f-6d0e6e019813.png" width="400"/>
</div>

</details>


## Results and models

|                 Models                 | Dataset  | Inplace ReLU | dist_step | Total Batchsize (BZ_PER_GPU \* NGPU) | Total Iters* |  Iter  |   IS    |   FID   |                                                                             Config                                                                              |                                                                            Download                                                                            |                                                                               Log                                                                               |
| :------------------------------------: | :------: | :----------: | :-------: | :----------------------------------: | :----------: | :----: | :-----: | :-----: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   SAGAN-32x32-woInplaceReLU Best IS    | CIFAR10  |     w/o      |     5     |                 64x1                 |    500000    | 400000 | 9.3217  | 10.5030 |              [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sagan/sagan_32_woReLUinplace_lr-2e-4_ndisc5_cifar10_b64x1.py)              |        [model](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_woReUinplace_is-iter400000_20210730_125743-4008a9ca.pth)         |       [Log](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_woReUinplace_20210730_125449_fid-d50568a4_is-04008a9ca.json)        |
|   SAGAN-32x32-woInplaceReLU Best FID   | CIFAR10  |     w/o      |     5     |                 64x1                 |    500000    | 480000 | 9.3174  | 9.4252  |              [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sagan/sagan_32_woReLUinplace_lr-2e-4_ndisc5_cifar10_b64x1.py)              |        [model](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_woReUinplace_fid-iter480000_20210730_125449-d50568a4.pth)        |       [Log](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_woReUinplace_20210730_125449_fid-d50568a4_is-04008a9ca.json)        |
|    SAGAN-32x32-wInplaceReLU Best IS    | CIFAR10  |      w       |     5     |                 64x1                 |    500000    | 380000 | 9.2286  | 11.7760 |              [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sagan/sagan_32_wReLUinplace_lr-2e-4_ndisc5_cifar10_b64x1.py)               |        [model](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_wReLUinplace_is-iter380000_20210730_124937-c77b4d25.pth)         |        [Log](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_wReLUinplace_20210730_125155_fid-cbefb354_is-c77b4d25.json)        |
|   SAGAN-32x32-wInplaceReLU Best FID    | CIFAR10  |      w       |     5     |                 64x1                 |    500000    | 460000 | 9.2061  | 10.7781 |              [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sagan/sagan_32_wReLUinplace_lr-2e-4_ndisc5_cifar10_b64x1.py)               |        [model](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_wReLUinplace_fid-iter460000_20210730_125155-cbefb354.pth)        |        [Log](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_wReLUinplace_20210730_125155_fid-cbefb354_is-c77b4d25.json)        |
|  SAGAN-128x128-woInplaceReLU Best IS   | ImageNet |     w/o      |     1     |                 64x4                 |   1000000    | 980000 | 31.5938 | 36.7712 |       [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sagan/sagan_128_woReLUinplace_Glr-1e-4_Dlr-4e-4_ndisc1_imagenet1k_b64x4.py)       | [model](https://download.openmmlab.com/mmgen/sagan/sagan_imagenet1k_128_Glr1e-4_Dlr4e-4_ndisc1_b32x4_woReLUinplace_is-iter980000_20210730_163140-cfbebfc6.pth)  | [Log](https://download.openmmlab.com/mmgen/sagan/sagan_imagenet1k_128_Glr1e-4_Dlr4e-4_ndisc1_b32x4_woReLUinplace_20210730_163431_fid-d7916963_is-cfbebfc6.json) |
|  SAGAN-128x128-woInplaceReLU Best FID  | ImageNet |     w/o      |     1     |                 64x4                 |   1000000    | 950000 | 28.4936 | 34.7838 |       [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sagan/sagan_128_woReLUinplace_Glr-1e-4_Dlr-4e-4_ndisc1_imagenet1k_b64x4.py)       | [model](https://download.openmmlab.com/mmgen/sagan/sagan_imagenet1k_128_Glr1e-4_Dlr4e-4_ndisc1_b32x4_woReLUinplace_fid-iter950000_20210730_163431-d7916963.pth) | [Log](https://download.openmmlab.com/mmgen/sagan/sagan_imagenet1k_128_Glr1e-4_Dlr4e-4_ndisc1_b32x4_woReLUinplace_20210730_163431_fid-d7916963_is-cfbebfc6.json) |
| SAGAN-128x128-BigGAN Schedule Best IS  | ImageNet |     w/o      |     1     |                 32x8                 |   1000000    | 826000 | 69.5350 | 12.8295 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sagan/sagan_128_woReLUinplace_noaug_bigGAN_Glr1e-4_Dlr-4e-4_ndisc1_imagenet1k_b32x8.py) | [model](https://download.openmmlab.com/mmgen/sagan/sagan_128_woReLUinplace_noaug_bigGAN_imagenet1k_b32x8_Glr1e-4_Dlr-4e-4_ndisc1_20210818_210232-3f5686af.pth)  |  [Log](https://download.openmmlab.com/mmgen/sagan/sagan_128_woReLUinplace_noaug_bigGAN_imagenet1k_b32x8_Glr1e-4_Dlr-4e-4_ndisc1_20210818_210232-3f5686af.json)  |
| SAGAN-128x128-BigGAN Schedule Best FID | ImageNet |     w/o      |     1     |                 32x8                 |   1000000    | 826000 | 69.5350 | 12.8295 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sagan/sagan_128_woReLUinplace_noaug_bigGAN_Glr1e-4_Dlr-4e-4_ndisc1_imagenet1k_b32x8.py) | [model](https://download.openmmlab.com/mmgen/sagan/sagan_128_woReLUinplace_noaug_bigGAN_imagenet1k_b32x8_Glr1e-4_Dlr-4e-4_ndisc1_20210818_210232-3f5686af.pth)  |  [Log](https://download.openmmlab.com/mmgen/sagan/sagan_128_woReLUinplace_noaug_bigGAN_imagenet1k_b32x8_Glr1e-4_Dlr-4e-4_ndisc1_20210818_210232-3f5686af.json)  |

'\*' Iteration counting rule in our implementation is different from others. If you want to align with other codebases, you can use the following conversion formula:
```
total_iters (biggan/pytorch studio gan) = our_total_iters / dist_step
```

We also provide converted pre-train models from [Pytorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN).
To be noted that, in Pytorch Studio GAN, **inplace ReLU** is used in generator and discriminator.


|          Models          | Dataset  | Inplace ReLU | n_disc | Total Iters | IS (Our Pipeline) | FID (Our Pipeline) | IS (StudioGAN) | FID (StudioGAN) |                                                 Config                                                  |                                                        Download                                                         |                              Original Download link                              |
| :----------------------: | :------: | :----------: | :----: | :---------: | :---------------: | :----------------: | :------------: | :-------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------: |
|  SAGAN-32x32 StudioGAN   | CIFAR10  |      w       |   5    |   100000    |       9.116       |      10.2011       |     8.680      |     14.009      |  [Config](https://github.com/open-mmlab/mmgeneration/blob/master/configs/_base_/models/sagan_32x32.py)  |   [model](https://download.openmmlab.com/mmgen/sagan/sagan_32_cifar10_convert-studio-rgb_20210730_153321-080da7e2.pth)   | [model](https://drive.google.com/drive/folders/1FA8hcz4MB8-hgTwLuDA0ZUfr8slud5P_) |
| SAGAN0-128x128 StudioGAN | ImageNet |      w       |   1    |   1000000   |      27.367       |      40.1162       |     29.848     |     34.726      | [Config](https://github.com/open-mmlab/mmgeneration/blob/master/configs/_base_/models/sagan_128x128.py) | [model](https://download.openmmlab.com/mmgen/sagan/sagan_128_imagenet1k_convert-studio-rgb_20210730_153357-eddb0d1d.pth) | [model](https://drive.google.com/drive/folders/1ZYaqeeumDgxOPDhRR5QLeLFIpgBJ9S6B) |



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
