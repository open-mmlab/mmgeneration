# Improved-DDPM

> [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Denoising diffusion probabilistic models (DDPM) are a class of generative models which have recently been shown to produce excellent samples. We show that with a few simple modifications, DDPMs can also achieve competitive log-likelihoods while maintaining high sample quality. Additionally, we find that learning variances of the reverse diffusion process allows sampling with an order of magnitude fewer forward passes with a negligible difference in sample quality, which is important for the practical deployment of these models. We additionally use precision and recall to compare how well DDPMs and GANs cover the target distribution. Finally, we show that the sample quality and likelihood of these models scale smoothly with model capacity and training compute, making them easily scalable. We release our code at this https URL.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/147938745-a5ae5b6f-b0e1-4db6-9768-44c1c6c43755.png"/>
</div>

## Results and Models

<div align="center">
  <b> Denoising process of Improve-DDPM trained on CIFAR10 and ImageNet-64x64</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/28132635/148009529-46d3fc28-eaeb-4ae9-8831-fa9edea334cc.gif" width="300"/> &nbsp;&nbsp;
  <img src="https://user-images.githubusercontent.com/28132635/147954424-1c9e4623-5bed-4cdc-b49c-ab17d619f748.gif" width="300"/>
</div>

|             Models             |  Dataset   |   FID   |                                   Config                                    |                                    Download                                    |
| :----------------------------: | :--------: | :-----: | :-------------------------------------------------------------------------: | :----------------------------------------------------------------------------: |
| Improve-DDPM 32x32 Dropout=0.3 |  CIFAR10   | 3.8848  | [config](https://github.com/open-mmlab/mmgeneration/blob/master/configs/improved_ddpm/ddpm_cosine_hybird_timestep-4k_drop0.3_cifar10_32x32_b8x16_500k.py) | [model](https://download.openmmlab.com/mmgen/improved_ddpm/ddpm_cosine_hybird_timestep-4k_drop0.3_cifar10_32x32_b8x16_500k_20220103_222621-2f42f476.pth)\| [log](https://download.openmmlab.com/mmgen/improved_ddpm/ddpm_cosine_hybird_timestep-4k_drop0.3_cifar10_32x32_b8x16_500k_20220103_222621-2f42f476.json) |
|       Improve-DDPM 64x64       | ImageNet1k | 13.5181 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/improve_ddpm/ddpm_cosine_hybird_timestep-4k_imagenet1k_64x64_b8x16_1500k.py) | [model](https://download.openmmlab.com/mmgen/improved_ddpm/ddpm_cosine_hybird_timestep-4k_imagenet1k_64x64_b8x16_1500k_20220103_223919-b8f1a310.pth)\| [log](https://download.openmmlab.com/mmgen/improved_ddpm/ddpm_cosine_hybird_timestep-4k_imagenet1k_64x64_b8x16_1500k_20220103_223919-b8f1a310.json) |
| Improve-DDPM 64x64 Dropout=0.3 | ImageNet1k | 13.4094 | [config](https://github.com/open-mmlab/mmgeneration/blob/master/configs/improved_ddpm/ddpm_cosine_hybird_timestep-4k_drop0.3_imagenet1k_64x64_b8x16_1500k.py) | [model](https://download.openmmlab.com/mmgen/improved_ddpm/ddpm_cosine_hybird_timestep-4k_drop0.3_imagenet1k_64x64_b8x16_1500k_20220103_224427-7bb55975.pth)\| [log](https://download.openmmlab.com/mmgen/improved_ddpm/ddpm_cosine_hybird_timestep-4k_drop0.3_imagenet1k_64x64_b8x16_1500k_20220103_224427-7bb55975.json) |

`FID` comparison with official:

| Dataset  | CIFAR10  | ImageNet1k-64x64 |
| :------: | :------: | :--------------: |
|   Ours   |  3.8848  |   **13.5181**    |
| Official | **3.19** |       19.2       |

For FID evaluation, we follow the pipeline of [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch/blob/98459431a5d618d644d54cd1e9fceb1e5045648d/calculate_inception_moments.py#L52), where the whole training set is adopted to extract inception statistics, and Pytorch Studio GAN uses 50000 randomly selected samples. Besides, we also use [Tero's Inception](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) for feature extraction.

You can download the preprocessed inception state by the following url: [CIFAR10](https://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/cifar10.pkl) and [ImageNet1k-64x64](https://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/imagenet_64x64.pkl).

You can use following commands to extract those inception states by yourself.

```
# For CIFAR10
python tools/utils/inception_stat.py --data-cfg configs/_base_/datasets/cifar10_inception_stat.py --pklname cifar10.pkl --no-shuffle --inception-style stylegan --num-samples -1 --subset train

# For ImageNet1k-64x64
python tools/utils/inception_stat.py --data-cfg configs/_base_/datasets/imagenet_64x64_inception_stat.py --pklname imagenet_64x64.pkl --no-shuffle --inception-style stylegan --num-samples -1 --subset train
```

## Citation

<summary align="right"><a href="https://arxiv.org/abs/2102.09672">Improve-DDPM (arXiv'2021)</a></summary>

```latex
@article{nichol2021improved,
  title={Improved denoising diffusion probabilistic models},
  author={Nichol, Alex and Dhariwal, Prafulla},
  journal={arXiv preprint arXiv:2102.09672},
  year={2021}
}
```
