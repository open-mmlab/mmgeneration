# Self-attention generative adversarial networks

## Introduction
<!-- [ALGORITHM] -->
```latex
@inproceedings{zhang2019self,
  title={Self-attention generative adversarial networks},
  author={Zhang, Han and Goodfellow, Ian and Metaxas, Dimitris and Odena, Augustus},
  booktitle={International conference on machine learning},
  pages={7354--7363},
  year={2019},
  organization={PMLR}
}
```
<div align="center">
  <b> Results from our SAGAN trained in CIFAR10</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/28132635/127619657-67f2e62d-52e4-43d2-931f-6d0e6e019813.png" width="400"/>
</div>

|  Models   | Dataset  | Inplace ReLU | n_disc | Total Batchsize (BZ_PER_GPU \* NGPU) | Total Iters* |                                                                               Best IS (Iter)                                                                                |                                                                               Best FID (Iter)                                                                                |                                                                       Config                                                                        |                                                                               Log                                                                               |
| :-------: | :------: | :----------: | :----: | :----------------------------------: | :----------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| sagan_32  | CIFAR10  |     w/o      |   5    |                 64x1                 |    500000    |        [ 9.3217 (400000) ](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_woReUinplace_is-iter400000_20210730_125743-4008a9ca.pth)         |        [ 9.4252 (480000) ](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_woReUinplace_fid-iter480000_20210730_125449-d50568a4.pth )        |        [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sagan/sagan_32_woReLUinplace_cifar10_b64x1_lr-2e-4_ndisc5.py)        |       [Log](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_woReUinplace_20210730_125449_fid-d50568a4_is-04008a9ca.json)        |
| sagan_32  | CIFAR10  |      w       |   5    |                 64x1                 |    500000    |        [ 9.2286 (380000) ](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_wReLUinplace_is-iter380000_20210730_124937-c77b4d25.pth)         |        [ 10.7781 (460000) ](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_wReLUinplace_fid-iter460000_20210730_125155-cbefb354.pth)        |        [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sagan/sagan_32_wReLUinplace_cifar10_b64x1_lr-2e-4_ndisc5.py)         |        [Log](https://download.openmmlab.com/mmgen/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_wReLUinplace_20210730_125155_fid-cbefb354_is-c77b4d25.json)        |
| sagan_128 | ImageNet |     w/o      |   1    |                 64x4                 |   1000000    | [ 31.5938 (980000) ](https://download.openmmlab.com/mmgen/sagan/sagan_imagenet1k_128_Glr1e-4_Dlr4e-4_ndisc1_b32x4_woReLUinplace_is-iter980000_20210730_163140-cfbebfc6.pth) | [ 34.7838 (950000) ](https://download.openmmlab.com/mmgen/sagan/sagan_imagenet1k_128_Glr1e-4_Dlr4e-4_ndisc1_b32x4_woReLUinplace_fid-iter950000_20210730_163431-d7916963.pth) | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/sagan/sagan_128_woReLUinplace_imagenet1k_b64x4_Glr-1e-4_Dlr-4e-4_ndisc1.py) | [Log](https://download.openmmlab.com/mmgen/sagan/sagan_imagenet1k_128_Glr1e-4_Dlr4e-4_ndisc1_b32x4_woReLUinplace_20210730_163431_fid-d7916963_is-cfbebfc6.json) |

'\*' Iteration counting rule in our implementation is different from others. If you want to align with other codebases, you can use the following conversion formula:
```
total_iters (biggan/pytorch studio gan) = our_total_iters / n_disc
```

We also provide converted pre-train models from [Pytorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN).
To be noted that, in Pytorch Studio GAN, **inplace ReLU** is used in generator and discriminator.

|  Models   | Dataset  | Inplace ReLU | n_disc | Total Iters | IS (Ours Pipeline) | FID (Ours Pipeline) | IS (StudioGAN) | FID (StudioGAN) |                                                          Download                                                           |                                Original Download link                                |
| :-------: | :------: | :----------: | :----: | :---------: | :----------------: | :-----------------: | :----------: | :-----------: | :-------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------: |
| sagan_32  | CIFAR10  |      w       |   5    |   100000    |       9.116        |       10.2011       |    8.680     |    14.009     |   [Download](https://download.openmmlab.com/mmgen/sagan/sagan_32_cifar10_convert-studio-rgb_20210730_153321-080da7e2.pth)   | [Download](https://drive.google.com/drive/folders/1FA8hcz4MB8-hgTwLuDA0ZUfr8slud5P_) |
| sagan_128 | ImageNet |      w       |   1    |   1000000   |       27.367       |       40.1162       |    29.848    |    34.726     | [Download](https://download.openmmlab.com/mmgen/sagan/sagan_128_imagenet1k_convert-studio-rgb_20210730_153357-eddb0d1d.pth) | [Download](https://drive.google.com/drive/folders/1ZYaqeeumDgxOPDhRR5QLeLFIpgBJ9S6B) |


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
python tools/utils/inception_stat.py --data-cfg configs/_base_/datasets/imagenet_inception_stat.py --pklname imagenet.pkl --no-shuffle --inception-style stylegan --num-samples -1 --subset train
```
