# Analyzing and Improving the Image Quality of Stylegan (CVPR'2020)

## Introduction
<!-- [ALGORITHM] -->
```latex
@inproceedings{karras2020analyzing,
  title={Analyzing and improving the image quality of stylegan},
  author={Karras, Tero and Laine, Samuli and Aittala, Miika and Hellsten, Janne and Lehtinen, Jaakko and Aila, Timo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8110--8119},
  year={2020}
}

```
## Results and Models
<div align="center">
  <b> Results (compressed) from StyleGAN2 config-f trained by MMGeneration</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/113825919-25433100-97b4-11eb-84f7-5c66b3cfbc68.png" width="800"/>
</div>

|                Model                |     Comment     | FID50k |    P&R50k     |                                                            Config                                                             |                                                                Download                                                                 |
| :---------------------------------: | :-------------: | :----: | :-----------: | :---------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
|    stylegan2_config-f_ffhq_1024     | official weight | 2.8134 | 62.856/49.400 |      [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py)       |  [model](https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth)  |
| stylegan2_config-f_lsun-car_384x512 | official weight | 5.4316 | 65.986/48.190 |   [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_lsun-car_384x512_b4x8.py)   |  [model](https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth)   |
|    stylegan2_config-f_horse_256     | official weight |   -    |       -       | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_lsun-horse_256_b4x8_800k.py)  | [model](https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth)  |
|    stylegan2_config-f_church_256    | official weight |   -    |       -       | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_lsun-church_256_b4x8_800k.py) | [model](https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth) |
|     stylegan2_config-f_cat_256      | official weight |   -    |       -       |  [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_lsun-cat_256_b4x8_800k.py)   |  [model](https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth)   |
|     stylegan2_config-f_ffhq_256     |  our training   | 3.992  | 69.012/40.417 |    [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py)     |             [model](https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth)              |
|    stylegan2_config-f_ffhq_1024     |  our training   | 2.8185 | 68.236/49.583 |      [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py)       |             [model](https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth)             |
| stylegan2_config-f_lsun-car_384x512 |  our training   | 2.4116 | 66.760/50.576 |   [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_lsun-car_384x512_b4x8.py)   |      [model](https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_lsun-car_384x512_b4x8_1800k_20210424_160929-fc9072ca.pth)       |


## FP16 Support and Experiments

Currently, we have supported FP16 training for StyleGAN2, and here are the results for the mixed-precision training. (Experiments for FFHQ1024 will come soon.)

<div align="center">
  <b> Evaluation FID for FP32 and FP16 training </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/117523645-18e90880-afec-11eb-9327-da14362b8bfd.png" width="600"/>
</div>

As shown in the figure, we provide **3** ways to do mixed-precision training for `StyleGAN2`:

* [stylegan2_c2_fp16_PL-no-scaler](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_fp16_partial-GD_PL-no-scaler_ffhq_256_b4x8_800k.py): In this setting, we try our best to follow the official FP16 implementation in [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada). Similar to the official version, we only adopt FP16 training for the higher-resolution feature maps (the last 4 stages in G and the first 4 stages). Note that we do not adopt the `clamp` way to avoid gradient overflow used in the official implementation. We use the `autocast` function from `torch.cuda.amp` package.
* [stylegan2_c2_fp16-globalG-partialD_PL-R1-no-scaler](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_fp16-globalG-partialD_PL-R1-no-scaler_ffhq_256_b4x8_800k.py): In this config, we try to adopt mixed-precision training for the whole generator, but in partial discriminator (the first 4 higher-resolution stages). Note that we do not apply the loss scaler in the path length loss and gradient penalty loss. Because we always meet divergence after adopting the loss scaler to scale the gradient in these two losses.
* [stylegan2_c2_apex_fp16_PL-R1-no-scaler](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_apex_fp16_PL-R1-no-scaler_ffhq_256_b4x8_800k.py): In this setting, we adopt the [APEX](https://github.com/NVIDIA/apex) toolkit to implement mixed-precision training with multiple loss/gradient scalers. In APEX, you can assign different loss scalers for the generator and the discriminator respectively. Note that we still ignore the gradient scaler in the path length loss and gradient penalty loss.


|                                 Model                                 |                 Comment                 | Dataset | FID50k |                                                                            Config                                                                            |                                                                                                              Download                                                                                                               |
| :-------------------------------------------------------------------: | :-------------------------------------: | :-----: | :----: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                      stylegan2_config-f_ffhq_256                      |                baseline                 | FFHQ256 | 3.992  |                    [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py)                    |                                                            [ckpt](https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth)                                                            |
|     stylegan2_c2_fp16_partial-GD_PL-no-scaler_ffhq_256_b4x8_800k      |         partial layers in fp16          | FFHQ256 | 4.331  |     [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_fp16_partial-GD_PL-no-scaler_ffhq_256_b4x8_800k.py)      |     [ckpt](https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_fp16_partial-GD_PL-no-scaler_ffhq_256_b4x8_800k_20210508_114854-dacbe4c9.pth?versionId=CAEQKxiBgIDNhrOoyhciIGM1OTIwZmFiYWM4MzQyNzE4YzlkYjYxMTNhZjU1ZGFj)      |
| stylegan2_c2_fp16-globalG-partialD_PL-R1-no-scaler_ffhq_256_b4x8_800k |           the whole G in fp16           | FFHQ256 | 4.362  | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_fp16-globalG-partialD_PL-R1-no-scaler_ffhq_256_b4x8_800k.py) | [ckpt](https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_fp16-globalG-partialD_PL-R1-no-scaler_ffhq_256_b4x8_800k_20210508_114930-ef8270d4.pth?versionId=CAEQKxiBgIDOhrOoyhciIDM4ZTQxYzkxZTE4ZjQ2ZjM4ZmU3YzlhOWNkYWI1OWQ1) |
|       stylegan2_c2_apex_fp16_PL-R1-no-scaler_ffhq_256_b4x8_800k       | the whole G&D in fp16 + two loss scaler | FFHQ256 | 4.614  |       [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_apex_fp16_PL-R1-no-scaler_ffhq_256_b4x8_800k.py)       |       [ckpt](https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_apex_fp16_PL-R1-no-scaler_ffhq_256_b4x8_800k_20210508_114701-c2bb8afd.pth?versionId=CAEQKxiBgMDQhrOoyhciIGE0ZGJkOWM2MTNjMzQ3Mjk4Y2NmMWMyNTViOTNiZTNh)       |


In addition, we also provide `QuickTestImageDataset` to users for quickly checking whether the code can be run correctly. It's more important for FP16 experiments, because some cuda operations may no support mixed precision training. Esepcially for `APEX`, you can use [this config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_apex_fp16_quicktest_ffhq_256_b4x8_800k.py) in your local machine by running:

```bash
bash tools/dist_train.sh \
  configs/styleganv2/stylegan2_c2_apex_fp16_quicktest_ffhq_256_b4x8_800k.py 1 \
  --work-dir ./work_dirs/quick-test
```

With a simliar way, users can switch to [config for partial-GD](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_fp16_quicktest_ffhq_256_b4x8_800k.py) and [config for globalG-partialD](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_fp16-globalG-partialD_PL-R1-no-scaler_ffhq_256_b4x8_800k.py) to test the other two mixed precision training configuration.


*Note that to use the [APEX](https://github.com/NVIDIA/apex) toolkit, you have to installed it following the official guidance. (APEX is not included in our requirements.) If you are using GPUs without tensor core, you would better to switch to the newer PyTorch version (>= 1.7,0). Otherwise, the APEX installation or running may meet several bugs.*

## About Different Implementations of FID Metric

|            Model             |     Comment     | FID50k |   FID Version   |                                                       Config                                                       |                                                                                                                       Download                                                                                                                        |
| :--------------------------: | :-------------: | :----: | :-------------: | :----------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| stylegan2_config-f_ffhq_1024 | official weight | 2.8732 | Tero's StyleGAN | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py) | [model](https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth) &#124; [FID-Reals](https://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/ffhq-1024-50k-stylegan.pkl) |
| stylegan2_config-f_ffhq_1024 |  our training   | 2.9413 | Tero's StyleGAN | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py) |            [model](https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth) &#124; [FID-Reals](https://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/ffhq-1024-50k-stylegan.pkl)            |
| stylegan2_config-f_ffhq_1024 | official weight | 2.8134 |   Our PyTorch   | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py) |   [model](https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth) &#124; [FID-Reals](https://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/ffhq-1024-50k-rgb.pkl)    |
| stylegan2_config-f_ffhq_1024 |  our training   | 2.8185 |   Our PyTorch   | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py) |              [model](https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth) &#124; [FID-Reals](https://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/ffhq-1024-50k-rgb.pkl)               |

In this table, we observe that the FID with Tero's inception network is similar to that with PyTorch Inception (in MMGeneration). Thus, we use the FID with PyTorch's Inception net (but the weight is not the official model zoo) by default. Because it can be run on different PyTorch versions. If you use [Tero's Inception net](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt), your PyTorch must meet `>=1.6.0`.

More precalculated inception pickle files are listed here:

- FFHQ 256x256 real inceptions, PyTorch InceptionV3. [download](https://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/ffhq-256-50k-rgb.pkl)
- LSUN-Car 384x512 real inceptions, PyTorch InceptionV3. [download](https://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/lsun-car-512_50k_rgb.pkl)

## About Different Implementation and Setting of PR Metric

|                     Model                      |           P&R Details            | Precision | Recall |
| :--------------------------------------------: | :------------------------------: | :-------: | :----: |
| stylegan2_config-f_ffhq_1024 (official weight) |  use Tero's VGG16, P&R50k_full   |  67.876   | 49.299 |
| stylegan2_config-f_ffhq_1024 (official weight) |     use Tero's VGG16, P&R50k     |  62.856   | 49.400 |
| stylegan2_config-f_ffhq_1024 (official weight) | use PyTorch's VGG16, P&R50k_full |  67.662   | 55.460 |

As shown in this table, `P&R50k_full` is the metric used in StyleGANv1 and StyleGANv2. `full` indicates that we use the whole dataset for extracting the real distribution, e.g., 70000 images in FFHQ dataset. However, adopting the VGG16 provided from [Tero](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt) requires that your PyTorch version must fulfill `>=1.6.0`. Be careful about using the PyTorch's VGG16 to extract features, which will cause higher precision and recall.
