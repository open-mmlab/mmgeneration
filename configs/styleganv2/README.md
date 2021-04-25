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
|    stylegan2_config-f_ffhq_1024     | official weight | 2.8134 | 62.856/49.400 |      [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py)       |  [model](http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth)  |
| stylegan2_config-f_lsun-car_384x512 | official weight | 5.4316 | 65.986/48.190 |   [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_lsun-car_384x512_b4x8.py)   |  [model](http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth)   |
|    stylegan2_config-f_horse_256     | official weight |   -    |       -       | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_lsun-horse_256_b4x8_800k.py)  | [model](http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth)  |
|    stylegan2_config-f_church_256    | official weight |   -    |       -       | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_lsun-church_256_b4x8_800k.py) | [model](http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth) |
|     stylegan2_config-f_cat_256      | official weight |   -    |       -       |  [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_lsun-cat_256_b4x8_800k.py)   |  [model](http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth)   |
|     stylegan2_config-f_ffhq_256     |  our training   | 4.892  | 69.006/40.439 |    [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py)     |             [model](http://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth)              |
|    stylegan2_config-f_ffhq_1024     |  our training   | 2.8185 | 68.236/49.583 |      [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py)       |             [model](http://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth)             |
| stylegan2_config-f_lsun-car_384x512 |  our training   | 2.4116 | 66.760/50.576 |   [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_lsun-car_384x512_b4x8.py)   |      [model](http://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_lsun-car_384x512_b4x8_1800k_20210424_160929-fc9072ca.pth)       |



## About Different Implementations of FID Metric

|            Model             |     Comment     | FID50k |   FID Version   |                                                       Config                                                       |                                                                                                                       Download                                                                                                                        |
| :--------------------------: | :-------------: | :----: | :-------------: | :----------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| stylegan2_config-f_ffhq_1024 | official weight | 2.8732 | Tero's StyleGAN | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py) | [model](http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth) &#124; [FID-Reals](http://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/ffhq-1024-50k-stylegan.pkl) |
| stylegan2_config-f_ffhq_1024 |  our training   | 2.9413 | Tero's StyleGAN | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py) |            [model](http://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth) &#124; [FID-Reals](http://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/ffhq-1024-50k-stylegan.pkl)            |
| stylegan2_config-f_ffhq_1024 | official weight | 2.8134 |   Our PyTorch   | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py) |   [model](http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth) &#124; [FID-Reals](http://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/ffhq-1024-50k-rgb.pkl)    |
| stylegan2_config-f_ffhq_1024 |  our training   | 2.8185 |   Our PyTorch   | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py) |              [model](http://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth) &#124; [FID-Reals](http://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/ffhq-1024-50k-rgb.pkl)               |

In this table, we observe that the FID with Tero's inception network is similar to that with PyTorch Inception (in MMGeneration). Thus, we use the FID with PyTorch's Inception net (but the weight is not the official model zoo) by default. Because it can be run on different PyTorch versions. If you use [Tero's Inception net](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt), your PyTorch must meet `>=1.6.0`.

More precalculated inception pickle files are listed here:

- FFHQ 256x256 real inceptions, PyTorch InceptionV3. [download](http://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/ffhq-256-50k-rgb.pkl)
- LSUN-Car 384x512 real inceptions, PyTorch InceptionV3. [download](http://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/lsun-car-512_50k_rgb.pkl)

## About Different Implementation and Setting of PR Metric

|                     Model                      |           P&R Details            | Precision | Recall |
| :--------------------------------------------: | :------------------------------: | :-------: | :----: |
| stylegan2_config-f_ffhq_1024 (official weight) |  use Tero's VGG16, P&R50k_full   |  67.876   | 49.299 |
| stylegan2_config-f_ffhq_1024 (official weight) |     use Tero's VGG16, P&R50k     |  62.856   | 49.400 |
| stylegan2_config-f_ffhq_1024 (official weight) | use PyTorch's VGG16, P&R50k_full |  67.662   | 55.460 |

As shown in this table, `P&R50k_full` is the metric used in StyleGANv1 and StyleGANv2. `full` indicates that we use the whole dataset for extracting the real distribution, e.g., 70000 images in FFHQ dataset. However, adopting the VGG16 provided from [Tero](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt) requires that your PyTorch version must fulfill `>=1.6.0`. Be careful about using the PyTorch's VGG16 to extract features, which will cause higher precision and recall.
