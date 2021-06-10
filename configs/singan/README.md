# Singan: Learning a Generative Model from a Single Natural Image (ICCV'2019)

## Introduction
<!-- [ALGORITHM] -->

```latex
@inproceedings{shaham2019singan,
  title={Singan: Learning a generative model from a single natural image},
  author={Shaham, Tamar Rott and Dekel, Tali and Michaeli, Tomer},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4570--4580},
  year={2019}
}
```

## Results and Models

<div align="center">
  <b> SinGAN balloons</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/113702715-7861a900-970c-11eb-9dd8-0743cc30701f.png" width="800"/>
</div>


| Model  |                                      Data                                       | Num Scales |                                               Config                                               |                                                                                               Download                                                                                                |
| :----: | :-----------------------------------------------------------------------------: | :--------: | :------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SinGAN | [balloons.png](https://download.openmmlab.com/mmgen/dataset/singan/balloons.png) |     8      | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/singan/singan_balloons.py) | [ckpt](https://download.openmmlab.com/mmgen/singan/singan_balloons_20210406_191047-8fcd94cf.pth) &#124; [pkl](https://download.openmmlab.com/mmgen/singan/singan_balloons_20210406_191047-8fcd94cf.pkl) |
| SinGAN |  [fish.jpg](https://download.openmmlab.com/mmgen/dataset/singan/fish-crop.jpg)   |     10     |   [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/singan/singan_fish.py)   |      [ckpt](https://download.openmmlab.com/mmgen/singan/singan_fis_20210406_201006-860d91b6.pth) &#124; [pkl](https://download.openmmlab.com/mmgen/singan/singan_fis_20210406_201006-860d91b6.pkl)      |
| SinGAN | [bohemian.png](https://download.openmmlab.com/mmgen/dataset/singan/bohemian.png) |     10     | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/singan/singan_bohemian.py) | [ckpt](https://download.openmmlab.com/mmgen/singan/singan_bohemian_20210406_175439-f964ee38.pth) &#124; [pkl](https://download.openmmlab.com/mmgen/singan/singan_bohemian_20210406_175439-f964ee38.pkl) |


## Notes for using SinGAN

When training SinGAN models, users may obtain the number of scales (stages) in advance via the following commands. This number is important for constructing config file, which is related to the generator, discriminator, the training iterations and so on.

```shell
>>> from mmgen.datasets.singan_dataset import create_real_pyramid
>>> import mmcv
>>> real = mmcv.imread('real_img_path')
>>> _, _, num_scales = create_real_pyramid(real, min_size=25, max_size=300, scale_factor_init=0.75)
```

When testing SinGAN models, users have to modify the config file to add the `test_cfg`. As shown in `configs/singan/singan_balloons.py`, the only thing you need to do is add the path for `pkl` data. There are some important data containing in the pickle files which you can download from our website.

```python
test_cfg = dict(
    _delete_ = True
    pkl_data = 'path to pkl data'
)
```
