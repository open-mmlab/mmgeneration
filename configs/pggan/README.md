# Progressive Growing of GANs for Improved Quality, Stability, and Variation

## Introduction
<!-- [ALGORITHM] -->
```latex
@article{karras2017progressive,
  title={Progressive growing of gans for improved quality, stability, and variation},
  author={Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen, Jaakko},
  journal={arXiv preprint arXiv:1710.10196},
  year={2017}
}
```
<div align="center">
  <b> Results (compressed) from our PGGAN trained in CelebA-HQ@1024</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114009864-1df45400-9896-11eb-9d25-da9eabfe02ce.png" width="800"/>
</div>


|     Models      |    Details     | MS-SSIM |     SWD(xx,xx,xx,xx/avg)     |                                                        Config                                                         |                                                   Download                                                    |
| :-------------: | :------------: | :-----: | :--------------------------: | :-------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
|  pggan_128x128  | celeba-cropped | 0.3023  | 3.42, 4.04, 4.78, 20.38/8.15 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/pggan/pggan_celeba-cropped_128_g8_12Mimgs.py) |  [model](https://download.openmmlab.com/mmgen/pggan/pggan_celeba-cropped_128_g8_20210408_181931-85a2e72c.pth)  |
|  pggan_128x128  |  lsun-bedroom  | 0.0602  |  3.5, 2.96, 2.76, 9.65/4.72  |  [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/pggan/pggan_lsun-bedroom_128_g8_12Mimgs.py)  | [model](https://download.openmmlab.com/mmgen/pggan/pggan_lsun-bedroom_128x128_g8_20210408_182033-5e59f45d.pth) |
| pggan_1024x1024 |   celeba-hq    | 0.3379  | 8.93, 3.98, 3.07, 2.64/4.655 |   [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/pggan/pggan_celeba-hq_1024_g8_12Mimg.py)    |    [model](https://download.openmmlab.com/mmgen/pggan/pggan_celeba-hq_1024_g8_20210408_181911-f1ef51c3.pth)    |
