# Geometric GAN

## Introduction
<!-- [ALGORITHM] -->

```latex
@article{lim2017geometric,
  title={Geometric gan},
  author={Lim, Jae Hyun and Ye, Jong Chul},
  journal={arXiv preprint arXiv:1705.02894},
  year={2017}
}
```

## Results and models

<div align="center">
  <b> GGAN 64x64, CelebA-Cropped</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/116691577-9067d800-a9ed-11eb-8ea4-be79884d8502.PNG" width="800"/>
</div>

|    Models    |    Dataset     |               SWD               | MS-SSIM |   FID   |                                                                 Config                                                                 |                                                                                                                                                                                                             Download                                                                                                                                                                                                             |
| :----------: | :------------: | :-----------------------------: | :-----: | :-----: | :------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  GGAN 64x64  | CelebA-Cropped |    11.18, 12.21, 39.16/20.85    | 0.3318  | 20.1797 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/ggan/ggan_celeba-cropped_dcgan-archi_lr-1e-3_64_b128x1_12m.py) |                                                                                                                      [model](https://download.openmmlab.com/mmgen/ggan/ggan_celeba-cropped_dcgan-archi_lr-1e-3_64_b128x1_12m.pth?versionId=CAEQKhiBgICoybGKyRciIDg2M2UyMTMwNGVhMTQ3NDA4NWUxYTcxOTMyNjc5MjQ4)  &#124; [log](https://download.openmmlab.com/mmgen/ggan/ggan_celeba-cropped_dcgan-archi_lr-1e-3_64_b128x1_12m_20210430_113839.log.json?versionId=CAEQKhiBgICmrraryRciIGU0NjU0YmNhN2UxZjQwMjA4OWJkMjkwNjkwMGYwMmU2)                                                                                                                   |
| GGAN 128x128 | CelebA-Cropped | 9.81, 11.29, 19.22, 47.79/22.03 | 0.3149  | 18.7647 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/ggan/ggan_celeba-cropped_dcgan-archi_lr-1e-4_128_b64x1_10m.py) | [model](https://download.openmmlab.com/mmgen/ggan/ggan_celeba-cropped_dcgan-archi_lr-1e-4_128_b64x1_10m_20210430_143027-516423dc.pth?versionId=CAEQKhiBgMCp9J6HyRciIDg3YzMyYzliM2M5YTRkZDBhNTY4MWIwMWIxZjE4MzU5) &#124; [log](https://download.openmmlab.com/mmgen/ggan/ggan_celeba-cropped_dcgan-archi_lr-1e-4_128_b64x1_10m_20210423_154258.log.json?versionId=CAEQKhiBgMCy9J6HyRciIDAwNGRkNTY1MjQzMjQwMTdhZDFmOTUyYmVkYzIxNmU5) |
|  GGAN 64x64  |  LSUN-Bedroom  |      9.1, 6.2, 12.27/9.19       | 0.0649  | 85.6629 |  [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/ggan/ggan_lsun-bedroom_lsgan_archi_lr-1e-4_64_b128x1_20m.py)  |   [model](https://download.openmmlab.com/mmgen/ggan/ggan_lsun-bedroom_lsgan_archi_lr-1e-4_64_b128x1_20m_20210430_143114-5d99b76c.pth?versionId=CAEQKhiBgICZ9J6HyRciIGI2MDA4ZjJlMmUxODRjODk4OTIyMzkzMmE1MDBhNWJk) &#124; [log](https://download.openmmlab.com/mmgen/ggan/ggan_lsun-bedroom_lsgan_archi_lr-1e-4_64_b128x1_20m_20210428_202027.log.json?versionId=CAEQKhiBgMCu9J6HyRciIDZiMTExODExYmEwNTRhYjRhYzE0YTU1MTM5NzE5Y2Ew)   |

Note: In the original implementation of [GGAN](https://github.com/lim0606/pytorch-geometric-gan), they set `G_iters` to 10. However our framework does not support `G_iters` currently, so we dropped the settings in the original implementation and conducted several experiments with our own settings. We have shown above the experiment results with the lowest `fid` score. \
Original settings and our settings:
|       Models       |     Dataset    | Architecture | optimizer |  lr_G  |  lr_D  | G_iters | D_iters |
|:------------------:|:--------------:|:------------:|:---------:|:------:|:------:|:-------:|:-------:|
| GGAN(origin) 64x64 | CelebA-Cropped |  dcgan-archi |  RMSprop  | 0.0002 | 0.0002 |    10   |    1    |
|  GGAN(ours) 64x64  | CelebA-Cropped |  dcgan-archi |    Adam   |  0.001 |  0.001 |    1    |    1    |
| GGAN(origin) 64x64 |  LSUN-Bedroom  |  dcgan-archi |  RMSprop  | 0.0002 | 0.0002 |    10   |    1    |
|  GGAN(ours) 64x64  |  LSUN-Bedroom  |  lsgan-archi |    Adam   | 0.0001 | 0.0001 |    1    |    1    |
