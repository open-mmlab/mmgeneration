# Least Squares Generative Adversarial Networks

## Introduction
<!-- [ALGORITHM] -->

```latex
@inproceedings{mao2017least,
  title={Least squares generative adversarial networks},
  author={Mao, Xudong and Li, Qing and Xie, Haoran and Lau, Raymond YK and Wang, Zhen and Paul Smolley, Stephen},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2794--2802},
  year={2017}
}
```

## Results and models

<div align="center">
  <b> LSGAN 64x64, CelebA-Cropped</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/116498716-f4e74200-a8dc-11eb-9c28-5549d96e20a6.png" width="800"/>
</div>


|    Models     |    Dataset     |               SWD               | MS-SSIM |   FID   |                                                                  Config                                                                  |                                                                                                                                                                                                              Download                                                                                                                                                                                                               |
| :-----------: | :------------: | :-----------------------------: | :-----: | :-----: | :--------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  LSGAN 64x64  | CelebA-Cropped |     6.16, 6.83, 37.64/16.87     | 0.3216  | 11.9258 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/lsgan/lsgan_celeba-cropped_dcgan-archi_lr-1e-3_64_b128x1_12m.py) | [model](https://download.openmmlab.com/mmgen/lsgan/lsgan_celeba-cropped_dcgan-archi_lr-1e-3_64_b128x1_12m_20210429_144001-92ca1d0d.pth?versionId=CAEQKhiBgIDS1crxyBciIDAxNzgzOTE2ZDNiNDQ4ZGU4MmI5MGY1YjdmNjg0Nzkw)&#124; [log](https://download.openmmlab.com/mmgen/lsgan/lsgan_celeba-cropped_dcgan-archi_lr-1e-3_64_b128x1_12m_20210422_131925.log.json?versionId=CAEQKhiBgMDdwvHxyBciIGQwOThmY2MzNGY4NjQ4MjE5NzdmYzQwYjhmMTcyMjIy) |
|  LSGAN 64x64  |  LSUN-Bedroom  |      5.66, 9.0, 18.6/11.09      | 0.0671  | 30.7390 |  [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/lsgan/lsgan_lsun-bedroom_dcgan-archi_lr-1e-4_64_b128x1_12m.py)  |   [model](https://download.openmmlab.com/mmgen/lsgan/lsgan_lsun-bedroom_dcgan-archi_lr-1e-4_64_b128x1_12m_20210429_144602-ec4ec6bb.pth?versionId=CAEQKhiBgMDc1crxyBciIDc0NGE5OTc1YmUwNzQ1OTg4YzY5MDkyOTYyY2VhZGVm)&#124; [log](https://download.openmmlab.com/mmgen/lsgan/lsgan_lsun-bedroom_dcgan-archi_lr-1e-4_64_b128x1_12m_20210423_005020.log.json?versionId=CAEQKhiBgIDdwvHxyBciIDg4YWI3ZGRlYzNmMDRmOTc5OWU5NWJkNTZjMjQ0MjFm)   |
| LSGAN 128x128 | CelebA-Cropped | 21.66, 9.83, 16.06, 70.76/29.58 | 0.3691  | 38.3752 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/lsgan/lsgan_celeba-cropped_dcgan-archi_lr-1e-4_128_b64x1_10m.py) | [model](https://download.openmmlab.com/mmgen/lsgan/lsgan_celeba-cropped_dcgan-archi_lr-1e-4_128_b64x1_10m_20210429_144229-01ba67dc.pth?versionId=CAEQKhiBgMDS1crxyBciIGU4N2JhNGQ0YjU2YTQ2OWI5MWUxZmQ1NmUwNzY3MmUx)&#124; [log](https://download.openmmlab.com/mmgen/lsgan/lsgan_celeba-cropped_dcgan-archi_lr-1e-4_128_b64x1_10m_20210423_132126.log.json?versionId=CAEQKhiBgICMw_HxyBciIDQ2MzZlNTViMTNjNTRjN2JhNWRlMzViMzg5YzlhODc3) |
| LSGAN 128x128 |  LSUN-Bedroom  |  19.52, 9.99, 7.48, 14.3/12.82  | 0.0612  | 51.5500 |  [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/lsgan/lsgan_lsun-bedroom_lsgan-archi_lr-1e-4_128_b64x1_10m.py)  |   [model](https://download.openmmlab.com/mmgen/lsgan/lsgan_lsun-bedroom_lsgan-archi_lr-1e-4_128_b64x1_10m_20210429_155605-cf78c0a8.pth?versionId=CAEQKhiBgMDnw8LyyBciIGQzNmRjYjI5ODA1OTQ4Mjc5MGFiZGRmNzJjZTU1NDA1)&#124; [log](https://download.openmmlab.com/mmgen/lsgan/lsgan_lsun-bedroom_lsgan-archi_lr-1e-4_128_b64x1_10m_20210429_142302.log.json?versionId=CAEQKhiBgMCDo8jyyBciIDE1YTRmNGYyZTYyYzQyZjdiZGMxNjIxMWFjM2UwMzM2)   |
