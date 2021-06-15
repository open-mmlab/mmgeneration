# Improved Training of Wasserstein GANs

## Introduction
<!-- [ALGORITHM] -->

```latex
@article{gulrajani2017improved,
  title={Improved Training of Wasserstein GANs},
  author={Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and Dumoulin, Vincent and Courville, Aaron},
  journal={arXiv preprint arXiv:1704.00028},
  year={2017}
}
```

## Results and models

<div align="center">
  <b> WGAN-GP 128, CelebA-Cropped</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/113997469-c00e3f00-988a-11eb-81dc-19b05698b74b.png" width="800"/>
</div>

|   Models    |    Dataset     |      Details       |              SWD              | MS-SSIM |                                                               Config                                                                |                                                           Download                                                           |
| :---------: | :------------: | :----------------: | :---------------------------: | :-----: | :---------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: |
| WGAN-GP 128 | CelebA-Cropped |         GN         | 5.87, 9.76, 9.43, 18.84/10.97 | 0.2601  |   [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/wgan-gp/wgangp_GN_celeba-cropped_128_b64x1_160kiter.py)   |   [model](https://download.openmmlab.com/mmgen/wgangp/wgangp_GN_celeba-cropped_128_b64x1_160k_20210408_170611-f8a99336.pth)   |
| WGAN-GP 128 |  LSUN-Bedroom  | GN, GP-lambda = 50 | 11.7, 7.87, 9.82, 25.36/13.69 |  0.059  | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/wgan-gp/wgangp_GN_GP-50_lsun-bedroom_128_b64x1_160kiter.py) | [model](https://download.openmmlab.com/mmgen/wgangp/wgangp_GN_GP-50_lsun-bedroom_128_b64x1_130k_20210408_170509-56f2a37c.pth) |
