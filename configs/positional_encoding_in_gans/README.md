# Positional Encoding as Spatial Inductive Bias in GANs (CVPR'2021)

## Introduction
<!-- [ALGORITHM] -->

```latex
@article{xu2020positional,
  title={Positional Encoding as Spatial Inductive Bias in GANs},
  author={Xu, Rui and Wang, Xintao and Chen, Kai and Zhou, Bolei and Loy, Chen Change},
  journal={arXiv preprint arXiv:2012.05217},
  year={2020}
}
```

## Results and models for MS-PIE

<div align="center">
  <b> 896x896 results generated from a 256 generator using MS-PIE</b>
  <br/>
  <img src="https://download.openmmlab.com/mmgen/pe_in_gans/mspie_256-896_demo.png" width="800"/>
</div>


|            Models            | Reference in Paper |     Scales     | FID50k |   P&R10k    |                                                                            Config                                                                            |                                                                 Download                                                                  |
| :--------------------------: | :----------------: | :------------: | :----: | :---------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------: |
|  stylegan2_c2_256_baseline   |   Tab.5 config-a   |      256       |  5.56  | 75.92/51.24 |           [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/stylegan2_c2_ffhq_256_b3x8_1100k.py)           |    [model](https://download.openmmlab.com/mmgen/pe_in_gans/stylegan2_c2_config-a_ffhq_256x256_b3x8_1100k_20210406_145127-71d9634b.pth)     |
|  stylegan2_c2_512_baseline   |   Tab.5 config-b   |      512       |  4.91  | 75.65/54.58 |           [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/stylegan2_c2_ffhq_512_b3x8_1100k.py)           |    [model](https://download.openmmlab.com/mmgen/pe_in_gans/stylegan2_c2_config-b_ffhq_512x512_b3x8_1100k_20210406_145142-e85e5cf4.pth)     |
| ms-pie_stylegan2_c2_config-c |   Tab.5 config-c   | 256, 384, 512  |  3.35  | 73.84/55.77 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/mspie-stylegan2_c2_config-c_ffhq_256-512_b3x8_1100k.py)  | [model](https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c2_config-c_ffhq_256-512_b3x8_1100k_20210406_144824-9f43b07d.pth)  |
| ms-pie_stylegan2_c2_config-d |   Tab.5 config-d   | 256, 384, 512  |  3.50  | 73.28/56.16 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/mspie-stylegan2_c2_config-d_ffhq_256-512_b3x8_1100k.py)  | [model](https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c2_config-d_ffhq_256-512_b3x8_1100k_20210406_144840-dbefacf6.pth)  |
| ms-pie_stylegan2_c2_config-e |   Tab.5 config-e   | 256, 384, 512  |  3.15  | 74.13/56.88 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/mspie-stylegan2_c2_config-e_ffhq_256-512_b3x8_1100k.py)  | [model](https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c2_config-e_ffhq_256-512_b3x8_1100k_20210406_144906-98d5a42a.pth)  |
| ms-pie_stylegan2_c2_config-f |   Tab.5 config-f   | 256, 384, 512  |  2.93  | 73.51/57.32 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/mspie-stylegan2_c2_config-f_ffhq_256-512_b3x8_1100k.py)  | [model](https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c2_config-f_ffhq_256-512_b3x8_1100k_20210406_144927-4f4d5391.pth)  |
| ms-pie_stylegan2_c1_config-g |   Tab.5 config-g   | 256, 384, 512  |  3.40  | 73.05/56.45 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/mspie-stylegan2_c1_config-g_ffhq_256-512_b3x8_1100k.py)  | [model](https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c1_config-g_ffhq_256-512_b3x8_1100k_20210406_144758-2df61752.pth)  |
| ms-pie_stylegan2_c2_config-h |   Tab.5 config-h   | 256, 384, 512  |  4.01  | 72.81/54.35 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/mspie-stylegan2_c2_config-h_ffhq_256-512_b3x8_1100k.py)  | [model](https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c2_config-h_ffhq_256-512_b3x8_1100k_20210406_145006-84cf3f48.pth)  |
| ms-pie_stylegan2_c2_config-i |   Tab.5 config-i   | 256, 384, 512  |  3.76  | 73.26/54.71 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/mspie-stylegan2_c2_config-i_ffhq_256-512_b3x8_1100k.py)  | [model](https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c2_config-i_ffhq_256-512_b3x8_1100k_20210406_145023-c2b0accf.pth)  |
| ms-pie_stylegan2_c2_config-j |   Tab.5 config-j   | 256, 384, 512  |  4.23  | 73.11/54.63 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/mspie-stylegan2_c2_config-j_ffhq_256-512_b3x8_1100k.py)  | [model](https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c2_config-j_ffhq_256-512_b3x8_1100k_20210406_145044-c407481b.pth)  |
| ms-pie_stylegan2_c2_config-k |   Tab.5 config-k   | 256, 384, 512  |  4.17  | 73.05/51.07 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/mspie-stylegan2_c2_config-k_ffhq_256-512_b3x8_1100k.py)  | [model](https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c2_config-k_ffhq_256-512_b3x8_1100k_20210406_145105-6d8cc39f.pth)  |
| ms-pie_stylegan2_c2_config-f | higher-resolution  | 256, 512, 896  |  4.10  | 72.21/50.29 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/mspie-stylegan2_c2_config-f_ffhq_256-896_b3x8_1100k.py)  | [model](https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c2_config-f_ffhq_256-896_b3x8_1100k_20210406_144943-6c18ad5d.pth)  |
| ms-pie_stylegan2_c1_config-f | higher-resolution  | 256, 512, 1024 |  6.24  | 71.79/49.92 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/mspie-stylegan2_c1_config-f_ffhq_256-1024_b2x8_1600k.py) | [model](https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c1_config-f_ffhq_256-1024_b2x8_1600k_20210406_144716-81cbdc96.pth) |



Note that we report the FID and P&R metric (FFHQ dataset) in the largest scale.
## Results and Models for SinGAN

<div align="center">
  <b> Positional Encoding in SinGAN</b>
  <br/>
  <img src="https://nbei.github.io/gan-pos-encoding/teaser-web-singan.png" width="800"/>
</div>


|              Model              |                                      Data                                       | Num Scales |                                                                    Config                                                                    |                                                                                                                        Download                                                                                                                         |
| :-----------------------------: | :-----------------------------------------------------------------------------: | :--------: | :------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|         SinGAN + no pad         | [balloons.png](https://download.openmmlab.com/mmgen/dataset/singan/balloons.png) |     8      |      [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/singan_interp-pad_balloons.py)      |           [ckpt](https://download.openmmlab.com/mmgen/pe_in_gans/singan_interp-pad_balloons_20210406_180014-96f51555.pth) &#124; [pkl](https://download.openmmlab.com/mmgen/pe_in_gans/singan_interp-pad_balloons_20210406_180014-96f51555.pkl)           |
| SinGAN + no pad + no bn in disc | [balloons.png](https://download.openmmlab.com/mmgen/dataset/singan/balloons.png) |     8      | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/singan_interp-pad_disc-nobn_balloons.py) | [ckpt](https://download.openmmlab.com/mmgen/pe_in_gans/singan_interp-pad_disc-nobn_balloons_20210406_180059-7d63e65d.pth) &#124; [pkl](https://download.openmmlab.com/mmgen/pe_in_gans/singan_interp-pad_disc-nobn_balloons_20210406_180059-7d63e65d.pkl) |
| SinGAN + no pad + no bn in disc |  [fish.jpg](https://download.openmmlab.com/mmgen/dataset/singan/fish-crop.jpg)   |     10     |   [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/singan_interp-pad_disc-nobn_fish.py)   |      [ckpt](https://download.openmmlab.com/mmgen/pe_in_gans/singan_interp-pad_disc-nobn_fis_20210406_175720-9428517a.pth) &#124; [pkl](https://download.openmmlab.com/mmgen/pe_in_gans/singan_interp-pad_disc-nobn_fis_20210406_175720-9428517a.pkl)      |
|          SinGAN + CSG           |  [fish.jpg](https://download.openmmlab.com/mmgen/dataset/singan/fish-crop.jpg)   |     10     |           [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/singan_csg_fish.py)            |                       [ckpt](https://download.openmmlab.com/mmgen/pe_in_gans/singan_csg_fis_20210406_175532-f0ec7b61.pth) &#124; [pkl](https://download.openmmlab.com/mmgen/pe_in_gans/singan_csg_fis_20210406_175532-f0ec7b61.pkl)                       |
|          SinGAN + CSG           | [bohemian.png](https://download.openmmlab.com/mmgen/dataset/singan/bohemian.png) |     10     |         [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/singan_csg_bohemian.py)          |                  [ckpt](https://download.openmmlab.com/mmgen/pe_in_gans/singan_csg_bohemian_20210407_195455-5ed56db2.pth) &#124; [pkl](https://download.openmmlab.com/mmgen/pe_in_gans/singan_csg_bohemian_20210407_195455-5ed56db2.pkl)                  |
|        SinGAN + SPE-dim4        |  [fish.jpg](https://download.openmmlab.com/mmgen/dataset/singan/fish-crop.jpg)   |     10     |         [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/singan_spe-dim4_fish.py)         |                 [ckpt](https://download.openmmlab.com/mmgen/pe_in_gans/singan_spe-dim4_fish_20210406_175933-f483a7e3.pth) &#124; [pkl](https://download.openmmlab.com/mmgen/pe_in_gans/singan_spe-dim4_fish_20210406_175933-f483a7e3.pkl)                 |
|        SinGAN + SPE-dim4        | [bohemian.png](https://download.openmmlab.com/mmgen/dataset/singan/bohemian.png) |     10     |       [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/singan_spe-dim4_bohemian.py)       |             [ckpt](https://download.openmmlab.com/mmgen/pe_in_gans/singan_spe-dim4_bohemian_20210406_175820-6e484a35.pth) &#124; [pkl](https://download.openmmlab.com/mmgen/pe_in_gans/singan_spe-dim4_bohemian_20210406_175820-6e484a35.pkl)             |
|        SinGAN + SPE-dim8        | [bohemian.png](https://download.openmmlab.com/mmgen/dataset/singan/bohemian.png) |     10     |       [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/positional_encoding_in_gans/singan_spe-dim8_bohemian.py)       |             [ckpt](https://download.openmmlab.com/mmgen/pe_in_gans/singan_spe-dim8_bohemian_20210406_175858-7faa50f3.pth) &#124; [pkl](https://download.openmmlab.com/mmgen/pe_in_gans/singan_spe-dim8_bohemian_20210406_175858-7faa50f3.pkl)             |
