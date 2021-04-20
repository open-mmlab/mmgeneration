# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

## Introduction
[ALGORITHM]

```latex
@article{radford2015unsupervised,
  title={Unsupervised representation learning with deep convolutional generative adversarial networks},
  author={Radford, Alec and Metz, Luke and Chintala, Soumith},
  journal={arXiv preprint arXiv:1511.06434},
  year={2015}
}
```

## Results and models

<div align="center">
  <b> DCGAN 64x64, CelebA-Cropped</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/113991928-871f9b80-9885-11eb-920e-d389c603fed8.png" width="800"/>
</div>

|   Models    |    Dataset     |           SWD            | MS-SSIM |                                                        Config                                                         |                                                                                                                       Download                                                                                                                        |
| :---------: | :------------: | :----------------------: | :-----: | :-------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DCGAN 64x64 | CelebA-Cropped |  8.93,10.53,50.32/23.26  | 0.2899  | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/dcgan/dcgan_celeba-cropped_64_b128x1_300k.py) | [model](http://download.openmmlab.com/mmgen/dcgan/dcgan_celeba-cropped_64_b128x1_300kiter_20210408_161607-1f8a2277.pth) &#124; [log](http://download.openmmlab.com/mmgen/dcgan/dcgan_celeba-cropped_64_b128x1_300kiter_20210408_161607-1f8a2277.json) |
| DCGAN 64x64 |  LSUN-Bedroom  | 42.79, 34.55, 98.46/58.6 | 0.2095  | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/dcgan/dcgan_lsun-bedroom_64x64_b128x1_5e.py)  |         [model](http://download.openmmlab.com/mmgen/dcgan/dcgan_lsun-bedroom_64_b128x1_5e_20210408_161713-117c498b.pth) &#124; [log](http://download.openmmlab.com/mmgen/dcgan/dcgan_lsun-bedroom_64_b128x1_5e_20210408_161713-117c498b.json)         |
