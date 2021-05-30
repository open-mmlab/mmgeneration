<div align="center">
    <img src="https://user-images.githubusercontent.com/12726765/114528756-de55af80-9c7b-11eb-94d7-d3224ada1585.png" width="400"/>
</div>

Documentation: https://mmgeneration.readthedocs.io/

## Introduction

English | [简体中文](README_zh-CN.md)

MMGeneration is a powerful toolkit for generative models, especially for GANs now. It is based on PyTorch and [MMCV](https://github.com/open-mmlab/mmcv). The master branch works with **PyTorch 1.5+**.

<div align="center">
    <img src="https://user-images.githubusercontent.com/12726765/114534478-9a65a900-9c81-11eb-8087-de8b6816eed8.png" width="800"/>
</div>


## Major Features

- **High-quality Training Performance:** We currently support training on Unconditional GANs, Internal GANs, and Image Translation Models. Support for conditional models will come soon.
- **Powerful Application Toolkit:** A plentiful toolkit containing multiple applications in GANs is provided to users. GAN interpolation, GAN projection, and GAN manipulations are integrated into our framework. It's time to play with your GANs! ([Tutorial for applications](docs/tutorials/applications.md))
- **Efficient Distributed Training for Generative Models:** For the highly dynamic training in generative models, we adopt a new way to train dynamic models with `MMDDP`. ([Tutorial for DDP](docs/tutorials/ddp_train_gans.md))
- **New Modular Design for Flexible Combination:** A new design for complex loss modules is proposed for customizing the links between modules, which can achieve flexible combination among different modules. ([Tutorial for new modular design](docs/tutorials/customize_losses.md))


<table>
<thead>
  <tr>
    <td>
<div align="center">
  <b> Training Visualization</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114509105-b6f4e780-9c67-11eb-8644-110b3cb01314.gif" width="200"/>
</div></td>
    <td>
<div align="center">
  <b> GAN Interpolation</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114679300-9fd4f900-9d3e-11eb-8f37-c36a018c02f7.gif" width="200"/>
</div></td>
    <td>
<div align="center">
  <b> GAN Projector</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114524392-c11ee200-9c77-11eb-8b6d-37bc637f5626.gif" width="200"/>
</div></td>
    <td>
<div align="center">
  <b> GAN Manipulation</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114523716-20302700-9c77-11eb-804e-327ae1ca0c5b.gif" width="200"/>
</div></td>
  </tr>
</thead>
</table>

## Highlight

* **Positional Encoding as Spatial Inductive Bias in GANs (CVPR2021)** has been released in `MMGeneration`.  [\[Config\]](configs/positional_encoding_in_gans/README.md), [\[Project Page\]](https://nbei.github.io/gan-pos-encoding.html)
* Mixed-precision training (FP16) for StyleGAN2 has been supported. Please check [the comparison](configs/styleganv2/README.md) between different implementations.
## Changelog

v0.2.0 was released on 30/05/2021. Please refer to [changelog.md](docs/changelog.md) for details and release history.

## ModelZoo

These methods have been carefully studied and supported in our frameworks:


<details open>
<summary>Unconditional GANs (click to collapse)</summary>

- ✅ [DCGAN](configs/dcgan/README.md) (ICLR'2016)
- ✅ [WGAN-GP](configs/wgan-gp/README.md) (NIPS'2017)
- ✅ [LSGAN](configs/lsgan/README.md) (ICCV'2017)
- ✅ [GGAN](configs/ggan/README.md) (Axiv'2017)
- ✅ [PGGAN](configs/pggan/README.md) (ICLR'2018)
- ✅ [StyleGANV1](configs/styleganv1/README.md) (CVPR'2019)
- ✅ [StyleGANV2](configs/styleganv2/README.md) (CVPR'2020)
- ✅ [Positional Encoding in GANs](configs/positional_encoding_in_gans/README.md) (CVPR'2021)

</details>

<details open>
<summary>Image2Image Translation (click to collapse)</summary>

- ✅ [Pix2Pix](configs/pix2pix/README.md) (CVPR'2017)
- ✅ [CycleGAN](configs/cyclegan/README.md) (ICCV'2017)

</details>

<details open>
<summary>Internal Learning (click to collapse)</summary>

- ✅ [SinGAN](configs/singan/README.md) (ICCV'2019)

</details>


## License

This project is released under the [Apache 2.0 license](LICENSE). Some operations in `MMGeneration` are with other licenses instead of Apache2.0. Please refer to [LICENSES.md](LICENSES.md) for the careful check, if you are using our code for commercial matters.

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of MMGeneration. [docs/quick_run.md](docs/quick_run.md) can offer full guidance for quick run. For other details and tutorials, please go to our [documentation](https://mmgeneration.readthedocs.io/).

## Contributing

We appreciate all contributions to improve MMGeneration. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) in MMCV for more details about the contributing guideline.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2021mmgeneration,
    title={{MMGeneration}: OpenMMLab Generative Model Toolbox and Benchmark},
    author={MMGeneration Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmgeneration}},
    year={2021}
}
```

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab's next-generation toolbox for generative models.
