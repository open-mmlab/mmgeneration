<div align="center">
    <img src="https://user-images.githubusercontent.com/12726765/114528756-de55af80-9c7b-11eb-94d7-d3224ada1585.png" width="400"/>
      <div>&nbsp;</div>
   <div align="center">
     <b><font size="5">OpenMMLab website</font></b>
     <sup>
       <a href="https://openmmlab.com">
         <i><font size="4">HOT</font></i>
       </a>
     </sup>
     &nbsp;&nbsp;&nbsp;&nbsp;
     <b><font size="5">OpenMMLab platform</font></b>
     <sup>
       <a href="https://platform.openmmlab.com">
         <i><font size="4">TRY IT OUT</font></i>
       </a>
     </sup>
   </div>
   <div>&nbsp;</div>
</div>

[![PyPI](https://img.shields.io/pypi/v/mmgen)](https://pypi.org/project/mmgen)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmgeneration.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmgeneration/workflows/build/badge.svg)](https://github.com/open-mmlab/mmgeneration/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmgeneration/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmgeneration)
[![license](https://img.shields.io/github/license/open-mmlab/mmgeneration.svg)](https://github.com/open-mmlab/mmgeneration/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmgeneration.svg)](https://github.com/open-mmlab/mmgeneration/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmgeneration.svg)](https://github.com/open-mmlab/mmgeneration/issues)

[📘Documentation](https://mmgeneration.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmgeneration.readthedocs.io/en/latest/get_started.html#installation) |
[👀Model Zoo](https://mmgeneration.readthedocs.io/en/latest/modelzoo_statistics.html) |
[🆕Update News](https://github.com/open-mmlab/mmgeneration/blob/master/docs/en/changelog.md) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmgeneration/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmgeneration/issues)

English | [简体中文](README_zh-CN.md)

## What's New

MMGeneration has been merged in [MMEditing](https://github.com/open-mmlab/mmediting/tree/1.x). And we have supported new generation tasks and models. We highlight the following new features:

- 🌟 Text2Image

  - ✅ [GLIDE](https://github.com/open-mmlab/mmediting/tree/1.x/projects/glide/configs/README.md)
  - ✅ [Disco-Diffusion](https://github.com/open-mmlab/mmediting/tree/1.x/configs/disco_diffusion/README.md)
  - ✅ [Stable-Diffusion](https://github.com/open-mmlab/mmediting/tree/1.x/configs/stable_diffusion/README.md)

- 🌟 3D-aware Generation

  - ✅ [EG3D](https://github.com/open-mmlab/mmediting/tree/1.x/configs/eg3d/README.md)

## Introduction

MMGeneration is a powerful toolkit for generative models, especially for GANs now. It is based on PyTorch and [MMCV](https://github.com/open-mmlab/mmcv). The master branch works with **PyTorch 1.5+**.

<div align="center">
    <img src="https://user-images.githubusercontent.com/12726765/114534478-9a65a900-9c81-11eb-8087-de8b6816eed8.png" width="800"/>
</div>

## Major Features

- **High-quality Training Performance:** We currently support training on Unconditional GANs, Internal GANs, and Image Translation Models. Support for conditional models will come soon.
- **Powerful Application Toolkit:** A plentiful toolkit containing multiple applications in GANs is provided to users. GAN interpolation, GAN projection, and GAN manipulations are integrated into our framework. It's time to play with your GANs! ([Tutorial for applications](docs/en/tutorials/applications.md))
- **Efficient Distributed Training for Generative Models:** For the highly dynamic training in generative models, we adopt a new way to train dynamic models with `MMDDP`. ([Tutorial for DDP](docs/en/tutorials/ddp_train_gans.md))
- **New Modular Design for Flexible Combination:** A new design for complex loss modules is proposed for customizing the links between modules, which can achieve flexible combination among different modules. ([Tutorial for new modular design](docs/en/tutorials/customize_losses.md))

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

- **Positional Encoding as Spatial Inductive Bias in GANs (CVPR2021)** has been released in `MMGeneration`.  [\[Config\]](configs/positional_encoding_in_gans/README.md), [\[Project Page\]](https://nbei.github.io/gan-pos-encoding.html)
- Conditional GANs have been supported in our toolkit. More methods and pre-trained weights will come soon.
- Mixed-precision training (FP16) for StyleGAN2 has been supported. Please check [the comparison](configs/styleganv2/README.md) between different implementations.

## Changelog

v0.7.3 was released on 14/04/2023. Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

## Installation

MMGeneration depends on [PyTorch](https://pytorch.org/) and [MMCV](https://github.com/open-mmlab/mmcv).
Below are quick steps for installation.

**Step 1.**
Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

```python
pip3 install torch torchvision

```

**Step 2.**
Install MMCV with [MIM](https://github.com/open-mmlab/mim).

```
pip3 install openmim
mim install mmcv-full
```

**Step 3.**
Install MMGeneration from source.

```
git clone https://github.com/open-mmlab/mmgeneration.git
cd mmgeneration
pip3 install -e .
```

Please refer to [get_started.md](docs/en/get_started.md) for more detailed instruction.

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMGeneration. [docs/en/quick_run.md](docs/en/quick_run.md) can offer full guidance for quick run. For other details and tutorials, please go to our [documentation](https://mmgeneration.readthedocs.io/).

## ModelZoo

These methods have been carefully studied and supported in our frameworks:

<details open>
<summary>Unconditional GANs (click to collapse)</summary>

- ✅ [DCGAN](configs/dcgan/README.md) (ICLR'2016)
- ✅ [WGAN-GP](configs/wgan-gp/README.md) (NIPS'2017)
- ✅ [LSGAN](configs/lsgan/README.md) (ICCV'2017)
- ✅ [GGAN](configs/ggan/README.md) (arXiv'2017)
- ✅ [PGGAN](configs/pggan/README.md) (ICLR'2018)
- ✅ [StyleGANV1](configs/styleganv1/README.md) (CVPR'2019)
- ✅ [StyleGANV2](configs/styleganv2/README.md) (CVPR'2020)
- ✅ [StyleGANV3](configs/styleganv3/README.md) (NeurIPS'2021)
- ✅ [Positional Encoding in GANs](configs/positional_encoding_in_gans/README.md) (CVPR'2021)

</details>

<details open>
<summary>Conditional GANs (click to collapse)</summary>

- ✅ [SNGAN](configs/sngan_proj/README.md) (ICLR'2018)
- ✅ [Projection GAN](configs/sngan_proj/README.md) (ICLR'2018)
- ✅ [SAGAN](configs/sagan/README.md) (ICML'2019)
- ✅ [BIGGAN/BIGGAN-DEEP](configs/biggan/README.md) (ICLR'2019)

</details>

<details open>
<summary>Tricks for GANs (click to collapse)</summary>

- ✅ [ADA](configs/ada/README.md) (NeurIPS'2020)

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

<details open>
<summary>Denoising Diffusion Probabilistic Models (click to collapse)</summary>

- ✅ [Improved DDPM](configs/improved_ddpm/README.md) (arXiv'2021)

</details>

## Related-Applications

- ✅ [MMGEN-FaceStylor](https://github.com/open-mmlab/MMGEN-FaceStylor)

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

## License

This project is released under the [Apache 2.0 license](LICENSE). Some operations in `MMGeneration` are with other licenses instead of Apache2.0. Please refer to [LICENSES.md](LICENSES.md) for the careful check, if you are using our code for commercial matters.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
