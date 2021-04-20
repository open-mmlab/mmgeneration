<div align="center">
    <img src="https://user-images.githubusercontent.com/12726765/114528756-de55af80-9c7b-11eb-94d7-d3224ada1585.png" width="400"/>
</div>

文档: https://mmgeneration.readthedocs.io/

## 简介

[English](README.md) | 简体中文

MMGeneration 是一个基于 PyTorch 和[MMCV](https://github.com/open-mmlab/mmcv)的强有力的生成模型工具箱，尤其专注于 GAN 模型。
主分支目前支持 **PyTorch 1.5** 以上的版本。

<div align="center">
    <img src="https://user-images.githubusercontent.com/12726765/114534478-9a65a900-9c81-11eb-8087-de8b6816eed8.png" width="800"/>
</div>


## 主要特性

- **高质量高性能的训练:** 我们目前支持 Unconditional GANs, Internal GANs, 以及 Image Translation Models 的训练。很快将会支持 conditional models 的训练。
- **强有力的应用工具箱:** 为用户提供了丰富的工具箱，包含 GANs 中的多种应用。我们的框架集成了 GANs 的插值，投影和编辑。请用你的 GANs 尽情尝试！([应用教程](docs/tutorials/applications.md))
- **生成模型的高效分布式训练:** 对于生成模型中的高度动态训练，我们采用 `MMDDP` 的新方法来训练动态模型。([DDP教程](docs/tutorials/ddp_train_gans.md))
- **灵活组合的新型模块化设计:** 针对复杂的损失模块，我们提出了一种新的设计，可以自定义模块之间的链接，实现不同模块之间的灵活组合。 ([新模块化设计教程](docs/tutorials/customize_losses.md))


<table>
<thead>
  <tr>
    <td>
<div align="center">
  <b> 训练可视化</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114509105-b6f4e780-9c67-11eb-8644-110b3cb01314.gif" width="200"/>
</div></td>
    <td>
<div align="center">
  <b> GAN 插值</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114679300-9fd4f900-9d3e-11eb-8f37-c36a018c02f7.gif" width="200"/>
</div></td>
    <td>
<div align="center">
  <b> GAN 投影</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114524392-c11ee200-9c77-11eb-8b6d-37bc637f5626.gif" width="200"/>
</div></td>
    <td>
<div align="center">
  <b> GAN 编辑</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114523716-20302700-9c77-11eb-804e-327ae1ca0c5b.gif" width="200"/>
</div></td>
  </tr>
</thead>
</table>

## 亮点

* **Positional Encoding as Spatial Inductive Bias in GANs (CVPR2021)** 已在 `MMGeneration` 中发布.  [\[配置文件\]](configs/positional_encoding_in_gans/README.md), [\[项目主页\]](https://nbei.github.io/gan-pos-encoding.html)
## 更新日志

v0.1.0 在 20/04/2021 发布。 关于细节和发布历史，请参考 [changelog.md](docs/changelog.md)。

## 模型库

这些算法在我们的框架中得到了认真研究和支持。


<details open>
<summary>Unconditional GANs (点击折叠)</summary>

- ✅ [DCGAN](configs/dcgan/README.md) (ICLR'2016)
- ✅ [WGAN-GP](configs/wgan-gp/README.md) (NIPS'2017)
- ✅ [PGGAN](configs/pggan/README.md) (ICLR'2018)
- ✅ [StyleGANV1](configs/styleganv1/README.md) (CVPR'2019)
- ✅ [StyleGANV2](configs/styleganv2/README.md) (CVPR'2020)
- ✅ [Positional Encoding in GANs](configs/positional_encoding_in_gans/README.md) (CVPR'2021)

</details>

<details open>
<summary>Image2Image Translation (点击折叠)</summary>

- ✅ [Pix2Pix](configs/pix2pix/README.md) (CVPR'2017)
- ✅ [CycleGAN](configs/cyclegan/README.md) (ICCV'2017)

</details>

<details open>
<summary>Internal Learing (点击折叠)</summary>

- ✅ [SinGAN](configs/dcgan/README.md) (ICCV'2019)

</details>


## 开源许可证

该项目采用 [Apache 2.0 license](LICENSE) 开源许可证。`MMGeneration` 中的一些操作使用了其他许可证。如果您使用我们的代码进行商业事务，请参考 [许可证](LICENSES.md) 并仔细检查。

## 安装

请参考[快速入门](docs/get_started.md)进行安装。

## 快速入门

对于 `MMGeneration` 的基本使用请参考 [快速入门](docs/get_started.md)。其他细节和教程，请参考我们的[文档](https://mmgeneration.readthedocs.io/)。

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMGeneration 所作出的努力。请参考[贡献指南](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 引用

如果您发现此项目对您的研究有用，请考虑引用：

```BibTeX
@misc{2021mmgeneration,
    title={{MMGeneration}: OpenMMLab Generative Model Toolbox and Benchmark},
    author={MMGeneration Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmgeneration}},
    year={2020}
}
```

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱与测试基准
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱与测试基准
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用3D目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包.
