<div align="center">
    <img src="https://user-images.githubusercontent.com/12726765/114528756-de55af80-9c7b-11eb-94d7-d3224ada1585.png" width="400"/>
</div>


## Introduction

MMGeneration is a powerful toolkit for generative models, especially for GANs now. It is based on PyTorch and [MMCV](https://github.com/open-mmlab/mmcv).

<div align="center">
    <img src="https://user-images.githubusercontent.com/12726765/114534478-9a65a900-9c81-11eb-8087-de8b6816eed8.png" width="800"/>
</div>


## Major Features

- **High-quality Training Performance:** We currently support training on Unconditional GANs, Internal GANs, and Image Translation Models. Support for conditional models will come soon.
- **Useful Application Toolkit:** A plentiful toolkit containing multiple applications in GANs are provided to users. GAN interpolation, GAN projection, and GAN manipulations are integrated in our framework. It's time to play with your GANs!
- **Efficient Distributed Training for Generative Models:** For the highly dynamic training in generative models, we adopt a new way to train dynamic models with `MMDDP`.
- **New Modular Design for Flexible Combination:** A new design for complex loss modules is proposed for customizing the links between modules, which can achieve flexible combination among different modules.


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


### `MMGeneration` will be released on **2021/04/20**.

Our new work, **Positional Encoding as Spatial Inductive Bias in GANs accepted by CVPR2021**, will also be released based on `MMGeneration`.
