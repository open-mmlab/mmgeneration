# Overview

MMGeneration is a powerful toolkit for generative models, including GANs and diffusion models. It is based on PyTorch and [MMCV](https://github.com/open-mmlab/mmcv). The [dev-1.x](https://github.com/open-mmlab/mmgeneration/tree/dev-1.x) branch works with [**PyTorch 1.5+**](https://pytorch.org/).

## supported tasks

Now, MMGeneration support 4 tasks of image generation. Lists are as follows.

- Unconditional GANs
- Conditional GANs
- Image2Image Translation
- Internal Learning
- Diffusion Models

<div align="center">
  <b> StyleGAN3 Images</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/150450502-c182834f-796f-4397-bd38-df1efe4a8a47.png" width="800"/>
</div>

<div align="center">
  <b> BigGAN Images </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/127615534-6278ce1b-5cff-4189-83c6-9ecc8de08dfc.png" width="800"/>
</div>

## highlight

- **High-quality Training Performance:** MMGeneration currently support training on Unconditional GANs, Conditional GANs, Internal GANs, Image Translation Models, and diffusion models.
- **Powerful Application Toolkit:** A toolkit that provides plentiful applications to users. MMGeneration supports GAN interpolation, GAN projection, GAN manipulations and many other popular GAN's applications. It's time to play with your GANs! ([Tutorial for applications](advanced_guides/applications.md))
- **Efficient Distributed Training for Generative Models:** With support of [MMSeparateDistributedDataParallel](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/wrappers/seperate_distributed.py), distributed training for dynamic architectures can be easily implemented.
- **New Modular Design for Flexible Combination:** A new design for complex loss modules is proposed for customizing the links between modules, which can achieve flexible combination among different modules.(Tutorial for [losses](advanced_guides/losses.md))

## get started

To get started with our repo, please refer to [get_started.md](get_started.md).

## user guides

For elementary guides on basic usage, please refer to [user_guides](user_guides).

## advanced guides

To learn design and structure of MMGeneration, as well as how to extend the repo, how to use multiple repos and other advanced usages, please refer to [advanced_guides](advanced_guides).
