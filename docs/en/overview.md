# Overview

MMGeneration is a powerful toolkit for generative models, including GANs and diffusion models. It is based on PyTorch and [MMCV](https://github.com/open-mmlab/mmcv). The master branch works with **PyTorch 1.5+**.

## supported tasks

Now, we support 4 tasks of image generation. Lists as follows.

- Unconditional GANs
- Conditional GANs
- Image2Image Translation
- Internal Learning
- Denoising Diffusion Probabilistic Models

## highlight

- **High-quality Training Performance:** We currently support training on Unconditional GANs, Internal GANs, and Image Translation Models. Support for conditional models will come soon.
- **Powerful Application Toolkit:** A plentiful toolkit containing multiple applications in GANs is provided to users. GAN interpolation, GAN projection, and GAN manipulations are integrated into our framework. It's time to play with your GANs! ([Tutorial for applications](docs/en/tutorials/applications.md))

## get started

To get started with our repo, please refer to [get_started.md](docs/en/get_started.md).

## user guides

For elementary guides on basic usage, please refer to [user_guides](docs/en/user_guides).

## advanced guides

To learn design and structure of our framework, as well as how to extend the repo, how to use multiple repos and other advanced usages, please refer to [advanced_guides](docs/en/advanced_guides).
