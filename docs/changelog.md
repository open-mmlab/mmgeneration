# Changelog

## v0.1.0 (20/04/2021)

**Highlights**

- MMGeneration is released.

**Main Features**

- High-quality Training Performance: We currently support training on Unconditional GANs(`DCGAN`, `WGAN-GP`,`PGGAN`,  `StyleGANV1`, `StyleGANV2`, `Positional Encoding in GANs`), Internal GANs(`SinGAN`), and Image Translation Models(`Pix2Pix`, `CycleGAN`). Support for conditional models will come soon.
- Powerful Application Toolkit: A plentiful toolkit containing multiple applications in GANs is provided to users. GAN interpolation, GAN projection, and GAN manipulations are integrated into our framework. It's time to play with your GANs!
- Efficient Distributed Training for Generative Models: For the highly dynamic training in generative models, we adopt a new way to train dynamic models with `MMDDP`.
- New Modular Design for Flexible Combination: A new design for complex loss modules is proposed for customizing the links between modules, which can achieve flexible combination among different modules.
