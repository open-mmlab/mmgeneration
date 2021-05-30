# Changelog

## v0.1.0 (20/04/2021)

**Highlights**

- MMGeneration is released.

**Main Features**

- High-quality Training Performance: We currently support training on Unconditional GANs (`DCGAN`, `WGAN-GP`, `PGGAN`, `StyleGANV1`, `StyleGANV2`, `Positional Encoding in GANs`), Internal GANs (`SinGAN`), and Image Translation Models (`Pix2Pix`, `CycleGAN`). Support for conditional models will come soon.
- Powerful Application Toolkit: A plentiful toolkit containing multiple applications in GANs is provided to users. GAN interpolation, GAN projection, and GAN manipulations are integrated into our framework. It's time to play with your GANs!
- Efficient Distributed Training for Generative Models: For the highly dynamic training in generative models, we adopt a new way to train dynamic models with `MMDDP`.
- New Modular Design for Flexible Combination: A new design for complex loss modules is proposed for customizing the links between modules, which can achieve flexible combination among different modules.


## v0.2.0 (30/05/2021)

#### Highlights
- Support new methods: LSGAN, GGAN.
- Support mixed-precision training (FP16): official PyTorch Implementation and APEX (#11, #20)

#### New Features

- Add the experiment of MNIST in DCGAN (#24)
- Add support for uploading checkpoints to `Ceph` system (cloud server) (#27)
- Add the functionality of saving the best checkpoint in GenerativeEvalHook (#21)

#### Fix bugs and Improvements

- Fix loss of sample-cfg argument (#13)
- Add `pbar` to offline eval and fix bug in grayscale image evaluation/saving (#23)
- Fix error when data_root option in val_cfg or test_cfg are set as None (#28)
- Change latex in quick_run.md to svg url and fix number of checkpoints in modelzoo_statistics.md (#34)
