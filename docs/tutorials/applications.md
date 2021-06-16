# Tutorial 7: Applications with Generative Models

## Interpolation
The generative model in the GAN architecture learns to map points in the latent space to generated images. The latent space has no meaning other than the meaning applied to it via the generative model. Generally, we want to explore the structure of latent space, one thing we can do is to interpolate a sequence of points between two endpoints in the latent space, and see the results these points yield. (Eg. we believe that features that are absent in either endpoint appear in the middle of a linear interpolation path is a sign that the latent space is entangled and the factors of variation are not properly separated.)

Indeed, we have provided a application script to users. You can use [apps/interpolate_sample.py](https://github.com/open-mmlab/mmgeneration/tree/master/apps/interpolate_sample.py) with the following commands:

```bash
python apps/interpolate_sample.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    [--show-mode ${SHOW_MODE}] \
    [--endpoint ${ENDPOINT}] \
    [--interval ${INTERVAL}] \
    [--space ${SPACE}] \
    [--samples-path ${SAMPLES_PATH}] \
    [--batch-size ${BATCH_SIZE}] \
```
Here, we provide two kinds of `show-mode`, `sequence`, and `group`. In `sequence` mode, we sample a sequence of endpoints first, then interpolate points between two endpoints in order, generated images will be saved individually. In `group` mode, we sample several pairs of endpoints, then interpolate points between two endpoints in a pair, generated images will be saved in a single picture. What's more, `space` refers to the latent code space, you can choose 'z' or 'w' (especially refer to style space in StyleGAN series), `endpoint` indicates the number of endpoints you want to sample (should be set to even number in `group` mode), `interval` means the number of points (include endpoints) you interpolate between two endpoints.

Note that more customized arguments are also offered to customizing your interpolating procedure.
Please use `python apps/interpolate_sample.py --help` to check more details.

## Projection
Inverting the synthesis network g is an interesting problem that has many applications. For example, manipulating a given image in the latent feature space requires finding a matching latent code for it first. Generally, you can reconstruct a target image by optimizing over the latent vector, using lpips and pixel-wise loss as the objective function.

Indeed, we have provided an application script to users to find the matching latent vector w of StyleGAN series synthesis network for given images. You can use [apps/stylegan_projector.py](https://github.com/open-mmlab/mmgeneration/tree/master/apps/stylegan_projector.py) with the following commands:

```bash
python apps/stylegan_projector.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    ${FILES}
    [--results-path ${RESULTS_PATH}]
```
Here, `FILES` refer to the images' path, and the projection latent and reconstructed images will be saved in `results-path`.
Note that more customized arguments are also offered to customizing your projection procedure.
Please use `python apps/stylegan_projector.py --help` to check more details.

## Manipulation
A general application of StyleGAN based models is manipulating the latent space to control the attributes of the synthesized images. Here, we provide a simple but popular algorithm based on [SeFa](https://arxiv.org/pdf/2007.06600.pdf) to users. Of course, we modify the original version in calculating eigenvectors and offer a more flexible interface.

To manipulate your generator, you can run the script [apps/modified_sefa.py](https://github.com/open-mmlab/mmgeneration/tree/master/apps/modified_sefa.py) with the following command:

```shell
python apps/modified_sefa.py --cfg ${CONFIG} --ckpt ${CKPT} \
    -i ${INDEX} -d ${DEGREE} --degree-step ${D_STEP} \
    -l ${LAYER_NO} \
    [--eigen-vector ${PATH_EIGEN_VEC}]
```

In this script, the eigenvector for the generator parameter will be calculated if `eigen-vector` is None. Meanwhile, we will save it in the same directory of the `ckpt` file, so that users can apply this pre-calculated vector. The demo of `Positional Encoding as Spatial Inductive Bias in GANs` just comes from this script. Here is an example for users to get similar results with our demo.

The `${INDEX}` indicates which eigenvector we will apply to manipulate the images. In general cases, each index controls one independent attribute, which is guaranteed by the disentangled representation in StyleGAN. We suggest that users can try different indexes to find the one you want. The `--degree` sets the range of the multiplication factor. In our experiments, we observe that an unsymmetric range like `[-3, 8]` is very helpful. Thus, we allow for setting the lower and higher bound in this argument. `--layer` or `-l` defines which layer we will apply the eigenvector. Some properties, like lighting, are only related to 1-2 layers in the generator.

Taking the lighting attribute as an example, we adopt the following command on our MS-PIE-StyleGAN2-256 model:

```shell
python apps/modified_sefa.py \
    --config configs/positional_encoding_in_gans/mspie-stylegan2_c2_config-f_ffhq_256-512_b3x8_1100k.py \
    --ckpt https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c2_config-f_ffhq_256-512_b3x8_1100k_20210406_144927-4f4d5391.pth \
    -i 15 -d 8. --degree-step 0.5 -l 8 9 --sample-path ./work_dirs/sefa-exp/ \
    --sample-cfg chosen_scale=4 randomize_noise=False
```

Importantly, after setting `chosen_scale=4`, we can manipulate the 512x512 images with a simple 256-scale generator.
