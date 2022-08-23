# Inference

Currently, we support various popular generative models, including unconditional GANs, conditional GANs, image translation models, internal GANs and diffusion models.
In this section, we will specify how to sample fake images by using our existing models. For model inference, all of the APIs have been included in [mmgen/apis/inference.py](https://github.com/open-mmlab/mmgeneration/tree/1.x/mmgen/apis/inference.py). The most important function is `init_model` for creating a generative model from a config. Then, adopting the sampling function in this file with the generative model will offer you the synthesized images.

## Sample images with unconditional GANs

MMGeneration provides high-level APIs for sampling images with unconditional GANs. Here is an example for building StyleGAN2-256 and obtaining the synthesized images.

```python
import mmcv
from mmgen.apis import init_model, sample_unconditional_model

# Specify the path to model config and checkpoint file
config_file = 'configs/styleganv2/stylegan2_c2_8xb4_ffhq-1024x1024.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth'

device = 'cuda:0'
# init a generatvie
model = init_model(config_file, checkpoint_file, device=device)
# sample images
fake_imgs = sample_unconditional_model(model, 4)
```

Indeed, we have already provided a more friendly demo script to users. You can use [demo/unconditional_demo.py](https://github.com/open-mmlab/mmgeneration/tree/1.x/mmgen/demo/unconditional_demo.py) with the following commands:

```shell
python demo/unconditional_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    [--save-path ${SAVE_PATH}] \
    [--device ${GPU_ID}]
```

Note that more arguments are also offered to customizing your sampling procedure. Please use `python demo/unconditional_demo.py --help` to check more details.

## Sample images with conditional GANs

MMGeneration provides high-level APIs for sampling images with conditional GANs. Here is an example for building SAGAN-128 and obtaining the synthesized images.

```python
import mmcv
from mmgen.apis import init_model, sample_conditional_model

# Specify the path to model config and checkpoint file
config_file = 'configs/sagan/sagan_woReLUinplace-Glr1e-4_Dlr4e-4_noaug-ndisc1-8xb32-bigGAN-sch_imagenet1k-128x128.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmgen/sagan/sagan_128_woReLUinplace_noaug_bigGAN_imagenet1k_b32x8_Glr1e-4_Dlr-4e-4_ndisc1_20210818_210232-3f5686af.pth'

device = 'cuda:0'
# init a generatvie
model = init_model(config_file, checkpoint_file, device=device)
# sample images with random label
fake_imgs = sample_conditional_model(model, 4)

# sample images with the same label
fake_imgs = sample_conditional_model(model, 4, label=0)

# sample images with specific labels
fake_imgs = sample_conditional_model(model, 4, label=[0, 1, 2, 3])
```

Indeed, we have already provided a more friendly demo script to users. You can use [demo/conditional_demo.py](https://github.com/open-mmlab/mmgeneration/tree/1.x/mmgen/demo/conditional_demo.py) with the following commands:

```shell
python demo/conditional_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    [--label] ${LABEL} \
    [--samples-per-classes] ${SAMPLES_PER_CLASSES} \
    [--sample-all-classes] \
    [--save-path ${SAVE_PATH}] \
    [--device ${GPU_ID}]
```

If `--label` is not passed, images with random labels would be generated.
If `--label` is passed, we would generate `${SAMPLES_PER_CLASSES}` images for each input label.
If `sample_all_classes` is set true in command line, `--label` would be ignored and the generator will output images for all categories.

Note that more arguments are also offered to customizing your sampling procedure. Please use `python demo/conditional_demo.py --help` to check more details.

## Sample images with image translation models

MMGeneration provides high-level APIs for translating images by using image translation models. Here is an example of building Pix2Pix and obtaining the translated images.

```python
import mmcv

from mmgen.apis import init_model, sample_img2img_model

# Specify the path to model config and checkpoint file
config_file = 'configs/pix2pix/pix2pix_vanilla-unet-bn_wo-jitter-flip-4xb1-190kiters_edges2shoes.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmgen/pix2pix/refactor/pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes_convert-bgr_20210902_170902-0c828552.pth'
# Specify the path to image you want to translate
image_path = 'tests/data/paired/test/33_AB.jpg'
device = 'cuda:0'
# init a generatvie
model = init_model(config_file, checkpoint_file, device=device)
# translate a single image
translated_image = sample_img2img_model(model, image_path, target_domain='photo')
```

Indeed, we have already provided a more friendly demo script to users. You can use [demo/translation_demo.py](https://github.com/open-mmlab/mmgeneration/tree/1.x/mmgen/demo/translation_demo.py) with the following commands:

```shell
python demo/translation_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    ${IMAGE_PATH}
    [--save-path ${SAVE_PATH}] \
    [--device ${GPU_ID}]
```

Note that more customized arguments are also offered to customizing your sampling procedure. Please use `python demo/translation_demo.py --help` to check more details.

## Sample images with diffusion models

MMGeneration provides high-level APIs for sampling images with diffusion models. Here is an example for building I-DDPM and obtaining the synthesized images.

```python
import mmcv

from mmgen.apis import init_model, sample_ddpm_model

# Specify the path to model config and checkpoint file
config_file = 'configs/improved_ddpm/ddpm_cosine-hybird-timestep-4k_16xb8-1500kiters_imagenet1k-64x64.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmgen/improved_ddpm/ddpm_cosine_hybird_timestep-4k_imagenet1k_64x64_b8x16_1500k_20220103_223919-b8f1a310.pth'
device = 'cuda:0'
# init a generatvie
model = init_model(config_file, checkpoint_file, device=device)
# sample images
fake_imgs = sample_ddpm_model(model, 4)
```

Indeed, we have already provided a more friendly demo script to users. You can use [demo/ddpm_demo.py](https://github.com/open-mmlab/mmgeneration/tree/1.x/mmgen/demo/ddpm_demo.py) with the following commands:

```shell
python demo/ddpm_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    [--save-path ${SAVE_PATH}] \
    [--device ${GPU_ID}]
```

Note that more arguments are also offered to customizing your sampling procedure. Please use `python demo/ddpm_demo.py --help` to check more details.
