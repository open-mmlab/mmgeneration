# 1: Inference and train with existing models and standard datasets

Currently, we support various popular generative models, including unconditional GANs, image translation models, and internal GANs. Meanwhile, our framework has been tested on multiple standard datasets, e.g., FFHQ, CelebA, and LSUN. This note will show how to perform common tasks on these existing models and standard datasets, including:

- Use existing models to generate random samples
- Test existing models on standard datasets.
- Train predefined models on standard datasets.

## Generate samples with existing models

In this section, we will specify how to sample fake images by using our unconditional GANs and image translation models. For model inference, all of the APIs have been included in [mmgen/apis/inference.py](https://github.com/open-mmlab/mmgeneration/tree/master/mmgen/apis/inference.py). The most important function is `init_model` for creating a generative model from a config. Then, adopting the sampling function in this file with the generative model will offer you the synthesized images.

### Sample images with unconditional GANs

MMGeneration provides high-level APIs for sampling images by using unconditional GANs. Here is an example of building StyleGAN2-256 and obtaining the synthesized images.

```python
import mmcv
from mmgen.apis import init_model, sample_uncoditional_model

# Specify the path to model config and checkpoint file
config_file = 'configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth'

device = 'cuda:0'
# init a generatvie
model = init_model(config_file, checkpoint_file, device=device)
# sample images
fake_imgs = sample_uncoditional_model(model, 4)
```

Indeed, we have already provided a more friendly demo script to users. You can use [demo/unconditional_demo.py](https://github.com/open-mmlab/mmgeneration/tree/master/mmgen/demo/unconditional_demo.py) with the following commands:

```bash
python demo/unconditional_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    [--save-path ${SAVE_PATH}] \
    [--device ${GPU_ID}]
```
Note that more customized arguments are also offered to customizing your sampling procedure. Please use `python demo/unconditional_demo.py --help` to check more details.

### Sample images with image translation models
MMGeneration provides high-level APIs for translating images by using image translation models. Here is an example of building Pix2Pix and obtaining the translated images.

```python
import mmcv

from mmgen.apis import init_model, sample_img2img_model

# Specify the path to model config and checkpoint file
config_file = 'configs/pix2pix/pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmgen/pix2pix/pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes_convert-bgr_20210410_174116-aaaa3687.pth'
# Specify the path to image you want to transalte
image_path = 'tests/data/paired/test/33_AB.jpg'
device = 'cuda:0'
# init a generatvie
model = init_model(config_file, checkpoint_file, device=device)
# translate a single image
translated_image = sample_img2img_model(model, image_path)
```

Indeed, we have already provided a more friendly demo script to users. You can use [demo/translation_demo.py](https://github.com/open-mmlab/mmgeneration/tree/master/mmgen/demo/translation_demo.py) with the following commands:

```bash
python demo/translation_demo.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    ${IMAGE_PATH}
    [--save-path ${SAVE_PATH}] \
    [--device ${GPU_ID}]
```
Note that more customized arguments are also offered to customizing your sampling procedure. Please use `python demo/translation_demo.py --help` to check more details.

# 2: Prepare dataset for training and testing

This section details how to prepare the dataset for MMGeneration and provides a standard way which we have used in our default configs. We recommend that all of the users may follow the following steps to organize their datasets.

### Datasets for unconditional models

It's much easier to prepare dataset for unconditional models. Firstly, please make a directory, named `data`, in the MMGeneration project. After that, all of datasets can be used by adopting the technology of symlink (soft link).

```bash
mkdir data

ln -s absolute_path_to_dataset ./data/dataset_name
```

Since unconditional models only need real images for training and testing, all you need to do is link your dataset to the `data` directory. Our dataset will automatically check all of the images in a specified path (recursively).

Here, we provide several download links of datasets frequently used in unconditional models: [LSUN](http://dl.yf.io/lsun/), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [CelebA-HQ](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P), [FFHQ](https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP).

### Datasets for image translation models

For translation models, now we offer two settings for datasets called paired image dataset and unpaired image dataset.

For paired image dataset, every image is formed by concatenating two corresponding images from two domains along the width dimension. You are supposed to make two folders "train" and "test" filled with images of this format for training and testing. Folder structure is presented below.
```
./data/dataset_name/
├── test
│   └── XXX.jpg
└── train
    └── XXX.jpg

```

For unpaired image dataset, you are supposed to make two folders "trainA" and "testA" filled with images from domain A and two folders "trainB" and "testB" filled with images from domain B. Folder structure is presented below.

```
./data/dataset_name/
├── testA
│   └── XXX.jpg
├── testB
│   └── XXX.jpg
├── trainA
│   └── XXX.jpg
└── trainB
    └── XXX.jpg

```

Please read the section `Datasets for unconditional models` and also use the symlink (soft link) to build up the dataset.

Here, we provide download links of datasets used in [Pix2Pix](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/) and [CycleGAN](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).

# 3: Train existing models

Currently, we have tested all of the model on distributed training. Thus, we highly recommend to adopt distributed training with our scripts. The basic usage is as follows:

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS_NUMBER} \
    --work-dir ./work_dirs/experiments/experiments_name \
    [optional arguments]
```

If you are using slurm system, the following commands can help you start training"

```shell
bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${WORK_DIR} \
    [optional arguments]
```

There two scripts wrap [tools/train.py](https://github.com/open-mmlab/mmgeneration/tree/master/tools/train.py) with distributed training entrypoint. The `optional arguments` are defined in [tools/train.py](https://github.com/open-mmlab/mmgeneration/tree/master/tools/train.py). Users can also set `random-seed` and `resume-from` with these arguments.

Note that the name of `work_dirs` has already been put into our `.gitignore` file. Users can put any files here without concern about changing git related files. Here is an example command that we use to train our `1024x1024 StyleGAN2 ` model.

```shell
bash tools/slurm_train.sh openmmlab-platform stylegan2-1024 \
    configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py \
    work_dirs/experiments/stylegan2_c2_ffhq_1024_b4x8
```

During training, log files and checkpoints will be saved to the working directory. At the beginning of our development, we evaluate our model after the training finishes. However, the evaluation hook has been already supported to evaluate our models in the training procedure. More details can be found in our tutorial for running time configuration.


# 4: Test existing models

Currently, we have supported **6 evaluation metrics**, i.e., MS-SSIM, SWD, IS, FID, Precision&Recall, and PPL. For unconditional GANs, we have provided unified evaluation scripts in [tools/evaluation.py](https://github.com/open-mmlab/mmgeneration/tree/master/tools/evaluation.py). Additionally, [configs/_base_/default_metrics.py](https://github.com/open-mmlab/mmgeneration/tree/master/configs/_base_/default_metrics.py) also offers the commonly used configurations to users. If users want to evaluate their models with some metrics, you can add the `metrics` into your config file like this:

```python
# at the end of the configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py
metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl='work_dirs/inception_pkl/ffhq-256-50k-rgb.pkl',
        bgr2rgb=True))
```
(We will specify how to obtain `inception_pkl` in the [FID](#FID) section.)
Then, users can use the evaluation script with the following command:

```shell
bash eval.sh ${CONFIG_FILE} ${CKPT_FILE} --batch-size 10 --online
```
If you are in slurm enviroment, please switch to the [tools/slurm_eval.sh](https://github.com/open-mmlab/mmgeneration/tree/master/tools/slurm_eval.sh) by using the following commands:

```shell
bash slurm_eval.sh ${PLATFORM} ${JOBNAME} ${CONFIG_FILE} ${CONFIG_FILE} \
    --batch-size 10
    --online
```

As you can see, we have provided two modes for evaluating your models, i.e., `online`, and `offline`. `online` mode indicates that the synthesized images will be directly passed to the metrics instead of being saved to the file system. If users have set the `--samples-path` argument, `offline` mode will save the generated images in this directory so that users can use them for other tasks. Besides, users can use the `offline` mode to sample images:

```shell
# for general envs
bash eval.sh ${CONFIG_FILE} ${CKPT_FILE} --eval none

# for slurm
bash slurm_eval.sh ${PLATFORM} ${JOBNAME} ${CONFIG_FILE} ${CONFIG_FILE} \
    --eval none
```

Next, we will specify the details of different metrics one by one.

## **FID**
Fréchet Inception Distance is a measure of similarity between two datasets of images. It was shown to correlate well with the human judgment of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks. FID is calculated by computing the Fréchet distance between two Gaussians fitted to feature representations of the Inception network.

In `MMGeneration`, we provide two versions for FID calculation. One is the commonly used PyTorch version and the other one is used in StyleGAN paper. Meanwhile, we have compared the difference between these two implementations in the StyleGAN2-FFHQ1024 model (the details can be found [here](https://github.com/open-mmlab/mmgeneration/blob/master/configs/styleganv2/README.md)). Fortunately, there is a marginal difference in the final results. Thus, we recommend users adopt the more convenient PyTorch version.

**About PyTorch version and Tero's version:** The commonly used PyTorch version adopts the modified InceptionV3 network to extract features for real and fake images. However, Tero's FID requires a [script module](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) for Tensorflow InceptionV3. Note that applying this script module needs `PyTorch >= 1.6.0`.

**About extracting real inception data:** For convenience, we always extract the features for real images in advance. In `MMGeneration`, we have provided [tools/utils/inception_stat.py](https://github.com/open-mmlab/mmgeneration/blob/master/tools/utils/inception_stat.py) for users to prepare the real inception data. After running the following command, the extracted features will be saved in a `pkl` file.

```shell
python tools/utils/inception_stat.py ${IMGS_PATH} ${PKLNAME} --size ${SIZE}
```
In the above command, the script will take the PyTorch InceptionV3 by default. If you want the Tero's InceptionV3, you will need to switch to the script module:

```shell
python tools/utils/inception_stat.py ${IMGS_PATH} ${PKLNAME} --size ${SIZE} \
    --inception-style stylegan --inception-pth ${PATH_SCRIPT_MODULE}
```

To use the FID metric, you should add the metric in a config file like this:

```python
metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl='work_dirs/inception_pkl/ffhq-256-50k-rgb.pkl',
        bgr2rgb=True))
```
If the `inception_pkl` is not set, the metric will calculate the real inception statistics on the fly. If you hope to use the Tero's InceptionV3, please use the following metric configuration:

```python
metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl='work_dirs/inception_pkl/ffhq-1024-50k-stylegan.pkl', inception_args=dict(
            type='StyleGAN',
            inception_path='work_dirs/cache/inception-2015-12-05.pt')))

```
The `inception_path` indicates the path to Tero's script module.

## Precision and Recall
Our `Precision and Recall` implementation follows the version used in StyleGAN2. In this metric, a VGG network will be adopted to extract the features for images. Unfortunately, we have not found a PyTorch VGG implementation leading to similar results with Tero's version used in StyleGAN2. (About the differences, please see this [file](https://github.com/open-mmlab/mmgeneration/blob/master/configs/styleganv2/README.md).) Thus, in our implementation, we adopt [Teor's VGG](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt) network by default. Importantly, applying this script module needs `PyTorch >= 1.6.0`. If with a lower PyTorch version, we will use the PyTorch official VGG network for feature extraction.

To evaluate with `P&R`, please add the following configuration in the config file:

```python
metrics = dict(
    PR=dict(
        type='PR',
        num_images=50000))
```

## IS
Inception score is an objective metric for evaluating the quality of generated images, proposed in [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf). It uses an InceptionV3 model to predict the class of the generated images, and suppose that 1) If an image is of high quality, it will be categorized into a specific class. 2) If images are of high diversity, the range of images' classes will be wide. So the KL-divergence of the conditional probability and marginal probability can indicate the quality and diversity of generated images. You can see the complete implementation in `metrics.py`, which refers to https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py.
If you want to evaluate models with `IS` metrics, you can add the `metrics` into your config file like this:
```python
# at the end of the configs/pix2pix/pix2pix_vanilla_unet_bn_1x1_80k_facades.py
metrics = dict(
    IS=dict(type='IS', num_images=106, image_shape=(3, 256, 256)))
```
You can run the command below to calculate IS.
```shell
python tools/utils/translation_eval.py ./configs/pix2pix/pix2pix_vanilla_unet_bn_1x1_80k_facades.py\
    https://download.openmmlab.com/mmgen/pix2pix/pix2pix_vanilla_unet_bn_1x1_80k_facades.py_20210410_174537-36d956f1.pth \
    --eval IS
```
To be noted that, the selection of Inception V3 and image resize method can significantly influence the final IS score. Therefore, we strongly recommend users may download the [Tero's script model of Inception V3](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) (load this script model need torch >= 1.6) and use `Bicubic` interpolation with `Pillow` backend. We provide a template for the [data process pipline](https://github.com/open-mmlab/mmgeneration/tree/master/configs/_base_/datasets/Inception_Score.py) as well.

We also perform a survey on the influence of data loading pipeline and the version of pretrained Inception V3 on the IS result. All IS are evaluated on the same group of images which are randomly selected from the ImageNet dataset.

<details> <summary> Show the Comparsion Results </summary>

| Code Base | Inception V3 Version | Data Loader Backend | Resize Interpolation Method | IS |
| -- | -- | -- | -- | -- |
| [OpenAI (baseline)](https://github.com/openai/improved-gan) | Tensorflow | Pillow | Pillow Bicubic | **312.255 +/- 4.970** |
| [StyleGAN-Ada](https://github.com/NVlabs/stylegan2-ada-pytorch) | Tero's Script Model | Pillow | Pillow Bicubic | 311.895 +/ 4.844 |
| mmgen (Ours) | Pytorch Pretrained | cv2 | cv2 Bilinear | 322.932 +/- 2.317 |
| mmgen (Ours) | Pytorch Pretrained | cv2 | cv2 Bicubic | 324.604 +/- 5.157 |
| mmgen (Ours) | Pytorch Pretrained | cv2 | Pillow Bicubic | 318.161 +/- 5.330 |
| mmgen (Ours) | Pytorch Pretrained | Pillow | Pillow Bilinear | 313.126 +/- 5.449 |
| mmgen (Ours) | Pytorch Pretrained | Pillow | cv2 Bilinear | 318.021+/-3.864 |
| mmgen (Ours) | Pytorch Pretrained | Pillow | Pillow Bicubic | 317.997 +/- 5.350 |
| mmgen (Ours) | Tero's Script Model | cv2 | cv2 Bilinear | 318.879 +/- 2.433 |
| mmgen (Ours) | Tero's Script Model | cv2 | cv2 Bicubic | 316.125 +/- 5.718 |
| mmgen (Ours) | Tero's Script Model | cv2 | Pillow Bicubic | **312.045 +/- 5.440** |
| mmgen (Ours) | Tero's Script Model | Pillow | Pillow Bilinear | 308.645 +/- 5.374 |
| mmgen (Ours) | Tero's Script Model | Pillow | Pillow Bicubic | 311.733 +/- 5.375 |

</details>

## PPL
Perceptual path length measures the difference between consecutive images (their VGG16 embeddings) when interpolating between two random inputs. Drastic changes mean that multiple features have changed together and that they might be entangled. Thus, a smaller PPL score appears to indicate higher overall image quality by experiments. \
As a basis for our metric, we use a perceptually-based pairwise image distance that is calculated as a weighted difference between two VGG16 embeddings, where the weights are fit so that the metric agrees with human perceptual similarity judgments.
If we subdivide a latent space interpolation path into linear segments, we can define the total perceptual length of this segmented path as the sum of perceptual differences over each segment, and a natural definition for the perceptual path length would be the limit of this sum under infinitely fine subdivision, but in practice we approximate it using a small subdivision <img src="https://latex.codecogs.com/svg.image?\epsilon&space;=&space;10^{-4}" title="\epsilon = 10^{-4}" />.
The average perceptual path length in latent `space` Z, over all possible endpoints, is therefore

<img src="https://latex.codecogs.com/svg.image?L_Z&space;=&space;E[\frac{1}{\epsilon^2}d(G(slerp(z_1,z_2;t))),&space;G(slerp(z_1,z_2;t&plus;\epsilon)))]" title="L_Z = E[\frac{1}{\epsilon^2}d(G(slerp(z_1,z_2;t))), G(slerp(z_1,z_2;t+\epsilon)))]" />

Computing the average perceptual path length in latent `space` W is carried out in a similar fashion:

<img src="https://latex.codecogs.com/svg.image?L_Z&space;=&space;E[\frac{1}{\epsilon^2}d(G(slerp(z_1,z_2;t))),&space;G(slerp(z_1,z_2;t&plus;\epsilon)))]" title="L_Z = E[\frac{1}{\epsilon^2}d(G(slerp(z_1,z_2;t))), G(slerp(z_1,z_2;t+\epsilon)))]" />

Where <img src="https://latex.codecogs.com/svg.image?\inline&space;z_1,&space;z_2&space;\sim&space;P(z)" title="\inline z_1, z_2 \sim P(z)" />, and <img src="https://latex.codecogs.com/svg.image?\inline&space;t&space;\sim&space;U(0,1)" title="\inline t \sim U(0,1)" /> if we set `sampling` to full, <img src="https://latex.codecogs.com/svg.image?\inline&space;t&space;\in&space;\{0,1\}" title="\inline t \in \{0,1\}" /> if we set `sampling` to end. <img src="https://latex.codecogs.com/svg.image?\inline&space;G" title="\inline G" /> is the generator(i.e. <img src="https://latex.codecogs.com/svg.image?\inline&space;g&space;\circ&space;f" title="\inline g \circ f" /> for style-based networks), and <img src="https://latex.codecogs.com/svg.image?\inline&space;d(.,.)" title="\inline d(.,.)" /> evaluates the perceptual distance between the resulting images.We compute the expectation by taking 100,000 samples (set `num_images` to 50,000 in our code).

You can find the complete implementation in `metrics.py`, which refers to https://github.com/rosinality/stylegan2-pytorch/blob/master/ppl.py.
If you want to evaluate models with `PPL` metrics, you can add the `metrics` into your config file like this:
```python
# at the end of the configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py
metrics = dict(
    ppl_wend=dict(type='PPL', space='W', sampling='end', num_images=50000, image_shape=(3, 1024, 1024)))
```
You can run the command below to calculate PPL.

```shell
python tools/evaluation.py ./configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py \
    https://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth \
    --batch-size 2 --online --eval ppl_wend
```

## SWD
Sliced Wasserstein distance is a discrepancy measure for probability distributions, and smaller distance indicates generated images look like the real ones. We obtain the Laplacian pyramids of every image and extract patches from the Laplacian pyramids as descriptors, then SWD can be calculated by taking the sliced Wasserstein distance of the real and fake descriptors.
You can see the complete implementation in `metrics.py`, which refers to https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py.
If you want to evaluate models with `SWD` metrics, you can add the `metrics` into your config file like this:
```python
# at the end of the configs/pggan/pggan_celeba-cropped_128_g8_12Mimgs.py
metrics = dict(swd16k=dict(type='SWD', num_images=16384, image_shape=(3, 128, 128)))
```
You can run the command below to calculate SWD.

```shell
python tools/evaluation.py ./configs/pggan/pggan_celeba-cropped_128_g8_12Mimgs.py \
    https://download.openmmlab.com/mmgen/pggan/pggan_celeba-cropped_128_g8_20210408_181931-85a2e72c.pth \
    --batch-size 64 --online --eval swd16k
```

## MS-SSIM
Multi-scale structural similarity is used to measure the similarity of two images. We use MS-SSIM here to measure the diversity of generated images, and a low MS-SSIM score indicates the high diversity of generated images. You can see the complete implementation in `metrics.py`, which refers to https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py.
If you want to evaluate models with `MS-SSIM` metrics, you can add the `metrics` into your config file like this:
```python
# at the end of the configs/pggan/pggan_celeba-cropped_128_g8_12Mimgs.py
metrics = dict(ms_ssim10k=dict(type='MS_SSIM', num_images=10000))
```
You can run the command below to calculate MS-SSIM.

```shell
python tools/evaluation.py ./configs/pggan/pggan_celeba-cropped_128_g8_12Mimgs.py \
    https://download.openmmlab.com/mmgen/pggan/pggan_celeba-cropped_128_g8_20210408_181931-85a2e72c.pth \
    --batch-size 64 --online --eval ms_ssim10k
```


# 5: Evaluation during training

In this section, we will discuss how to evaluate the generative models, especially for GANs, in the training. Note that `MMGeneration` only supports distributed training and the evaluation metric adopted in the training procedure should also be run in a distributed style. Currently, only `FID` has been implemented and tested in an efficient distributed version. Other metrics with efficient distributed version will be supported in the recent future. Thus, in the following part, we will specify how to evaluate your models with `FID` metric in training.

In [eval_hooks.py](https://github.com/open-mmlab/mmgeneration/blob/master/mmgen/core/evaluation/eval_hooks.py), `GenerativeEvalHook` is provided to evaluate generative models duiring training. The most important argument for this hook is `metrics`. In fact, users can directly copy the configs in the last section to define the evaluation metric. To evaluate the model with `FID` metric, please add the following python codes in your config file:

```python
# define the evaluation keywords, otherwise evalution will not be
# added in training
evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=dict(
        type='FID',
        num_images=50000,
        inception_pkl='path_to_inception_pkl',
        bgr2rgb=True),
    sample_kwargs=dict(sample_model='ema'))
```

For `FID` evaluation, our distributed version only takes about **400 seconds (7 minutes)**. Thus, it will not influence the training time significantly. In addition, users should also offer the `val` dataset, even if this metric will not use the files from this dataset:

```python
data = dict(
    samples_per_gpu=4,
    train=dict(dataset=dict(imgs_root='./data/ffhq/ffhq_imgs/ffhq_256')),
    val=dict(imgs_root='./data/ffhq/ffhq_imgs/ffhq_256'))
```

We highly recommend that users should pre-calculate the inception pickle file in advance, which will reduce the evaluation cost significantly.
