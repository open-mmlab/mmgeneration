# train_test

## Prepare dataset for training and testing

This section details how to prepare the dataset for MMGeneration and provides a standard way which we have used in our default configs. We recommend that all of the users may follow the following steps to organize their datasets.

### Datasets for unconditional models

It's much easier to prepare dataset for unconditional models. Firstly, please make a directory, named `data`, in the MMGeneration project. After that, all of datasets can be used by adopting the technology of symlink (soft link).

```shell
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

## Train existing models

Currently, we have tested all of the model on distributed training. Thus, we highly recommend to adopt distributed training with our scripts. The basic usage is as follows:

```shell
sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS_NUMBER} \
    --work-dir ./work_dirs/experiments/experiments_name \
    [optional arguments]
```

If you are using slurm system, the following commands can help you start training"

```shell
sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${WORK_DIR} \
    [optional arguments]
```

There two scripts wrap [tools/train.py](https://github.com/open-mmlab/mmgeneration/tree/master/tools/train.py) with distributed training entrypoint. The `optional arguments` are defined in [tools/train.py](https://github.com/open-mmlab/mmgeneration/tree/master/tools/train.py). Users can also set `random-seed` and `resume-from` with these arguments.

Note that the name of `work_dirs` has already been put into our `.gitignore` file. Users can put any files here without concern about changing git related files. Here is an example command that we use to train our `1024x1024 StyleGAN2 ` model.

```shell
sh tools/slurm_train.sh openmmlab-platform stylegan2-1024 \
    configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py \
    work_dirs/experiments/stylegan2_c2_ffhq_1024_b4x8
```

During training, log files and checkpoints will be saved to the working directory. At the beginning of our development, we evaluate our model after the training finishes. However, the evaluation hook has been already supported to evaluate our models in the training procedure. More details can be found in our tutorial for running time configuration.

### Training with multiple machines

If you launch with multiple machines simply connected with ethernet, you can simply run following commands:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

Usually it is slow if you do not have high speed networking like InfiniBand.

If you launch with slurm, the command is the same as that on single machine described above, but you need refer to [slurm_train.sh](https://github.com/open-mmlab/mmgeneration/blob/master/tools/slurm_train.sh) to set appropriate parameters and environment variables.

### Training on CPU

The process of training on the CPU is consistent with single GPU training. We just need to disable GPUs before the training process.

```shell
export CUDA_VISIBLE_DEVICES=-1
```

And then run this script.

```shell
python tools/train.py config --work-dir WORK_DIR
```

**Note**:

We do not recommend users to use CPU for training because it is too slow. We support this feature to allow users to debug on machines without GPU for convenience.

## Test existing models

Currently, we have supported **9 evaluation metrics**, i.e., MS-SSIM, SWD, IS, FID, Precision&Recall, PPL, Equivarience, TransFID, TransIS. We have provided unified evaluation scripts in [tools/test.py](https://github.com/open-mmlab/mmgeneration/tree/1.x/tools/test.py) for all models. If users want to evaluate their models with some metrics, you can add the `metrics` into your config file like this:

```python
# at the end of the configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema'),
    dict(type='PrecisionAndRecall', fake_nums=50000, prefix='PR-50K'),
    dict(type='PerceptualPathLength', fake_nums=50000, prefix='ppl-w')
]
```

As above, `metrics` consists of multiple metric dictionaries. Each metric will contain `type` to indicate the category of the metric. `fake_nums` denotes the number of images generated by model. Some metrics will output a dictionary of results, you can also set `prefix`  to specify the prefix of the results.
If you set prefix of FID as `FID-Full-50k`, then an example of output may be

```bash
FID-Full-50k/fid: 3.6561  FID-Full-50k/mean: 0.4263  FID-Full-50k/cov: 3.2298
```

Then users can test models with command below:

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CKPT_FILE}
```

If you are in slurm environment, please switch to the [tools/slurm_test.sh](https://github.com/open-mmlab/mmgeneration/tree/1.x/tools/slurm_test.sh) by using the following commands:

```shell
sh slurm_test.sh ${PLATFORM} ${JOBNAME} ${CONFIG_FILE} ${CKPT_FILE}
```

Next, we will specify the details of different metrics one by one.

### **FID** and **TransFID**

Fréchet Inception Distance is a measure of similarity between two datasets of images. It was shown to correlate well with the human judgment of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks. FID is calculated by computing the Fréchet distance between two Gaussians fitted to feature representations of the Inception network.

In `MMGeneration`, we provide two versions for FID calculation. One is the commonly used PyTorch version and the other one is used in StyleGAN paper. Meanwhile, we have compared the difference between these two implementations in the StyleGAN2-FFHQ1024 model (the details can be found [here](https://github.com/open-mmlab/mmgeneration/blob/master/configs/styleganv2/README.md)). Fortunately, there is a marginal difference in the final results. Thus, we recommend users adopt the more convenient PyTorch version.

**About PyTorch version and Tero's version:** The commonly used PyTorch version adopts the modified InceptionV3 network to extract features for real and fake images. However, Tero's FID requires a [script module](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) for Tensorflow InceptionV3. Note that applying this script module needs `PyTorch >= 1.6.0`.

**About extracting real inception data:** For the users' convenience, the real features will be automatically extracted at test time and saved locally, and the stored features will be automatically read at the next test. Specifically, we will calculate a hash value based on the parameters used to calculate the real features, and use the hash value to mark the feature file, and when testing, if the `inception_pkl` is not set, we will look for the feature in `MMGEN_CACHE_DIR` (~/.cache/openmmlab/mmgen/). If cached inception pkl is not found, then extracting will be performed.

To use the FID metric, you should add the metric in a config file like this:

```python
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema')
]
```

If you work on an new machine, then you can copy the `pkl` files in `MMGEN_CACHE_DIR` and copy them to new machine and set `inception_pkl` field.

```python
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        inception_pkl=
        'work_dirs/inception_pkl/inception_state-capture_mean_cov-full-33ad4546f8c9152e4b3bdb1b0c08dbaf.pkl',  # copied from old machine
        sample_model='ema')
]
```

`TransFID` has same usage as `FID`, but it's designed for translation models like `pix2pix` and `cyclegan`, which is designed for our evaluator. You can refer
to [evaluation](../advanced_guides/evaluation.md) for details.

### **IS** and **TransIS**

Inception score is an objective metric for evaluating the quality of generated images, proposed in [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf). It uses an InceptionV3 model to predict the class of the generated images, and suppose that 1) If an image is of high quality, it will be categorized into a specific class. 2) If images are of high diversity, the range of images' classes will be wide. So the KL-divergence of the conditional probability and marginal probability can indicate the quality and diversity of generated images. You can see the complete implementation in `metrics.py`, which refers to https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py.
If you want to evaluate models with `IS` metrics, you can add the `metrics` into your config file like this:

```python
# at the end of the configs/biggan/biggan_2xb25-500kiters_cifar10-32x32.py
metrics = [
    xxx,
    dict(
        type='IS',
        prefix='IS-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema')
]
```

To be noted that, the selection of Inception V3 and image resize method can significantly influence the final IS score. Therefore, we strongly recommend users may download the [Tero's script model of Inception V3](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) (load this script model need torch >= 1.6) and use `Bicubic` interpolation with `Pillow` backend. We provide a template for the [data process pipline](https://github.com/open-mmlab/mmgeneration/tree/master/configs/_base_/datasets/Inception_Score.py) as well.

Corresponding to config, you can set `resize_method` and `use_pillow_resize` for image resizing. You can also set `inception_style` as `StyleGAN` for recommended tero's inception model, or `PyTorch` for torchvision's implementation. For environment without internet, you can download the inception's weights, and set `inception_path` to your inception model.

We also perform a survey on the influence of data loading pipeline and the version of pretrained Inception V3 on the IS result. All IS are evaluated on the same group of images which are randomly selected from the ImageNet dataset.

<details> <summary> Show the Comparison Results </summary>

|                            Code Base                            | Inception V3 Version | Data Loader Backend | Resize Interpolation Method |          IS           |
| :-------------------------------------------------------------: | :------------------: | :-----------------: | :-------------------------: | :-------------------: |
|   [OpenAI (baseline)](https://github.com/openai/improved-gan)   |      Tensorflow      |       Pillow        |       Pillow Bicubic        | **312.255 +/- 4.970** |
| [StyleGAN-Ada](https://github.com/NVlabs/stylegan2-ada-pytorch) | Tero's Script Model  |       Pillow        |       Pillow Bicubic        |   311.895 +/ 4.844    |
|                          mmgen (Ours)                           |  Pytorch Pretrained  |         cv2         |        cv2 Bilinear         |   322.932 +/- 2.317   |
|                          mmgen (Ours)                           |  Pytorch Pretrained  |         cv2         |         cv2 Bicubic         |   324.604 +/- 5.157   |
|                          mmgen (Ours)                           |  Pytorch Pretrained  |         cv2         |       Pillow Bicubic        |   318.161 +/- 5.330   |
|                          mmgen (Ours)                           |  Pytorch Pretrained  |       Pillow        |       Pillow Bilinear       |   313.126 +/- 5.449   |
|                          mmgen (Ours)                           |  Pytorch Pretrained  |       Pillow        |        cv2 Bilinear         |    318.021+/-3.864    |
|                          mmgen (Ours)                           |  Pytorch Pretrained  |       Pillow        |       Pillow Bicubic        |   317.997 +/- 5.350   |
|                          mmgen (Ours)                           | Tero's Script Model  |         cv2         |        cv2 Bilinear         |   318.879 +/- 2.433   |
|                          mmgen (Ours)                           | Tero's Script Model  |         cv2         |         cv2 Bicubic         |   316.125 +/- 5.718   |
|                          mmgen (Ours)                           | Tero's Script Model  |         cv2         |       Pillow Bicubic        | **312.045 +/- 5.440** |
|                          mmgen (Ours)                           | Tero's Script Model  |       Pillow        |       Pillow Bilinear       |   308.645 +/- 5.374   |
|                          mmgen (Ours)                           | Tero's Script Model  |       Pillow        |       Pillow Bicubic        |   311.733 +/- 5.375   |

</details>

`TransIS` has same usage as `IS`, but it's designed for translation models like `pix2pix` and `cyclegan`, which is designed for our evaluator. You can refer
to [evaluation](../advanced_guides/evaluation.md) for details.

### Precision and Recall

Our `Precision and Recall` implementation follows the version used in StyleGAN2. In this metric, a VGG network will be adopted to extract the features for images. Unfortunately, we have not found a PyTorch VGG implementation leading to similar results with Tero's version used in StyleGAN2. (About the differences, please see this [file](https://github.com/open-mmlab/mmgeneration/blob/1.x/configs/styleganv2/README.md).) Thus, in our implementation, we adopt [Teor's VGG](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt) network by default. Importantly, applying this script module needs `PyTorch >= 1.6.0`. If with a lower PyTorch version, we will use the PyTorch official VGG network for feature extraction.

To evaluate with `P&R`, please add the following configuration in the config file:

```python
metrics = [
    dict(type='PrecisionAndRecall', fake_nums=50000, prefix='PR-50K')
]
```

### PPL

Perceptual path length measures the difference between consecutive images (their VGG16 embeddings) when interpolating between two random inputs. Drastic changes mean that multiple features have changed together and that they might be entangled. Thus, a smaller PPL score appears to indicate higher overall image quality by experiments. \
As a basis for our metric, we use a perceptually-based pairwise image distance that is calculated as a weighted difference between two VGG16 embeddings, where the weights are fit so that the metric agrees with human perceptual similarity judgments.
If we subdivide a latent space interpolation path into linear segments, we can define the total perceptual length of this segmented path as the sum of perceptual differences over each segment, and a natural definition for the perceptual path length would be the limit of this sum under infinitely fine subdivision, but in practice we approximate it using a small subdivision `` $`\epsilon=10^{-4}`$ ``.
The average perceptual path length in latent `space` Z, over all possible endpoints, is therefore

`` $$`L_Z = E[\frac{1}{\epsilon^2}d(G(slerp(z_1,z_2;t))), G(slerp(z_1,z_2;t+\epsilon)))]`$$ ``

Computing the average perceptual path length in latent `space` W is carried out in a similar fashion:

`` $$`L_Z = E[\frac{1}{\epsilon^2}d(G(slerp(z_1,z_2;t))), G(slerp(z_1,z_2;t+\epsilon)))]`$$ ``

Where `` $`z_1, z_2 \sim P(z)`$ ``, and `` $` t \sim U(0,1)`$ `` if we set `sampling` to full, `` $` t \in \{0,1\}`$ `` if we set `sampling` to end. `` $` G`$ `` is the generator(i.e. `` $` g \circ f`$ `` for style-based networks), and `` $` d(.,.)`$ `` evaluates the perceptual distance between the resulting images.We compute the expectation by taking 100,000 samples (set `num_images` to 50,000 in our code).

You can find the complete implementation in `metrics.py`, which refers to https://github.com/rosinality/stylegan2-pytorch/blob/master/ppl.py.
If you want to evaluate models with `PPL` metrics, you can add the `metrics` into your config file like this:

```python
# at the end of the configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py
metrics = [
    xxx,
    dict(type='PerceptualPathLength', fake_nums=50000, prefix='ppl-w')
]
```

### SWD

Sliced Wasserstein distance is a discrepancy measure for probability distributions, and smaller distance indicates generated images look like the real ones. We obtain the Laplacian pyramids of every image and extract patches from the Laplacian pyramids as descriptors, then SWD can be calculated by taking the sliced Wasserstein distance of the real and fake descriptors.
You can see the complete implementation in `metrics.py`, which refers to https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py.
If you want to evaluate models with `SWD` metrics, you can add the `metrics` into your config file like this:

```python
# at the end of the configs/dcgan/dcgan_1xb128-5epoches_lsun-bedroom-64x64.py
metrics = [
    dict(
        type='SWD',
        prefix='swd',
        fake_nums=16384,
        sample_model='orig',
        image_shape=(3, 64, 64))
]
```

### MS-SSIM

Multi-scale structural similarity is used to measure the similarity of two images. We use MS-SSIM here to measure the diversity of generated images, and a low MS-SSIM score indicates the high diversity of generated images. You can see the complete implementation in `metrics.py`, which refers to https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py.
If you want to evaluate models with `MS-SSIM` metrics, you can add the `metrics` into your config file like this:

```python
# at the end of the configs/dcgan/dcgan_1xb128-5epoches_lsun-bedroom-64x64.py
metrics = [
    dict(
        type='MS_SSIM', prefix='ms-ssim', fake_nums=10000,
        sample_model='orig')
]
```

### Equivarience

Equivarience of generative models refer to the exchangeability of model forward and geometric transformations. Currently this metric is only calculated for StyleGANv3,
you can see the complete implementation in `metrics.py`, which refers to https://github.com/NVlabs/stylegan3/blob/main/metrics/equivariance.py.
If you want to evaluate models with `Equivarience` metrics, you can add the `metrics` into your config file like this:

```python
# at the end of the configs/styleganv3/stylegan3-t_gamma2.0_8xb4-fp16-noaug_ffhq-256x256.py
metrics = [
    dict(
        type='Equivariance',
        fake_nums=50000,
        sample_mode='ema',
        prefix='EQ',
        eq_cfg=dict(
            compute_eqt_int=True, compute_eqt_frac=True, compute_eqr=True))
]
```

## Evaluation during training

Benefit from the `mmengine`'s `Runner`. We can evaluate model during training in a simple way as below.

```python
test_dataloader = xxx
metrics = xxx
val_evaluator = dict(metrics=metrics)
val_dataloader = test_dataloader
train_cfg = dict(xxx, val_begin=1, val_interval=10000)
```

You can set `val_begin` and `val_interval` to adjust when to begin valiadation and interval of validation.
