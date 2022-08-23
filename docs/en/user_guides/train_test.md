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

## Evaluation during training
