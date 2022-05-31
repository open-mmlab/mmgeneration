# Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMGeneration works on Linux, Windows and macOS. It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.5+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
```

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

On GPU platforms:

```shell
conda create -name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install Pytorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# Installation

We recommend that users follow our best practices to install MMGeneration. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

## Best Practices

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 1.** Install MMGeneration.

```shell
git clone https://github.com/open-mmlab/mmgeneration.git
cd mmgeneration
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Case b: If you use mmgeneration as a dependency or third-party package, install it with pip:

```shell
pip install mmgen
```

## Verify the Installation

To verify whether MMGeneration and the required environment are installed correctly, we can run sample Python code to initialize an unconditional model and use it to generate random samples:

```python
from mmgen.apis import init_model, sample_unconditional_model

config_file = 'configs/styleganv2/stylegan2_c2_lsun-church_256_b4x8_800k.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth'
device = 'cuda:0'
# init a generatvie
model = init_model(config_file, checkpoint_file, device=device)
# sample images
fake_imgs = sample_unconditional_model(model, 4)
```

The above code is supposed to run successfully upon you finish the installation.

## Customize Installation

### CUDA Version

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

```{note}
Installing CUDA runtime libraries is enough if you follow our best practices, because no CUDA code will be compiled locally. However if you hope to compile MMCV from source or develop other CUDA operators, you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads), and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in `conda install` command.
```

### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv-full built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### Using MMGeneration with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmgeneration/blob/master/docker/Dockerfile) to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.8, CUDA 11.1
# If you prefer other versions, just modified the Dockerfile
docker build -t mmgeneration docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmgeneration/data mmgeneration
```

## Trouble shooting

If you have some issues during the installation, please first view the [FAQ](faq.md) page.
You may [open an issue](https://github.com/open-mmlab/mmgeneration/issues/new/choose) on GitHub if no solution is found.

# Developing with multiple MMGeneration versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script uses the `MMGeneration` in the current directory.

To use the default MMGeneration installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
