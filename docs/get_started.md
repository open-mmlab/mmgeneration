## Prerequisites

- Linux
- Python 3.6+
- PyTorch 1.5+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV (MMCV-FULL)](https://mmcv.readthedocs.io/en/latest/#installation)

The compatible MMGeneration and MMCV versions are as below. Please install the correct version of MMCV to avoid installation issues.

| MMGeneration version |   MMCV version   |
| :------------------: | :--------------: |
|        master        | mmcv-full>=1.3.0 |

Note: You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

## Installation

1. Create a conda virtual environment and activate it. (Here, we assume the new environment is called ``open-mmlab``)

    ```shell
    conda create -n open-mmlab python=3.7 -y
    conda activate open-mmlab
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

    ```shell
    conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
    ```

    `E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.5.1., you need to install the prebuilt PyTorch with CUDA 9.2.

    ```shell
    conda install pytorch=1.5.1 cudatoolkit=9.2 torchvision=0.6.1 -c pytorch
    ```

    If you build PyTorch from source instead of installing the prebuilt package,
    you can use more CUDA versions such as 9.0.

3. Install mmcv-full, we recommend you to install the pre-build package as below.

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    Please replace `{cu_version}` and `{torch_version}` in the url to your desired one. For example, to install the latest `mmcv-full` with `CUDA 11` and `PyTorch 1.7.0`, use the following command:

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
    ```

    See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.
    Optionally you can choose to compile mmcv from source by the following command

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
    cd ..
    ```

    Or directly run

    ```shell
    pip install mmcv-full
    ```

4. Clone the MMGeneration repository.

    ```shell
    git clone https://github.com/open-mmlab/mmgeneration.git
    cd mmgeneration
    ```

5. Install build requirements and then install MMGeneration.

    ```shell
    pip install -r requirements.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

Note:

a. Following the above instructions, MMGeneration is installed on `dev` mode,
any local modifications made to the code will take effect without the need to reinstall it.

b. If you would like to use `opencv-python-headless` instead of `opencv
-python`,
you can install it before installing MMCV.

### Install with CPU only

The code can be built for CPU only environment (where CUDA isn't available).


### A from-scratch setup script

Assuming that you already have CUDA 10.1 installed, here is a full script for setting up MMGeneration with conda.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1 -c pytorch -y

# install the latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html

# install mmgeneration
git clone https://github.com/open-mmlab/mmgeneration.git
cd mmgeneration
pip install -r requirements.txt
pip install -v -e .
```

### Developing with multiple MMGeneration versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script uses the `MMGeneration` in the current directory.

To use the default MMGeneration installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## Verification

To verify whether MMGeneration and the required environment are installed correctly, we can run sample Python code to initialize an unconditional model and use it to generate random samples:

```python
from mmgen.apis import init_model, sample_uncoditional_model

config_file = 'configs/styleganv2/stylegan2_c2_lsun-church_256_b4x8_800k.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth'
device = 'cuda:0'
# init a generatvie
model = init_model(config_file, checkpoint_file, device=device)
# sample images
fake_imgs = sample_uncoditional_model(model, 4)
```

The above code is supposed to run successfully upon you finish the installation.
