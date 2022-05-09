## 依赖项

- Linux
- Python 3.6+
- PyTorch 1.5+
- CUDA 9.2+ (如果您从源码编译PyTorch， CUDA 9.0也是兼容的)
- GCC 5.4+
- [MMCV (MMCV-FULL)](https://mmcv.readthedocs.io/en/latest/#installation)

下面是MMGeneration与MMCV版本兼容信息。为防止出错请安装正确的MMCV版本。

| MMGeneration version |   MMCV version   |
| :------------------: | :--------------: |
|        master        | mmcv-full>=1.3.0 |

注：如果您已安装mmcv，需要先卸载 `pip uninstall mmcv`。 如果同时安装了mmcv和mmcv-full，将会报错 `ModuleNotFoundError`。

## 安装

1. 创建conda虚拟环境并激活。 (这里假设新环境叫 `open-mmlab`)

   ```shell
   conda create -n open-mmlab python=3.7 -y
   conda activate open-mmlab
   ```

2. 安装 PyTorch 和 torchvision，参考[官方安装指令](https://pytorch.org/)，比如，

   ```shell
   conda install pytorch torchvision -c pytorch
   ```

   注：确保您编译的CUDA版本和运行时CUDA版本相匹配。您可以在[PyTorch官网](https://pytorch.org/)检查预编译库支持的CUDA版本。

   `示例1` 如果您在`/usr/local/cuda`下安装了 CUDA 10.1 并想要安装
   PyTorch 1.5，您需要安装支持CUDA 10.1的PyTorch预编译版本。

   ```shell
   conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
   ```

   `示例2`如果您在`/usr/local/cuda`下安装了 CUDA 9.2 并想要安装
   PyTorch 1.5.1，您需要安装支持CUDA 9.2的PyTorch预编译版本。

   ```shell
   conda install pytorch=1.5.1 cudatoolkit=9.2 torchvision=0.6.1 -c pytorch
   ```

   如果您从源码编译PyTorch 而非安装预编译库， 您可以使用更多CUDA版本如9.0。

3. 安装 mmcv-full， 我们建议您按照下述方法安装预编译库。

   ```shell
   pip install mmcv-full={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
   ```

   请替换链接中的 `{cu_version}` 和 `{torch_version}` 为您想要的版本。 比如， 要安装支持 `CUDA 11` 和 `PyTorch 1.7.0`的 `mmcv-full`， 使用下面命令:

   ```shell
   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
   ```

   可在[这里](https://github.com/open-mmlab/mmcv#install-with-pip)查看兼容了不同PyTorch和CUDA的MMCV版本信息。
   您也可以选择按照下方命令从源码编译mmcv

   ```shell
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
   cd ..
   ```

   或者直接运行

   ```shell
   pip install mmcv-full
   ```

4. 克隆MMGeneration仓库。

   ```shell
   git clone https://github.com/open-mmlab/mmgeneration.git
   cd mmgeneration
   ```

5. 安装构建依赖项并安装MMGeneration。

   ```shell
   pip install -r requirements.txt
   pip install -v -e .  # or "python setup.py develop"
   ```

注:

a. 依照上面的说明， MMGeneration 会以 `dev` 形式安装，
对代码进行的任何本地修改都将生效，而不需要重新安装。

b. 如果您想要使用 `opencv-python-headless` 而非 `opencv-python`，
您可以在安装 MMCV 之前安装它。

### 安装CPU版本

本代码可在仅使用CPU的环境下编译 (当 CUDA 不可用时)。

### 一个从头开始的配置脚本

假设您已经安装了CUDA 10.1，下面是使用conda配置MMGeneration的完整脚本。

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1 -c pytorch -y

# install the latest mmcv
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html

# install mmgeneration
git clone https://github.com/open-mmlab/mmgeneration.git
cd mmgeneration
pip install -r requirements.txt
pip install -v -e .
```

需要注意的是，mmcv-full 只在 PyTorch 1.x.0 上编译， 因为1.x.0与1.x.1通常是保持兼容性的。 如果您的 PyTorch 版本是1.x.1， 您可以安装兼容PyTorch 1.x.0的mmcv-full，通常运行良好。

```shell
# We can ignore the micro version of PyTorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html
```

### 使用多个 MMGeneration 版本进行开发

训练和测试脚本已经修改了 `PYTHONPATH`， 以确保脚本使用当前目录中的`MMGeneration`。

要使用安装在环境中的默认MMGeneration而不是您正在使用的，您可以删除脚本中的以下代码行

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## 验证

为了验证MMGeneration和所需的环境是否正确安装，我们可以运行示例Python代码来初始化一个非条件模型，并使用它来生成随机样本:

```python
from mmgen.apis import init_model， sample_unconditional_model

config_file = 'configs/styleganv2/stylegan2_c2_lsun-church_256_b4x8_800k.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth'
device = 'cuda:0'
# init a generatvie
model = init_model(config_file， checkpoint_file， device=device)
# sample images
fake_imgs = sample_unconditional_model(model， 4)
```

当安装完成后，上面的代码可以成功运行。
