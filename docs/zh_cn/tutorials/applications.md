# Tutorial 8: 生成模型的应用

## 插值

以GAN为架构的生成模型学习将潜码空间中的点映射到生成的图像上。生成模型赋予了潜码空间的具体意义。一般来说，我们想探索潜码空间的结构，我们可以做的一件事是在潜码空间的两个端点之间插入一系列点，观察这些点生成的结果。(例如，我们认为，如果任何一个端点都不存在的特征出现在线性插值路径的中间点，则说明潜码空间是纠缠在一起的，动态属性没有得到适当的分离。)

我们为用户提供了一个应用脚本。你可以使用[apps/interpolate_sample.py](https://github.com/open-mmlab/mmgeneration/tree/master/apps/interpolate_sample.py)的以下命令进行无条件模型的插值。

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

在这里，我们提供两种显示模式（SHOW_MODE），即序列（sequence）和组（group）。在序列模式下，我们首先对一连串的端点进行采样，然后按顺序对两个端点之间的点进行插值，生成的图像将被单独保存。在组模式下，我们先采样几对端点，然后在每对端点之间进行插值，生成的图像将被保存在一张图片中。此外，`space` 指的是潜码空间，你可以选择'z'或'w'（指StyleGAN系列中的风格空间），`endpoint` 表示你要采样的端点数量（在 `group` 模式中应设置为偶数），`interval`表示你在两个端点之间插值的点的数量（包括端点）。

注意，我们还提供了更多的自定义参数来定制你的插值程序。
请使用`python apps/interpolate_sample.py --help`来查看更多细节。

如同上面的方法，你可以使用[apps/conditional_interpolate.py](https://github.com/open-mmlab/mmgeneration/tree/master/apps/conditional_interpolate.py)和下列命令进行条件模型的插值。

```bash
python apps/conditional_interpolate.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    [--show-mode ${SHOW_MODE}] \
    [--endpoint ${ENDPOINT}] \
    [--interval ${INTERVAL}] \
    [--embedding-name ${EMBEDDING_NAME}]
    [--fix-z] \
    [--fix-y] \
    [--samples-path ${SAMPLES_PATH}] \
    [--batch-size ${BATCH_SIZE}] \
```

在这里，与无条件模型不同，如果标签嵌入在 `conv_blocks` 之间共享，你需要提供嵌入层的名称。否则，你应该将 `embedding-name` 设置为 `NULL`。考虑到条件模型有噪声和标签作为输入，我们提供 `fix-z` 来固定噪声，`fix-y` 来固定标签。

## 投影

求生成网络 `g` 的逆是一个有趣的问题，有很多应用。例如，在潜码空间中操作一个给定的图像需要先为它找到一个匹配的潜码。一般来说，你可以通过对潜码进行优化来重建目标图像，使用 `lpips` 和像素级损失作为目标函数。

事实上，我们已经向用户提供了一个应用脚本，为给定的图像找到 `StyleGAN` 系列生成网络的匹配潜码向量w。你可以使用[apps/stylegan_projector.py](https://github.com/open-mmlab/mmgeneration/tree/master/apps/stylegan_projector.py)的以下命令来执行投影。

```bash
python apps/stylegan_projector.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    ${FILES}
    [--results-path ${RESULTS_PATH}]
```

这里，`FILES` 指的是图像的路径，而投影的潜码和重建的图像将被保存在 `results-path` 中。
注意，我们还提供了更多的自定义参数来定制你的投影程序。请使用`python apps/stylegan_projector.py --help`来查看更多细节。

## 编辑

基于 StyleGAN 模型的一个常见应用是操纵潜码空间来控制合成图像的属性。在这里，我们向用户提供了一个基于\[SeFa\]（https://arxiv.org/pdf/2007.06600.pdf%EF%BC%89%E7%9A%84%E7%AE%80%E5%8D%95%E8%80%8C%E6%B5%81%E8%A1%8C%E7%9A%84%E7%AE%97%E6%B3%95%E3%80%82%E8%BF%99%E9%87%8C%EF%BC%8C%E6%88%91%E4%BB%AC%E5%9C%A8%E8%AE%A1%E7%AE%97%E7%89%B9%E5%BE%81%E5%90%91%E9%87%8F%E6%97%B6%E5%AF%B9%E5%8E%9F%E5%A7%8B%E7%89%88%E6%9C%AC%E8%BF%9B%E8%A1%8C%E4%BA%86%E4%BF%AE%E6%94%B9%EF%BC%8C%E5%B9%B6%E6%8F%90%E4%BE%9B%E4%BA%86%E4%B8%80%E4%B8%AA%E6%9B%B4%E7%81%B5%E6%B4%BB%E7%9A%84%E6%8E%A5%E5%8F%A3%E3%80%82

为了操纵你的生成器，你可以用以下命令运行脚本[apps/modified_sefa.py](https://github.com/open-mmlab/mmgeneration/tree/master/apps/modified_sefa.py)。

```shell
python apps/modified_sefa.py --cfg ${CONFIG} --ckpt ${CKPT} \
    -i ${INDEX} -d ${DEGREE} --degree-step ${D_STEP} \
    -l ${LAYER_NO} \
    [--eigen-vector ${PATH_EIGEN_VEC}]
```

在这个脚本中，如果 `eigen-vector` 为 `None`，程序将计算生成器参数的特征向量。同时，我们将把该向量保存在 `ckpt` 文件的同一目录下，这样用户就可以应用这个预先计算的向量。`Positional Encoding as Spatial Inductive Bias in GANs` 的演示就来自这个脚本。下面是一个例子，供用户获得与我们的演示类似的结果。

`${INDEX}`表示我们将应用哪个特征向量来操作图像。在一般情况下，每个索引控制一个独立的属性，这是由 `StyleGAN` 中的解耦表示保证的。我们建议用户可以尝试不同的索引来找到你想要的那个属性。`--degree` 设定了乘法因子的范围。在我们的实验中，我们观察到像 `[-3, 8]` 这样的非对称范围是非常有帮助的。因此，我们允许在这个参数中设置下限和上限。`--layer` 或`--l` 定义了我们将应用哪一层的特征向量。有些属性，比如光照，只与生成器中的 1-2 层有关。

以光照属性为例，我们在 MS-PIE-StyleGAN2-256 模型上运行以下命令。

```shell
python apps/modified_sefa.py \
    --config configs/positional_encoding_in_gans/mspie-stylegan2-config-f_c2_8xb3-1100kiters_ffhq-256-512.py \
    --ckpt https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c2_config-f_ffhq_256-512_b3x8_1100k_20210406_144927-4f4d5391.pth \
    -i 15 -d 8. --degree-step 0.5 -l 8 9 --sample-path ./work_dirs/sefa-exp/ \
    --sample-cfg chosen_scale=4 randomize_noise=False
```

注意到，在设置 `chosen_scale=4` 之后，我们可以用一个简单的分辨率为256的生成器来操作512x512的图像。
