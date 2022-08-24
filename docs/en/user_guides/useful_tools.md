# Useful tools

We also provide tools for users.

## Print full config

We incorporate modular and inheritance design into our [config](config.md) system, which is convenient to conduct various experiments. If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

An Example:

```python
python tools/misc/print_config.py configs/styleganv2/stylegan2_c2-PL_8xb4-fp16-partial-GD-no-scaler-800kiters_ffhq-256x256.py
```

### Prepare a model for publishing

`tools/publish_model.py` helps users to prepare their model for publishing.

Before you upload a model to AWS, you may want to

1. convert model weights to CPU tensors
2. delete the optimizer states and
3. compute the hash of the checkpoint file and append time and the hash id to the
   filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

An Example:

```shell
python tools/publish_model.py work_dirs/stylegan2/latest.pth stylegan2_c2_8xb4_ffhq-1024x1024.pth
```

The final output filename will be `stylegan2_c2_8xb4_ffhq-1024x1024_{time}-{hash id}.pth`.
