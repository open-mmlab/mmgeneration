## FAQ

### Ignore some fields in the base configs

Sometimes, you may set `_delete_=True` to ignore some of fields in base configs.
You may refer to [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/config.md#%E5%88%A0%E9%99%A4%E5%AD%97%E5%85%B8%E4%B8%AD%E7%9A%84-key) for simple illustration.

You may have a careful look at [this tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/config.md) for better understanding of this feature.

### Use intermediate variables in configs

Some intermediate variables are used in the config files, like `train_pipeline`/`test_pipeline` in datasets.
It's worth noting that when modifying intermediate variables in the children configs, users need to pass the intermediate variables into corresponding fields again.
