# FAQ

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmgeneration/blob/master/.github/ISSUE_TEMPLATE/error-report.md) and make sure you fill in all required information in the template.

(1) Q: Why `MMCV==xxx is used but incompatible` is raised when import I try to import `mmgen`?

This is because the version of MMCV and MMGeneration are incompatible. Compatible MMGeneration and MMCV versions are shown as below. Please choose the correct version of MMCV to avoid installation issues.

| MMGeneration version |   MMCV version   |
| :------------------: | :--------------: |
|        master        | mmcv-full>=2.0.0 |

Note: You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

(2) Q: How can I ignore some fields in the base configs?

Sometimes, you may set `_delete_=True` to ignore some of fields in base configs.
You may refer to [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/config.md#%E5%88%A0%E9%99%A4%E5%AD%97%E5%85%B8%E4%B8%AD%E7%9A%84-key) for simple illustration.

You may have a careful look at [this tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/config.md) for better understanding of this feature.

(3) Q: How can I use intermediate variables in configs?

Some intermediate variables are used in the config files, like `train_pipeline`/`test_pipeline` in datasets.
It's worth noting that when modifying intermediate variables in the children configs, users need to pass the intermediate variables into corresponding fields again.
