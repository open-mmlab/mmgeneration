# FAQ

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmgeneration/blob/master/.github/ISSUE_TEMPLATE/error-report.md) and make sure you fill in all required information in the template.

## Installation

- Compatible MMGeneration and MMCV versions are shown as below. Please choose the correct version of MMCV to avoid installation issues.

| MMGeneration version |   MMCV version   |
| :------------------: | :--------------: |
|        master        | mmcv-full>=1.3.0 |

Note: You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.
