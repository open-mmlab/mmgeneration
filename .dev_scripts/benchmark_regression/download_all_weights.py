import argparse
import os
import os.path as osp
import re
from collections import OrderedDict
from importlib.machinery import SourceFileLoader
from pathlib import Path

from modelindex.load_model_index import load

MMGEN_ROOT = Path(__file__).absolute().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description='Download the checkpoints')
    parser.add_argument('checkpoint_root', help='Checkpoint file root path.')
    parser.add_argument(
        '--models', nargs='+', type=str, help='Specify model names to run.')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Whether force re-download the checkpoints.')
    parser.add_argument(
        '--model-list', type=str, help='Path of algorithm list to download')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only show download command but not run')

    args = parser.parse_args()
    return args


def download(args):
    # parse model-index.yml
    model_index_file = MMGEN_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    http_prefix = 'https://download.openmmlab.com/mmgen/'

    # load model list
    if args.model_list:
        file_list_path = args.model_list
        file_list = SourceFileLoader('model_list',
                                     file_list_path).load_module().model_list
    else:
        file_list = None

    if args.models:
        patterns = [re.compile(pattern) for pattern in args.models]
        filter_models = {}
        for k, v in models.items():
            if any([re.match(pattern, k) for pattern in patterns]):
                filter_models[k] = v
        if len(filter_models) == 0:
            print('No model found, please specify models in:')
            print('\n'.join(models.keys()))
            return
        models = filter_models

    for model_info in models.values():

        if file_list is not None and model_info.name not in file_list:
            continue

        model_weight_url = model_info.weights

        # use [:-1] because download url for translation is
        # 'METHOD/refactor/NAME.pth', use [:-1] can remain
        # `refactor` in `model_name` (`refactor/NAME.pth`)
        model_name = osp.join(
            *model_weight_url[len(http_prefix):].split('/')[:-1])
        download_path = osp.join(args.checkpoint_root, model_name)
        if osp.exists(download_path):
            print(f'Already exists {download_path}')
            # do not delete when dry-run is true
            if args.force and not args.dry_run:
                print(f'Delete {download_path} to force re-download.')
                os.system(f'rm -rf {download_path}')
            else:
                continue
        try:
            cmd_str = (f'wget -q --show-progress -P {download_path} '
                       f'{model_weight_url}')

            if args.dry_run:
                print(cmd_str)
            else:
                os.system(cmd_str)
        except Exception:
            # for older version of wget
            cmd_str = (f'wget -P {download_path} {model_weight_url}')
            if args.dry_run:
                print(cmd_str)
            else:
                os.system(cmd_str)


if __name__ == '__main__':
    args = parse_args()
    download(args)
