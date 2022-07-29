import argparse
import os
import os.path as osp
import re
from collections import OrderedDict
from pathlib import Path

from modelindex.load_model_index import load

MMGEN_ROOT = Path(__file__).absolute().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description='Download the checkpoints')
    parser.add_argument('checkpoint_root', help='Checkpoint file root path.')
    parser.add_argument(
        '--models', nargs='+', type=str, help='Specify model names to run.')

    args = parser.parse_args()
    return args


def download(args):
    # parse model-index.yml
    model_index_file = MMGEN_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    http_prefix = 'https://download.openmmlab.com/mmgen/'

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
        # model_weight = model_info.weights
        model_weight_url = model_info.weights
        model_weight = model_weight_url.replace(http_prefix, '')
        # print(model_weight)
        model_weight = osp.join(args.checkpoint_root, model_weight)
        print(model_weight)
        if osp.exists(model_weight):
            print(f'Already exists {model_weight}')
            continue
        os.system(f'wget -P {args.checkpoint_root} {model_weight}')


if __name__ == '__main__':
    args = parse_args()
    download(args)
