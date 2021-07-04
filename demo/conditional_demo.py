import argparse
import os
import sys

import mmcv
from mmcv import DictAction
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import init_model, sample_conditional_model  # isort:skip  # noqa
# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Generation demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos/conditional_samples.png',
        help='path to save uncoditional samples')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')

    # args for inference/sampling
    parser.add_argument(
        '--num-batches', type=int, default=4, help='Batch size in inference')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=12,
        help='The total number of samples')
    parser.add_argument(
        '--label', type=int, nargs='+', help='Labels used for sample.')
    parser.add_argument(
        '--sample-model',
        type=str,
        default='ema',
        help='Which model to use for sampling')
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')

    # args for classes sampling
    parser.add_argument(
        '--sample-all-classes',
        action='store_true',
        help='Whether sample all classes of the dataset.')
    parser.add_argument(
        '--samples-per_classes',
        type=int,
        default=5,
        help='Number of samples to generate in each classes.')

    # args for image grid
    parser.add_argument(
        '--padding', type=int, default=0, help='Padding in the image grid.')
    parser.add_argument(
        '--nrow',
        type=int,
        default=6,
        help='Number of images displayed in each row of the grid')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = init_model(
        args.config, checkpoint=args.checkpoint, device=args.device)

    if args.sample_cfg is None:
        args.sample_cfg = dict()

    if args.sample_all_classes:
        mmcv.print_log(
            '`sample_all_classes` is set as True, `num_samples`, `label`, '
            'and `nrows` would be ignored.', 'mmgen')

        # get num_classes
        if model.num_classes is not None:
            num_classes = model.num_classes
        elif model.generator.num_classes is not None:
            num_classes = model.generator.num_classes
        elif model.discriminator.num_classes is not None:
            num_classes = model.discriminator.num_classes
        else:
            raise AttributeError(
                'Cannot get attribute `num_classes` from '
                f'{type(model)}, {type(model.generator)} '
                f'and {type(model.discriminator)}. Please '
                'check your config.', 'mmgen')
        mmcv.print_log(f'Set `nrows` as number of classes (={num_classes}).',
                       'mmgen')

        # build label list
        label = []
        for idx in range(num_classes):
            label += [idx] * args.samples_per_classes
        num_samples = len(label)
        nrow = num_classes
    else:
        num_samples = args.num_samples
        label = args.label
        nrow = args.nrow

    results = sample_conditional_model(model, num_samples, args.num_batches,
                                       args.sample_model, label,
                                       **args.sample_cfg)
    results = (results[:, [2, 1, 0]] + 1.) / 2.

    # save images
    mmcv.mkdir_or_exist(os.path.dirname(args.save_path))
    utils.save_image(results, args.save_path, nrow=nrow, padding=args.padding)


if __name__ == '__main__':
    main()
