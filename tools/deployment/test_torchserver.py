# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import numpy as np
import requests
from PIL import Image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('model_name', help='The model name in the server')
    parser.add_argument(
        '--inference-addr',
        default='127.0.0.1:8080',
        help='Address and port of the inference server')
    parser.add_argument(
        '--img-path',
        type=str,
        default='demo.png',
        help='Path to save generated image.')
    parser.add_argument(
        '--img-size', type=int, default=128, help='Size of the output image.')
    parser.add_argument(
        '--sample-model',
        type=str,
        default='ema/orig',
        help='Which model you want to use.')
    args = parser.parse_args()
    return args


def save_results(contents, img_path, img_size):
    if not isinstance(contents, list):
        Image.frombytes('RGB', (img_size, img_size), contents).save(img_path)
        return

    imgs = []
    for content in contents:
        imgs.append(
            np.array(Image.frombytes('RGB', (img_size, img_size), content)))
    Image.fromarray(np.concatenate(imgs, axis=1)).save(img_path)


def main(args):
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name

    if args.sample_model == 'ema/orig':
        cont_ema = requests.post(url, {'sample_model': 'ema'}).content
        cont_orig = requests.post(url, {'sample_model': 'orig'}).content
        save_results([cont_ema, cont_orig], args.img_path, args.img_size)
        return

    response = requests.post(url, {'sample_model': args.sample_model})
    save_results(response.content, args.img_path, args.img_size)


if __name__ == '__main__':
    args = parse_args()
    main(args)
