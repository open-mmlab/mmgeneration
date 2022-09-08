import os

import yaml
import argparse
from mpi4py import MPI
import torch
from mmgen.runner.schedule import Schedule
from mmgen.runner.runner import Runner
from mmgen.models.diffusions.implicit_model import Model

def parse_args():
    parser = argparse.ArgumentParser(description='PPIM demo')

    parser.add_argument('config',
                        type=str,
                        default= '../configs/ddim/ddim_cifar10.yml',
                        help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos/ddim_samples.png',
        help='path to save unconditional samples')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')

    # args for inference/sampling
    parser.add_argument(
        '--sample_speed',
        type=int,
        default=50,
        help='control the total generation step')
    parser.add_argument(
        '--num-samples', type=int, default=12, help='the total number of samples')
    parser.add_argument(
        '--sample-model',
        type=str,
        default='generalized',
        help='sampling approach (generalized or ddpm_noisy)')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(mpi_rank)
    device = torch.device(args.device)
    schedule = Schedule(args, config['Schedule'])
    model = Model(args, config['Model']).to(device)

    runner = Runner(args, config, schedule, model)

    runner.sample_fid()