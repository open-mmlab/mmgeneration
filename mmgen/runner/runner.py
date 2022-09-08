import os
import sys
import time
import torch
import numpy as np
import torch.optim as optimi
import torch.utils.data as data
import torchvision.utils as tvu
import torch.utils.tensorboard as tb
from scipy import integrate
# from torchdiffeq import odeint
from tqdm.auto import tqdm


def inverse_data_transform(config, X):
    # if hasattr(config, "image_mean"):
    #     X = X + config.image_mean.to(X.device)[None, ...]

    if config['logit_transform']:
        X = torch.sigmoid(X)
    elif config['rescaled']:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)


class Runner(object):
    def __init__(self, args, config, schedule, model):
        self.args = args
        self.config = config
        self.diffusion_step = config['Schedule']['diffusion_step']
        self.sample_speed = args.sample_speed
        self.device = torch.device(args.device)

        self.schedule = schedule
        self.model = model

    def sample_fid(self):
        config = self.config['Sample']
        mpi_rank = 0
        if config['mpi4py']:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            mpi_rank = comm.Get_rank()

        model = self.model
        device = self.device
        pflow = True if self.args.method == 'PF' else False

        model.load_state_dict(torch.load(self.args.model_path, map_location=device), strict=True)
        model.eval()

        n = config['batch_size']
        total_num = config['total_num']

        skip = self.diffusion_step // self.sample_speed
        seq = range(0, self.diffusion_step, skip)
        seq_next = [-1] + list(seq[:-1])
        image_num = 0

        config = self.config['Dataset']
        if mpi_rank == 0:
            my_iter = tqdm(range(total_num // n + 1), ncols=120)
        else:
            my_iter = range(total_num // n + 1)

        for _ in my_iter:
            noise = torch.randn(n, config['channels'], config['image_size'],
                             config['image_size'], device=self.device)

            img = self.sample_image(noise, seq, model, pflow)

            img = inverse_data_transform(config, img)
            for i in range(img.shape[0]):
                if image_num+i > total_num:
                    break
                tvu.save_image(img[i], os.path.join(self.args.image_path, f"{mpi_rank}-{image_num+i}.png"))

            image_num += n

    def sample_image(self, noise, seq, model, pflow=False):
        with torch.no_grad():
            if pflow:
                shape = noise.shape
                device = self.device
                tol = 1e-5 if self.sample_speed > 1 else self.sample_speed

                def drift_func(t, x):
                    x = torch.from_numpy(x.reshape(shape)).to(device).type(torch.float32)
                    drift = self.schedule.denoising(x, None, t, model, pflow=pflow)
                    drift = drift.cpu().numpy().reshape((-1,))
                    return drift

                solution = integrate.solve_ivp(drift_func, (1, 1e-3), noise.cpu().numpy().reshape((-1,)),
                                               rtol=tol, atol=tol, method='RK45')
                img = torch.tensor(solution.y[:, -1]).reshape(shape).type(torch.float32)

            else:
                imgs = [noise]
                seq_next = [-1] + list(seq[:-1])

                start = True
                n = noise.shape[0]

                for i, j in zip(reversed(seq), reversed(seq_next)):
                    t = (torch.ones(n) * i).to(self.device)
                    t_next = (torch.ones(n) * j).to(self.device)

                    img_t = imgs[-1].to(self.device)
                    img_next = self.schedule.denoising(img_t, t_next, t, model, start, pflow)
                    start = False

                    imgs.append(img_next.to('cpu'))

                img = imgs[-1]

            return img