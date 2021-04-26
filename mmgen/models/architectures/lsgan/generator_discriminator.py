import torch
import torch.nn as nn

from mmgen.models.builder import MODULES
from ..common import get_module_device


@MODULES.register_module()
class LSGANGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.noise_size = 1024
        self.linear1 = nn.Sequential(nn.Linear(1024, 7 * 7 * 256))
        self.noise2feat_tail = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU())

        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    256, 256, 3, stride=2, output_padding=1, padding=1),
                nn.BatchNorm2d(256), nn.ReLU()))
        self.conv_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256), nn.ReLU()))

        self.conv_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    256, 256, 3, stride=2, output_padding=1, padding=1),
                nn.BatchNorm2d(256), nn.ReLU()))
        self.conv_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256), nn.ReLU()))

        self.conv_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    256, 128, 3, stride=2, output_padding=1, padding=1),
                nn.BatchNorm2d(128), nn.ReLU()))
        self.conv_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    128, 64, 3, stride=2, output_padding=1, padding=1),
                nn.BatchNorm2d(64), nn.ReLU()))
        self.conv_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1), nn.Tanh()))

    def forward(self, noise, num_batches=0, return_noise=False):
        """Forward function.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.

        Returns:
            torch.Tensor | dict: If not ``return_noise``, only the output image
                will be returned. Otherwise, a dict contains ``fake_img`` and
                ``noise_batch`` will be returned.
        """
        # receive noise and conduct sanity check.
        if isinstance(noise, torch.Tensor):
            assert noise.shape[1] == self.noise_size
            if noise.ndim == 2:
                noise_batch = noise
            else:
                raise ValueError('The noise should be in shape of (n, c)'
                                 f'but got {noise.shape}')
        # receive a noise generator and sample noise.
        elif callable(noise):
            noise_generator = noise
            assert num_batches > 0
            noise_batch = noise_generator((num_batches, self.noise_size))
        # otherwise, we will adopt default noise sampler.
        else:
            assert num_batches > 0
            noise_batch = torch.randn((num_batches, self.noise_size))

        # dirty code for putting data on the right device
        noise_batch = noise_batch.to(get_module_device(self))
        # noise2feat
        x = self.linear1(noise_batch)
        x = x.reshape((-1, 256, 7, 7))
        x = self.noise2feat_tail(x)
        # conv module
        for conv in self.conv_blocks:
            x = conv(x)

        if return_noise:
            return dict(fake_img=x, noise_batch=noise_batch)

        return x


@MODULES.register_module()
class LSGANDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(
            nn.Sequential(
                nn.Conv2d(3, 64, 5, stride=2, padding=2), nn.LeakyReLU(0.2)))
        self.conv_blocks.append(
            nn.Sequential(
                nn.Conv2d(64, 128, 5, stride=2, padding=2),
                nn.BatchNorm2d(128), nn.LeakyReLU(0.2)))
        self.conv_blocks.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 5, stride=2, padding=2),
                nn.BatchNorm2d(256), nn.LeakyReLU(0.2)))
        self.conv_blocks.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 5, stride=2, padding=2),
                nn.BatchNorm2d(512), nn.LeakyReLU(0.2)))
        self.decision = nn.Sequential(nn.Linear(7 * 7 * 512, 1), nn.Sigmoid())

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Fake or real image tensor.

        Returns:
            torch.Tensor: Prediction for the reality of the input image.
        """
        n = x.shape[0]
        for conv in self.conv_blocks:
            x = conv(x)
        x = x.reshape(n, -1)
        x = self.decision(x)

        # reshape to a flatten feature
        return x
