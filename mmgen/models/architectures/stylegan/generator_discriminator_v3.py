import torch
import torch.nn as nn

from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES
from .modules import MappingNetwork, SynthesisNetwork
from .utils import get_mean_latent


@MODULES.register_module()
class StyleGANv3Generator(nn.Module):

    def __init__(
            self,
            z_dim,
            c_dim,
            style_channels,
            out_size,
            img_channels,
            rgb2bgr=False,
            mapping_kwargs=dict(),
            **synthesis_kwargs,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.style_channels = style_channels
        self.out_size = out_size
        self.img_channels = img_channels
        self.rgb2bgr = rgb2bgr
        self.synthesis = SynthesisNetwork(
            style_channels=style_channels,
            out_size=out_size,
            img_channels=img_channels,
            **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.style_mapping = MappingNetwork(
            z_dim=z_dim,
            c_dim=c_dim,
            style_channels=style_channels,
            num_ws=self.num_ws,
            **mapping_kwargs)

    def forward(self,
                noise,
                label=None,
                num_batches=0,
                input_is_latent=False,
                truncation_psi=1,
                truncation_cutoff=None,
                truncation_latent=None,
                update_emas=False,
                return_noise=False,
                return_latent=False,
                **synthesis_kwargs):

        if isinstance(noise, torch.Tensor):
            assert noise.shape[1] == self.z_dim
            assert noise.ndim == 2, ('The noise should be in shape of (n, c), '
                                     f'but got {noise.shape}')
            noise_batch = noise

        # receive a noise generator and sample noise.
        elif callable(noise):
            noise_generator = noise
            assert num_batches > 0
            noise_batch = noise_generator((num_batches, self.z_dim))

        # otherwise, we will adopt default noise sampler.
        else:
            assert num_batches > 0
            noise_batch = torch.randn((num_batches, self.z_dim))

        if self.c_dim == 0:
            label_batch = None

        elif isinstance(label, torch.Tensor):
            label_batch = label
        elif callable(label):
            label_generator = label
            assert num_batches > 0
            label_batch = label_generator((num_batches, ))
        else:
            assert num_batches > 0
            label_batch = torch.randint(0, self.c_dim, (num_batches, ))

        device = get_module_device(self)
        noise_batch = noise_batch.to(device)

        if label_batch:
            label_batch = label_batch.to(device)

        ws = self.style_mapping(
            noise_batch,
            label_batch,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            truncation_latent=truncation_latent,
            update_emas=update_emas)
        out_img = self.synthesis(
            ws, update_emas=update_emas, **synthesis_kwargs)

        if self.rgb2bgr:
            out_img = out_img[:, [2, 1, 0], ...]

        if return_noise or return_latent:
            output = dict(
                fake_img=out_img,
                noise_batch=noise_batch,
                label=label_batch,
                latent=ws)
            return output

        return out_img

    def get_mean_latent(self, num_samples=4096, **kwargs):
        """Get mean latent of W space in this generator.

        Args:
            num_samples (int, optional): Number of sample times. Defaults
                to 4096.

        Returns:
            Tensor: Mean latent of this generator.
        """
        return get_mean_latent(self, num_samples, **kwargs)