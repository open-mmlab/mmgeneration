# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn.bricks import build_conv_layer
from torch import einsum


@MODULES.register_module()
class VectorQuantizer(nn.Module):
    """Discretization bottleneck part of the VQ-VAE.

    Ref:
        https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
        https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py

    Args:
        in_channels (torch.FloatTensor): The channel number of the input feature map.
        e_channels (torch.FloatTensor): The channel number of the embedding.
        beta (float): Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2.
    """

    def __init__(self, in_channels, e_channels, beta):
        super().__init__()
        self.in_channels = in_channels
        self.e_channels = e_channels
        self.beta = beta

        self.embedding = nn.Embedding(self.in_channels, self.e_channels)
        self.embedding.weight.data.uniform_(-1.0 / self.in_channels,
                                            1.0 / self.in_channels)

    def forward(self, z):
        """Forward function for the encoder network.

        Args:
            z (torch.FloatTensor): Input latent vectors.

        Returns:
            z_q (torch.FloatTensor): Quantized latent vectors.
            loss (torch.FloatTensor): The loss value.
            perplexity (torch.FloatTensor): The perplexity of the network.
            min_encodings (torch.FloatTensor): The closest embedding vector.
            min_encoding_indices (torch.Tensor): The indices of the closest embedding vector.
        """
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_channels)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings and get quantized latent vectors
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0],
                                    self.in_channels).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # compute perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        """Check for more easy handling with nn.Embedding.

        Args:
            indices (torch.Tensor): The index of the embedding.
            shape (tuple): The shape of the embedding.

        Returns:
            z_q (torch.FloatTensor): Quantized latent vectors.
        """
        min_encodings = torch.zeros(indices.shape[0],
                                    self.in_channels).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


@MODULES.register_module()
class VectorQuantizer2(nn.Module):
    """Improved version over VectorQuantizer, can be used as a drop-in
    replacement.

    Ref:
        https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
        https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py

    Args:
        in_channels (torch.FloatTensor): The channel number of the input feature map.
        e_channels (torch.FloatTensor): The channel number of the embeddings.
        beta (float): Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2.
        remap (numpy.Float, optional): Remapped embeddings. Defaults to None.
        unknown_index (str, optional): How to deal with the unknown index. Defaults to "random".
        sane_index_shape (bool, optional): Whether to reshape the indices of the closest embedding.
                                            Defaults to False.
        legacy (bool, optional): Whether to use the buggy version. Defaults to True.
    """

    def __init__(self,
                 in_channels,
                 e_channels,
                 beta,
                 remap=None,
                 unknown_index="random",
                 sane_index_shape=False,
                 legacy=True):
        super().__init__()
        self.in_channels = in_channels
        self.e_channels = e_channels
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.in_channels, self.e_channels)
        self.embedding.weight.data.uniform_(-1.0 / self.in_channels,
                                            1.0 / self.in_channels)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.in_channels} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = in_channels

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        """Remap the indices.

        Args:
            inds (torch.Tensor): The indices of the embeddings.

        Returns:
            (torch.Tensor): remapped indices.
        """
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(
                0, self.re_embed,
                size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        """Unmap the indices.

        Args:
            inds (torch.Tensor): The indices of the embeddings.

        Returns:
            (torch.Tensor): unmapped indices.
        """
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        """Forward function for the encoder network.

        Args:
            z (torch.FloatTensor): Input latent vectors.
            temp (float, optional): The temperature decay for Gumbel quantizer. Defaults to None.
            rescale_logits (bool, optional): Whether to rescale the logits. Defaults to False.
            return_logits (bool, optional): Return the logits or not. Defaults to False.

        Returns:
            z_q (torch.FloatTensor): Quantized latent vectors.
            loss (torch.FloatTensor): The loss value.
            perplexity (torch.FloatTensor): The perplexity of the network.
            min_encodings (torch.FloatTensor): The closest embedding.
            min_encoding_indices (torch.Tensor): The indices of the closest embedding.
        """
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"

        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_channels)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # reshape back to match original input shape
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(
                z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,
                                                                1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        """Check for more easy handling with nn.Embedding.

        Args:
            indices (torch.Tensor): The index of the embedding.
            shape (tuple): The shape of the embedding.

        Returns:
            z_q (torch.FloatTensor): Quantized latent vectors.
        """
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


@MODULES.register_module()
class GumbelQuantize(nn.Module):
    """Gumbel Softmax trick quantizer Categorical Reparameterization with
    Gumbel-Softmax, Jang et al. 2016 https://arxiv.org/abs/1611.01144.

    Ref:
        https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/model/quantize.py
        https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py

    Args:
        num_hiddens (int): The channel number of the hidden layer.
        in_channels (int): The channel number of the input feature map.
        e_channels (int): The channel number of the embedding.
        straight_through (bool, optional): Whether is eval mode. Defaults to True.
        kl_weight (float, optional): The weight of kl divergence. Defaults to 5e-4.
        temp_init (float, optional): The temperature decay for Gumbel quantizer. Defaults to 1.0.
        use_vqinterface (bool, optional): Whether to use different return formats. Defaults to True.
        remap (bool, optional): Whether to remap the embeddings. Defaults to None.
        unknown_index (str, optional): How to deal with the unknown index. Defaults to "random".
    """

    def __init__(self,
                 num_hiddens,
                 in_channels,
                 e_channels,
                 straight_through=True,
                 kl_weight=5e-4,
                 temp_init=1.0,
                 use_vqinterface=True,
                 remap=None,
                 unknown_index="random"):
        super().__init__()

        self.in_channels = in_channels
        self.e_channels = e_channels
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.proj = build_conv_layer(None, num_hiddens, in_channels, 1)
        self.embed = nn.Embedding(in_channels, e_channels)
        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.in_channels} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = in_channels

    def remap_to_used(self, inds):
        """Remap the indices.

        Args:
            inds (torch.Tensor): The indices of the embeddings.

        Returns:
            (torch.Tensor): remapped indices.
        """
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(
                0, self.re_embed,
                size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        """Unmap the indices.

        Args:
            inds (torch.Tensor): The indices of the embeddings.

        Returns:
            (torch.Tensor): unmapped indices.
        """
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        """Forward function for the encoder network.

        Args:
            z (torch.FloatTensor): Input latent vectors.
            temp (float, optional): The temperature decay for Gumbel quantizer. 
                                    Defaults to None.
            return_logits (bool, optional): Return the logits or not. 
                                            Defaults to False.

        Returns:
            z_q (torch.FloatTensor): Quantized latent vectors.
            loss (torch.FloatTensor): The loss value.
            inds (torch.Tensor): The indices of the embeddings.
        """

        # force hard = True when we are in eval mode
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:, self.used, ...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:, self.used, ...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot,
                     self.embed.weight)

        # add kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        loss = self.kl_weight * torch.sum(
            qy * torch.log(qy * self.in_channels + 1e-10), dim=1).mean()

        inds = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            inds = self.remap_to_used(inds)
        if self.use_vqinterface:
            if return_logits:
                return z_q, loss, (None, None, inds), logits
            return z_q, loss, (None, None, inds)
        return z_q, loss, inds

    def get_codebook_entry(self, indices, shape):
        """Check for more easy handling with nn.Embedding.

        Args:
            indices (torch.Tensor): The index of the embedding.
            shape (tuple): The shape of the embedding.

        Returns:
            z_q (torch.FloatTensor): Quantized latent vectors.
        """
        b, h, w, c = shape
        assert b * h * w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices,
                            num_classes=self.in_channels).permute(0, 3, 1,
                                                                  2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)

        return z_q
