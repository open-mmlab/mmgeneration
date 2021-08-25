# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from .pretrained_networks import vgg16


def normalize_tensor(in_feat, eps=1e-10):
    """L2 normalization.

    Args:
        in_feat (Tensor): Tensor with shape [N, C, H, W].
        eps (float, optional): Epsilon value to avoid computation error.
            Defaults to 1e-10.

    Returns:
        Tensor: Tensor after L2 normalization per-instance.
    """
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def spatial_average(in_tens, keepdim=True):
    """Returns the mean value of each row of the input tensor in the spatial
    dimension.

    Args:
        in_tens (Tensor): Tensor with shape [N, C, H, W].
        keepdim (bool, optional): If keepdim is True, the output tensor is of
            the shape [N, C, 1, 1]. Otherwise, the output will have shape
            [N, C]. Defaults to True.

    Returns:
        Tensor: Tensor after average pooling to 1x1 with shape [N, C, 1, 1] or
            [N, C].
    """
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_H=64):  # assumes scale factor is same for H and W
    """Upsamples the input to the given size.

    Args:
        in_tens (Tensor): Tensor with shape [N, C, H, W].
        out_H (int, optional): Output spatial size. Defaults to 64.

    Returns:
        Tensor: Output Tensor.
    """
    in_H = in_tens.shape[2]
    scale_factor = 1. * out_H / in_H

    return nn.Upsample(
        scale_factor=scale_factor, mode='bilinear', align_corners=False)(
            in_tens)


# Learned perceptual metric
class PNetLin(nn.Module):
    r"""
        Ref: https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py # noqa
    """

    def __init__(self,
                 pnet_rand=False,
                 pnet_tune=False,
                 use_dropout=True,
                 spatial=False,
                 version='0.1',
                 lpips=True):
        super().__init__()

        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.version = version
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]
        self.L = len(self.channels)
        self.net = vgg16(
            pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        self.lin0 = NetLinLayer(self.channels[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.channels[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.channels[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.channels[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.channels[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

    def forward(self, in0, in1, retPerLayer=False):
        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            self.scaling_layer(in0),
            self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(
                outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk])**2

        if self.lpips:
            if self.spatial:
                res = [
                    upsample(
                        self.lins[kk].model(diffs[kk]), out_H=in0.shape[2])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    spatial_average(
                        self.lins[kk].model(diffs[kk]), keepdim=True)
                    for kk in range(self.L)
                ]
        else:
            if self.spatial:
                res = [
                    upsample(
                        diffs[kk].sum(dim=1, keepdim=True), out_H=in0.shape[2])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    spatial_average(
                        diffs[kk].sum(dim=1, keepdim=True), keepdim=True)
                    for kk in range(self.L)
                ]

        val = sum(res)
        if retPerLayer:
            return (val, res)

        return val


class ScalingLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer(
            'shift',
            torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'scale',
            torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv."""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super().__init__()

        layers = [
            nn.Dropout(),
        ] if (use_dropout) else []
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class Dist2LogitLayer(nn.Module):
    """takes 2 distances, puts through fc layers, spits out value between [0,
    1] (if use_sigmoid is True)"""

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super().__init__()

        layers = [
            nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),
        ]
        layers += [
            nn.LeakyReLU(0.2, True),
        ]
        layers += [
            nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),
        ]
        layers += [
            nn.LeakyReLU(0.2, True),
        ]
        layers += [
            nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),
        ]
        if use_sigmoid:
            layers += [
                nn.Sigmoid(),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(
            torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)),
                      dim=1))


class BCERankingLoss(nn.Module):

    def __init__(self, chn_mid=32):
        super().__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.) / 2.
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)
