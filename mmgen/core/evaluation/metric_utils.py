# Copyright (c) OpenMMLab. All rights reserved.
import sys

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import is_module_wrapper

from mmgen.models.architectures.common import get_module_device
from mmgen.ops.stylegan3.ops import upfirdn2d


@torch.no_grad()
def extract_inception_features(dataloader,
                               inception,
                               num_samples,
                               inception_style='pytorch'):
    """Extract inception features for FID metric.

    Args:
        dataloader (:obj:`DataLoader`): Dataloader for images.
        inception (nn.Module): Inception network.
        num_samples (int): The number of samples to be extracted.
        inception_style (str): The style of Inception network, "pytorch" or
            "stylegan". Defaults to "pytorch".

    Returns:
        torch.Tensor: Inception features.
    """
    batch_size = dataloader.batch_size
    num_iters = num_samples // batch_size
    if num_iters * batch_size < num_samples:
        num_iters += 1
    # define mmcv progress bar
    pbar = mmcv.ProgressBar(num_iters)

    feature_list = []
    curr_iter = 1
    for data in dataloader:
        # a dirty walkround to support multiple datasets (mainly for the
        # unconditional dataset and conditional dataset). In our
        # implementation, unconditioanl dataset will return real images with
        # the key "real_img". However, the conditional dataset contains a key
        # "img" denoting the real images.
        if 'real_img' in data:
            # Mainly for the unconditional dataset in our MMGeneration
            img = data['real_img']
        else:
            # Mainly for conditional dataset in MMClassification
            img = data['img']
        pbar.update()

        # the inception network is not wrapped with module wrapper.
        if not is_module_wrapper(inception):
            # put the img to the module device
            img = img.to(get_module_device(inception))

        if inception_style == 'stylegan':
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            feature = inception(img, return_features=True)
        else:
            feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

        if curr_iter >= num_iters:
            break
        curr_iter += 1

    # Attention: the number of features may be different as you want.
    features = torch.cat(feature_list, 0)

    assert features.shape[0] >= num_samples
    features = features[:num_samples]

    # to change the line after pbar
    sys.stdout.write('\n')
    return features


def _hox_downsample(img):
    r"""Downsample images with factor equal to 0.5.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py  # noqa

    Args:
        img (ndarray): Images with order "NHWC".

    Returns:
        ndarray: Downsampled images with order "NHWC".
    """
    return (img[:, 0::2, 0::2, :] + img[:, 1::2, 0::2, :] +
            img[:, 0::2, 1::2, :] + img[:, 1::2, 1::2, :]) * 0.25


def _f_special_gauss(size, sigma):
    r"""Return a circular symmetric gaussian kernel.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py  # noqa

    Args:
        size (int): Size of Gaussian kernel.
        sigma (float): Standard deviation for Gaussian blur kernel.

    Returns:
        ndarray: Gaussian kernel.
    """
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


# Gaussian blur kernel
def get_gaussian_kernel():
    kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]],
                      np.float32) / 256.0
    gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5))
    return gaussian_k


def get_pyramid_layer(image, gaussian_k, direction='down'):
    gaussian_k = gaussian_k.to(image.device)
    if direction == 'up':
        image = F.interpolate(image, scale_factor=2)
    multiband = [
        F.conv2d(
            image[:, i:i + 1, :, :],
            gaussian_k,
            padding=2,
            stride=1 if direction == 'up' else 2) for i in range(3)
    ]
    image = torch.cat(multiband, dim=1)
    return image


def gaussian_pyramid(original, n_pyramids, gaussian_k):
    x = original
    # pyramid down
    pyramids = [original]
    for _ in range(n_pyramids):
        x = get_pyramid_layer(x, gaussian_k)
        pyramids.append(x)
    return pyramids


def laplacian_pyramid(original, n_pyramids, gaussian_k):
    """Calculate Laplacian pyramid.

    Ref: https://github.com/koshian2/swd-pytorch/blob/master/swd.py

    Args:
        original (Tensor): Batch of Images with range [0, 1] and order "NCHW"
        n_pyramids (int): Levels of pyramids minus one.
        gaussian_k (Tensor): Gaussian kernel with shape (1, 1, 5, 5).

    Return:
        list[Tensor]. Laplacian pyramids of original.
    """
    # create gaussian pyramid
    pyramids = gaussian_pyramid(original, n_pyramids, gaussian_k)

    # pyramid up - diff
    laplacian = []
    for i in range(len(pyramids) - 1):
        diff = pyramids[i] - get_pyramid_layer(pyramids[i + 1], gaussian_k,
                                               'up')
        laplacian.append(diff)
    # Add last gaussian pyramid
    laplacian.append(pyramids[len(pyramids) - 1])
    return laplacian


def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    r"""Get descriptors of one level of pyramids.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py  # noqa

    Args:
        minibatch (Tensor): Pyramids of one level with order "NCHW".
        nhood_size (int): Pixel neighborhood size.
        nhoods_per_image (int): The number of descriptors per image.

    Return:
        Tensor: Descriptors of images from one level batch.
    """
    S = minibatch.shape  # (minibatch, channel, height, width)
    assert len(S) == 4 and S[1] == 3
    N = nhoods_per_image * S[0]
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0:3, -H:H + 1, -H:H + 1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.view(-1)[idx]


def finalize_descriptors(desc):
    r"""Normalize and reshape descriptors.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py  # noqa

    Args:
        desc (list or Tensor): List of descriptors of one level.

    Return:
        Tensor: Descriptors after normalized along channel and flattened.
    """
    if isinstance(desc, list):
        desc = torch.cat(desc, dim=0)
    assert desc.ndim == 4  # (neighborhood, channel, height, width)
    desc -= torch.mean(desc, dim=(0, 2, 3), keepdim=True)
    desc /= torch.std(desc, dim=(0, 2, 3), keepdim=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc


def compute_pr_distances(row_features,
                         col_features,
                         num_gpus,
                         rank,
                         col_batch_size=10000):
    r"""Compute distances between real images and fake images.

    This function is used for calculate Precision and Recall metric.
    Refer to:https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/precision_recall.py  # noqa
    """
    assert 0 <= rank < num_gpus
    num_cols = col_features.shape[0]
    num_batches = ((num_cols - 1) // col_batch_size // num_gpus + 1) * num_gpus
    col_batches = torch.nn.functional.pad(
        col_features, [0, 0, 0, -num_cols % num_batches]).chunk(num_batches)
    dist_batches = []
    for col_batch in col_batches[rank::num_gpus]:
        dist_batch = torch.cdist(
            row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        for src in range(num_gpus):
            dist_broadcast = dist_batch.clone()
            if num_gpus > 1:
                torch.distributed.broadcast(dist_broadcast, src=src)
            dist_batches.append(dist_broadcast.cpu() if rank == 0 else None)
    return torch.cat(dist_batches, dim=1)[:, :num_cols] if rank == 0 else None


def normalize(a):
    """L2 normalization.

    Args:
        a (Tensor): Tensor with shape [N, C].

    Returns:
        Tensor: Tensor after L2 normalization per-instance.
    """
    return a / torch.norm(a, dim=1, keepdim=True)


def slerp(a, b, percent):
    """Spherical linear interpolation between two unnormalized vectors.

    Args:
        a (Tensor): Tensor with shape [N, C].
        b (Tensor): Tensor with shape [N, C].
        percent (float|Tensor): A float or tensor with shape broadcastable to
            the shape of input Tensors.

    Returns:
        Tensor: Spherical linear interpolation result with shape [N, C].
    """
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = percent * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def apply_integer_translation(x, tx, ty):
    _N, _C, H, W = x.shape
    tx = torch.as_tensor(tx * W).to(dtype=torch.float32, device=x.device)
    ty = torch.as_tensor(ty * H).to(dtype=torch.float32, device=x.device)
    ix = tx.round().to(torch.int64)
    iy = ty.round().to(torch.int64)

    z = torch.zeros_like(x)
    m = torch.zeros_like(x)
    if abs(ix) < W and abs(iy) < H:
        y = x[:, :, max(-iy, 0):H + min(-iy, 0), max(-ix, 0):W + min(-ix, 0)]
        z[:, :, max(iy, 0):H + min(iy, 0), max(ix, 0):W + min(ix, 0)] = y
        m[:, :, max(iy, 0):H + min(iy, 0), max(ix, 0):W + min(ix, 0)] = 1
    return z, m


def sinc(x):
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float('inf'))
    return torch.where(y < 1e-30, torch.ones_like(x), z)


def apply_fractional_translation(x, tx, ty, a=3):
    _N, _C, H, W = x.shape
    tx = torch.as_tensor(tx * W).to(dtype=torch.float32, device=x.device)
    ty = torch.as_tensor(ty * H).to(dtype=torch.float32, device=x.device)
    ix = tx.floor().to(torch.int64)
    iy = ty.floor().to(torch.int64)
    fx = tx - ix
    fy = ty - iy
    b = a - 1

    z = torch.zeros_like(x)
    zx0 = max(ix - b, 0)
    zy0 = max(iy - b, 0)
    zx1 = min(ix + a, 0) + W
    zy1 = min(iy + a, 0) + H
    if zx0 < zx1 and zy0 < zy1:
        taps = torch.arange(a * 2, device=x.device) - b
        filter_x = (sinc(taps - fx) * sinc((taps - fx) / a)).unsqueeze(0)
        filter_y = (sinc(taps - fy) * sinc((taps - fy) / a)).unsqueeze(1)
        y = x
        y = upfirdn2d.filter2d(
            y, filter_x / filter_x.sum(), padding=[b, a, 0, 0])
        y = upfirdn2d.filter2d(
            y, filter_y / filter_y.sum(), padding=[0, 0, b, a])
        y = y[:, :,
              max(b - iy, 0):H + b + a + min(-iy - a, 0),
              max(b - ix, 0):W + b + a + min(-ix - a, 0)]
        z[:, :, zy0:zy1, zx0:zx1] = y

    m = torch.zeros_like(x)
    mx0 = max(ix + a, 0)
    my0 = max(iy + a, 0)
    mx1 = min(ix - b, 0) + W
    my1 = min(iy - b, 0) + H
    if mx0 < mx1 and my0 < my1:
        m[:, :, my0:my1, mx0:mx1] = 1
    return z, m


def rotation_matrix(angle):
    angle = torch.as_tensor(angle).to(torch.float32)
    mat = torch.eye(3, device=angle.device)
    mat[0, 0] = angle.cos()
    mat[0, 1] = angle.sin()
    mat[1, 0] = -angle.sin()
    mat[1, 1] = angle.cos()
    return mat


def lanczos_window(x, a):
    x = x.abs() / a
    return torch.where(x < 1, sinc(x), torch.zeros_like(x))


def construct_affine_bandlimit_filter(mat,
                                      a=3,
                                      amax=16,
                                      aflt=64,
                                      up=4,
                                      cutoff_in=1,
                                      cutoff_out=1):
    assert a <= amax < aflt
    mat = torch.as_tensor(mat).to(torch.float32)

    # Construct 2D filter taps in input & output coordinate spaces.
    taps = ((torch.arange(aflt * up * 2 - 1, device=mat.device) + 1) / up -
            aflt).roll(1 - aflt * up)
    yi, xi = torch.meshgrid(taps, taps)
    xo, yo = (torch.stack([xi, yi], dim=2) @ mat[:2, :2].t()).unbind(2)

    # Convolution of two oriented 2D sinc filters.
    fin = sinc(xi * cutoff_in) * sinc(yi * cutoff_in)
    fout = sinc(xo * cutoff_out) * sinc(yo * cutoff_out)
    f = torch.fft.ifftn(torch.fft.fftn(fin) * torch.fft.fftn(fout)).real

    # Convolution of two oriented 2D Lanczos windows.
    wi = lanczos_window(xi, a) * lanczos_window(yi, a)
    wo = lanczos_window(xo, a) * lanczos_window(yo, a)
    w = torch.fft.ifftn(torch.fft.fftn(wi) * torch.fft.fftn(wo)).real

    # Construct windowed FIR filter.
    f = f * w

    # Finalize.
    c = (aflt - amax) * up
    f = f.roll([aflt * up - 1] * 2, dims=[0, 1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f,
                                [0, 1, 0, 1]).reshape(amax * 2, up, amax * 2,
                                                      up)
    f = f / f.sum([0, 2], keepdim=True) / (up**2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f


def apply_affine_transformation(x, mat, up=4, **filter_kwargs):
    _N, _C, H, W = x.shape
    mat = torch.as_tensor(mat).to(dtype=torch.float32, device=x.device)

    # Construct filter.
    f = construct_affine_bandlimit_filter(mat, up=up, **filter_kwargs)
    assert f.ndim == 2 and f.shape[0] == f.shape[1] and f.shape[0] % 2 == 1
    p = f.shape[0] // 2

    # Construct sampling grid.
    theta = mat.inverse()
    theta[:2, 2] *= 2
    theta[0, 2] += 1 / up / W
    theta[1, 2] += 1 / up / H
    theta[0, :] *= W / (W + p / up * 2)
    theta[1, :] *= H / (H + p / up * 2)
    theta = theta[:2, :3].unsqueeze(0).repeat([x.shape[0], 1, 1])
    g = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)

    # Resample image.
    y = upfirdn2d.upsample2d(x=x, f=f, up=up, padding=p)
    z = torch.nn.functional.grid_sample(
        y, g, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Form mask.
    m = torch.zeros_like(y)
    c = p * 2 + 1
    m[:, :, c:-c, c:-c] = 1
    m = torch.nn.functional.grid_sample(
        m, g, mode='nearest', padding_mode='zeros', align_corners=False)
    return z, m


def apply_fractional_rotation(x, angle, a=3, **filter_kwargs):
    angle = torch.as_tensor(angle).to(dtype=torch.float32, device=x.device)
    mat = rotation_matrix(angle)
    return apply_affine_transformation(
        x, mat, a=a, amax=a * 2, **filter_kwargs)


def apply_fractional_pseudo_rotation(x, angle, a=3, **filter_kwargs):
    angle = torch.as_tensor(angle).to(dtype=torch.float32, device=x.device)
    mat = rotation_matrix(-angle)
    f = construct_affine_bandlimit_filter(
        mat, a=a, amax=a * 2, up=1, **filter_kwargs)
    y = upfirdn2d.filter2d(x=x, f=f)
    m = torch.zeros_like(y)
    c = f.shape[0] // 2
    m[:, :, c:-c, c:-c] = 1
    return y, m
