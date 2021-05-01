# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License for StyleGAN2 with Adaptive Discriminator
# Augmentation (ADA)
# =======================================================================

# 1. Definitions

# "Licensor" means any person or entity that distributes its Work.

# "Software" means the original work of authorship made available under
# this License.

# "Work" means the Software and any additions to or derivative works of
# the Software that are made available under this License.

# The terms "reproduce," "reproduction," "derivative works," and
# "distribution" have the meaning as provided under U.S. copyright law;
# provided, however, that for the purposes of this License, derivative
# works shall not include works that remain separable from, or merely
# link (or bind by name) to the interfaces of, the Work.

# Works, including the Software, are "made available" under this License
# by including in or with the Work either (a) a copyright notice
# referencing the applicability of this License to the Work, or (b) a
# copy of this License.

# 2. License Grants

#     2.1 Copyright Grant. Subject to the terms and conditions of this
#     License, each Licensor grants to you a perpetual, worldwide,
#     non-exclusive, royalty-free, copyright license to reproduce,
#     prepare derivative works of, publicly display, publicly perform,
#     sublicense and distribute its Work and any resulting derivative
#     works in any form.

# 3. Limitations

#     3.1 Redistribution. You may reproduce or distribute the Work only
#     if (a) you do so under this License, (b) you include a complete
#     copy of this License with your distribution, and (c) you retain
#     without modification any copyright, patent, trademark, or
#     attribution notices that are present in the Work.

#     3.2 Derivative Works. You may specify that additional or different
#     terms apply to the use, reproduction, and distribution of your
#     derivative works of the Work ("Your Terms") only if (a) Your Terms
#     provide that the use limitation in Section 3.3 applies to your
#     derivative works, and (b) you identify the specific derivative
#     works that are subject to Your Terms. Notwithstanding Your Terms,
#     this License (including the redistribution requirements in Section
#     3.1) will continue to apply to the Work itself.

#     3.3 Use Limitation. The Work and any derivative works thereof only
#     may be used or intended for use non-commercially. Notwithstanding
#     the foregoing, NVIDIA and its affiliates may use the Work and any
#     derivative works commercially. As used herein, "non-commercially"
#     means for research or evaluation purposes only.

#     3.4 Patent Claims. If you bring or threaten to bring a patent claim
#     against any Licensor (including any claim, cross-claim or
#     counterclaim in a lawsuit) to enforce any patents that you allege
#     are infringed by any Work, then your rights under this License from
#     such Licensor (including the grant in Section 2.1) will terminate
#     immediately.

#     3.5 Trademarks. This License does not grant any rights to use any
#     Licensor’s or its affiliates’ names, logos, or trademarks, except
#     as necessary to reproduce the notices described in this License.

#     3.6 Termination. If you violate any term of this License, then your
#     rights under this License (including the grant in Section 2.1) will
#     terminate immediately.

# 4. Disclaimer of Warranty.

# THE WORK IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR
# NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER
# THIS LICENSE.

# 5. Limitation of Liability.

# EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL
# THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE
# SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
# OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK
# (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION,
# LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER
# COMMERCIAL DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGES.

# =======================================================================
"""Custom replacement for `torch.nn.functional.conv2d` that supports
arbitrarily high order gradients with zero performance penalty.

We modify the original implementation from Tero for supporting multiple
versions of PyTorch.
"""

import contextlib
import warnings

import torch

# pylint: disable=redefined-builtin
# pylint: disable=arguments-differ
# pylint: disable=protected-access

# Forcefully disable computation of gradients with respect to the weights.
weight_gradients_disabled = False


@contextlib.contextmanager
def no_weight_gradients():
    global weight_gradients_disabled
    old = weight_gradients_disabled
    weight_gradients_disabled = True
    yield
    weight_gradients_disabled = old


def conv2d(input,
           weight,
           bias=None,
           stride=1,
           padding=0,
           dilation=1,
           groups=1):
    if _should_use_custom_op(input):
        return _conv2d_gradfix(
            transpose=False,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=0,
            dilation=dilation,
            groups=groups).apply(input, weight, bias)
    return torch.nn.functional.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups)


def conv_transpose2d(input,
                     weight,
                     bias=None,
                     stride=1,
                     padding=0,
                     output_padding=0,
                     groups=1,
                     dilation=1):
    if _should_use_custom_op(input):
        return _conv2d_gradfix(
            transpose=True,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation).apply(input, weight, bias)
    return torch.nn.functional.conv_transpose2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation)


def _should_use_custom_op(input, enabled=True):
    assert isinstance(input, torch.Tensor)
    if (not enabled) or (not torch.backends.cudnn.enabled):
        print('CUDNN not enable.')
        return False
    if input.device.type != 'cuda':
        return False
    if any(
            torch.__version__.startswith(x)
            for x in ['1.5.', '1.6.', '1.7.', '1.8.']):
        return True
    warnings.warn(
        f'conv2d_gradfix not supported on PyTorch {torch.__version__}. '
        'Falling back to torch.nn.functional.conv2d().')
    return False


def _tuple_of_ints(xs, ndim):
    xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs, ) * ndim
    assert len(xs) == ndim
    assert all(isinstance(x, int) for x in xs)
    return xs


_conv2d_gradfix_cache = dict()


def _conv2d_gradfix(transpose, weight_shape, stride, padding, output_padding,
                    dilation, groups):
    # Parse arguments.
    ndim = 2
    weight_shape = tuple(weight_shape)
    stride = _tuple_of_ints(stride, ndim)
    padding = _tuple_of_ints(padding, ndim)
    output_padding = _tuple_of_ints(output_padding, ndim)
    dilation = _tuple_of_ints(dilation, ndim)

    # Lookup from cache.
    key = (transpose, weight_shape, stride, padding, output_padding, dilation,
           groups)
    if key in _conv2d_gradfix_cache:
        return _conv2d_gradfix_cache[key]

    # Validate arguments.
    assert groups >= 1
    assert len(weight_shape) == ndim + 2
    assert all(stride[i] >= 1 for i in range(ndim))
    assert all(padding[i] >= 0 for i in range(ndim))
    assert all(dilation[i] >= 0 for i in range(ndim))
    if not transpose:
        assert all(output_padding[i] == 0 for i in range(ndim))
    else:  # transpose
        assert all(0 <= output_padding[i] < max(stride[i], dilation[i])
                   for i in range(ndim))

    # Helpers.
    common_kwargs = dict(
        stride=stride, padding=padding, dilation=dilation, groups=groups)

    def calc_output_padding(input_shape, output_shape):
        if transpose:
            return [0, 0]
        return [
            input_shape[i + 2] - (output_shape[i + 2] - 1) * stride[i] -
            (1 - 2 * padding[i]) - dilation[i] * (weight_shape[i + 2] - 1)
            for i in range(ndim)
        ]

    # Forward & backward.
    class Conv2d(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input, weight, bias):
            assert weight.shape == weight_shape
            # keep the consistent datatype for FP16 training
            weight = weight.to(input.dtype)
            if not transpose:
                output = torch.nn.functional.conv2d(
                    input=input, weight=weight, bias=bias, **common_kwargs)
            else:  # transpose
                output = torch.nn.functional.conv_transpose2d(
                    input=input,
                    weight=weight,
                    bias=bias,
                    output_padding=output_padding,
                    **common_kwargs)
            ctx.save_for_backward(input, weight)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            grad_input = None
            grad_weight = None
            grad_bias = None

            if ctx.needs_input_grad[0]:
                p = calc_output_padding(
                    input_shape=input.shape, output_shape=grad_output.shape)
                grad_input = _conv2d_gradfix(
                    transpose=(not transpose),
                    weight_shape=weight_shape,
                    output_padding=p,
                    **common_kwargs).apply(grad_output, weight, None)
                assert grad_input.shape == input.shape

            if ctx.needs_input_grad[1] and not weight_gradients_disabled:
                grad_weight = Conv2dGradWeight.apply(grad_output, input)
                assert grad_weight.shape == weight_shape

            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum([0, 2, 3])

            return grad_input, grad_weight, grad_bias

    # Gradient with respect to the weights.
    class Conv2dGradWeight(torch.autograd.Function):

        @staticmethod
        def forward(ctx, grad_output, input):
            op = torch._C._jit_get_operation(
                'aten::cudnn_convolution_backward_weight' if not transpose else
                'aten::cudnn_convolution_transpose_backward_weight')
            if torch.__version__ > '1.7':
                flags = [
                    torch.backends.cudnn.benchmark,
                    torch.backends.cudnn.deterministic,
                    torch.backends.cudnn.allow_tf32  # for pytorch >= 1.7.1
                ]
            else:
                flags = [
                    torch.backends.cudnn.benchmark,
                    torch.backends.cudnn.deterministic
                ]

            # keep the consistent datatype for FP16 training
            input = input.to(grad_output.dtype)
            grad_weight = op(weight_shape, grad_output, input, padding, stride,
                             dilation, groups, *flags)
            assert grad_weight.shape == weight_shape
            ctx.save_for_backward(grad_output, input)
            return grad_weight

        @staticmethod
        def backward(ctx, grad2_grad_weight):
            grad_output, input = ctx.saved_tensors
            grad2_grad_output = None
            grad2_input = None

            if ctx.needs_input_grad[0]:
                grad2_grad_output = Conv2d.apply(input, grad2_grad_weight,
                                                 None)
                assert grad2_grad_output.shape == grad_output.shape

            if ctx.needs_input_grad[1]:
                p = calc_output_padding(
                    input_shape=input.shape, output_shape=grad_output.shape)
                grad2_input = _conv2d_gradfix(
                    transpose=(not transpose),
                    weight_shape=weight_shape,
                    output_padding=p,
                    **common_kwargs).apply(grad_output, grad2_grad_weight,
                                           None)
                assert grad2_input.shape == input.shape

            return grad2_grad_output, grad2_input

    _conv2d_gradfix_cache[key] = Conv2d
    return Conv2d
