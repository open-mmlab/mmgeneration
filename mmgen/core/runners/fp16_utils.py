# Copyright (c) OpenMMLab. All rights reserved.
import functools
from collections import abc
from inspect import getfullargspec

import numpy as np
import torch
import torch.nn as nn
from mmcv.utils import TORCH_VERSION

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.autocast would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    from torch.cuda.amp import autocast
except ImportError:
    pass


def nan_to_num(x, nan=0.0, posinf=None, neginf=None, *, out=None):
    r"""Replaces :literal:`NaN`, positive infinity, and negative infinity
    values in :attr:`input` with the values specified by :attr:`nan`,
    :attr:`posinf`, and :attr:`neginf`, respectively. By default,
    :literal:`NaN`s are replaced with zero, positive infinity is replaced with
    the greatest finite value representable by :attr:`input`'s dtype, and
    negative infinity is replaced with the least finite value representable by
    :attr:`input`'s dtype.

    .. note::

        This function is provided in ``PyTorch>=1.8.0``. Here is a
        reimplementation to avoid attribute error in lower PyTorch version.

    Args:
        x (Tensor): Input tensor.
        nan (Number, optional): the value to replace :literal:`NaN`\s with.
            Default is zero.
        posinf (Number, optional): if a Number, the value to replace positive
            infinity values with. If None, positive infinity values are
            replaced with the greatest finite value representable by
            :attr:`input`'s dtype. Default is None.
        neginf (Number, optional): if a Number, the value to replace negative
            infinity values with. If None, negative infinity values are
            replaced with the lowest finite value representable by
            :attr:`input`'s dtype. Default is None.

    Returns:
        Tensor: Output tensor.
    """
    try:
        return torch.nan_to_num(
            x, nan=nan, posinf=posinf, neginf=neginf, out=out)
    except AttributeError:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f'argument input (position 1) must be Tensor, not {type(x)}')
        if posinf is None:
            posinf = torch.finfo(x.dtype).max
        if neginf is None:
            neginf = torch.finfo(x.dtype).min
        assert nan == 0
        # a better choice is to use nansum, but this function is not supported
        # in PyTorch 1.5
        # x.unsqueeze(0).nansum(0)
        x[torch.isnan(x)] = 0.
        return torch.clamp(x, min=neginf, max=posinf, out=out)


def cast_tensor_type(inputs, src_type, dst_type):
    """Recursively convert Tensor in inputs from src_type to dst_type.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype): Source type..
        dst_type (torch.dtype): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    if isinstance(inputs, nn.Module):
        return inputs
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def auto_fp16(apply_to=None, out_fp32=False):
    """Decorator to enable fp16 training automatically.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If inputs arguments are fp32 tensors, they will
    be converted to fp16 automatically. Arguments other than fp32 tensors are
    ignored. If you are using PyTorch >= 1.6, torch.cuda.amp is used as the
    backend, otherwise, original mmcv implementation will be adopted.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp32 (bool): Whether to convert the output back to fp32.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp16
        >>>     @auto_fp16()
        >>>     def forward(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp16
        >>>     @auto_fp16(apply_to=('pred', ))
        >>>     def do_something(self, pred, others):
        >>>         pass
    """

    def auto_fp16_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@auto_fp16 can only be used to decorate the '
                                'method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)

            # define output type by class itself
            if hasattr(args[0], 'out_fp32') and args[0].out_fp32:
                _out_fp32 = True
            else:
                _out_fp32 = False

            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            # Here, we change the default behaviour with Yu Xiong's
            # implementation
            args_to_cast = [] if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            # NOTE: default args are not taken into consideration
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.float, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            if TORCH_VERSION != 'parrots' and TORCH_VERSION >= '1.6.0':
                output = autocast(enabled=True)(old_func)(*new_args,
                                                          **new_kwargs)
            else:
                # output = old_func(*new_args, **new_kwargs)
                raise RuntimeError('Please use PyTorch >= 1.6.0')
            # cast the results back to fp32 if necessary
            if out_fp32 or _out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output

        return new_func

    return auto_fp16_wrapper
