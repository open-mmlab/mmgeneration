import torch


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
        assert isinstance(x, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(x.dtype).max
        if neginf is None:
            neginf = torch.finfo(x.dtype).min
        assert nan == 0
        return torch.clamp(
            x.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)
