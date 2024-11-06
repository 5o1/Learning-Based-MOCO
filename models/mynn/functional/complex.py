import torch

from einops import rearrange

from typing import List


def complex_to_real(tensor : torch.Tensor) -> torch.Tensor:
    """
    Transform complex type Tensor to float type Tensor by concatenating the real and imaginary components along the channel dimension.  

    Parameters:
        tensor (torch.Tensor): The input complex tensor with dimention (batch, channel, height, width, 2).

    Returns:
        torch.Tensor: The transformed float tensor with dimention (batch, channel * 2, height, width).

    """
    tensor = torch.view_as_real(tensor)
    tensor = rearrange(tensor, '... c h w r -> ... (c r) h w').contiguous()
    return tensor





def real_to_complex(tensor : torch.Tensor) -> torch.Tensor:
    """
    Transform float type Tensor to complex type Tensor by splitting the input tensor along the channel dimension. 

    Parameters:
        tensor (torch.Tensor): The input float tensor with dimention (batch, channel * 2, height, width).

    Returns:
        torch.Tensor: The transformed complex tensor with dimention (batch, channel, height, width, 2).
    """
    tensor = rearrange(tensor, '... (c r) h w -> ... c h w r', r=2).contiguous()
    tensor = torch.view_as_complex(tensor)
    return tensor


def complexnorm_abs(tensor : torch.Tensor, dim: List[int]= [-2, -1], eps = 1e-13) -> torch.Tensor:
    r"""
    Normalization for complex numbers.

    It is followed by the normalization formula:

    .. math::
        \frac{z - \min(z)}{\max(z) - \min(z)}

    where :math:`z` is a complex number.

    Returns:
        torch.Tensor: The normalized complex tensor. Which is in the range of [0, 1].

    """
    abs_tensor = torch.abs(tensor)
    return (tensor - abs_tensor.amin(dim=dim, keepdim=True)) / (abs_tensor.amax(dim=dim, keepdim=True) - abs_tensor.amin(dim=dim, keepdim=True) + eps)


def complexnorm_meanstd(tensor : torch.Tensor, dim: List[int]= [-2, -1], eps = 1e-13) -> torch.Tensor:
    r"""
    Normalization for complex numbers.
    
    It is followed by the normalization formula:
    
    .. math::
        \frac{z - \mu}{\sigma}
    
    where :math:`\mu` is the mean and :math:`\sigma` is the standard deviation.

    Returns:
        torch.Tensor: The normalized complex tensor. Which is in the mean of 0 and standard deviation of 1.
    
    """
    abs_tensor = torch.abs(tensor)
    return (tensor - abs_tensor.mean(dim= dim, keepdim=True)) / (abs_tensor.std(dim = dim, keepdim = True) + eps)