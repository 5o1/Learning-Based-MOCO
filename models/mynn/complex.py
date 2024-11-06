import torch
from torch import nn

from typing import List
from functools import wraps
from einops import rearrange

from .functional import complex

class ComplexReLU(nn.ReLU):
    """ReLU activation function for complex numbers."""
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(tensor):
            tensor = torch.view_as_real(tensor)
            tensor = rearrange(tensor, 'b c h w r -> b (c r) h w')
            tensor = super().forward(tensor)
            tensor = rearrange(tensor, 'b (c r) h w -> b c h w r', r=2)
            tensor = torch.view_as_complex(tensor)
            return tensor
        else:
            return super().forward(tensor)
        

class ComplexBatchNorm2d(nn.BatchNorm2d):
    """Batch normalization for complex numbers."""
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(tensor):    
            real = super().forward(tensor.real)
            imag = super().forward(tensor.imag)
            tensor = rearrange([real, imag], 'r b c h w -> b c h w r')
            tensor = torch.view_as_complex(tensor)
            return tensor
        else:
            return super().forward(tensor)
        
class ComplexAbsNorm(nn.Module):
    """Normalization for complex numbers."""
    def __init__(self, dim : List[int] = [-2, -1]):
        super().__init__()
        self.dim = dim

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        return complex.complexnorm_abs(tensor, dim = self.dim)
    

class Complex2Real(nn.Module):
    """
    Transform complex type Tensor to float type Tensor by concatenating the real and imaginary components along the channel dimension.  

    **Input dimension:** (batch, channel, height, width)  

    **Output dimension:** (batch, 2 * channel, height, width)  

    """
    def forward(self, tensor : torch.Tensor):
        return complex.complex_to_real(tensor)
    




class Real2Complex(nn.Module):
    """
    Transform float type Tensor to complex type Tensor by splitting the input tensor along the channel dimension. 

    **Input dimension:** (batch, 2 * channel, height, width)

    **Output dimension:** (batch, channel, height, width)

    """
    def forward(self, tensor : torch.Tensor):
        return complex.real_to_complex(tensor)


def crc(func):
    """A decorator for complex-to-real-[to do something]-to-complex conversion."""
    @wraps(func)
    def wrapper(tensor):
        tensor = complex.complex_to_real(tensor)
        tensor = func(tensor)
        tensor = complex.real_to_complex(tensor)
        return tensor
    return wrapper
