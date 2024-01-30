from typing import List
from functools import wraps

import torch

from .functional.common import for_tuple
from .debug import Transform


class ComplexNorm(Transform):
    def __init__(self, dim : List[int] = (-2,-1), return_magnitude = False, for_keys = None):
        super().__init__(for_keys=for_keys)
        self.dim = dim
        self.return_magnitude = return_magnitude

    def do(self, x : torch.Tensor):
        x = torch.view_as_real(x)
        magnitude = torch.norm(x, dim=self.dim, keepdim=True)
        x = x / magnitude
        x = torch.view_as_complex(x)
        if self.return_magnitude:
            return x, magnitude
        return x


class ViewAsReal(Transform):
    def do(self, comp : torch.Tensor):
        if isinstance(comp, list):
            res = []
            for x in comp:
                x = torch.view_as_real(x)
                x = torch.moveaxis(x, -1, 0).contiguous()
                res.append(x)
            return res
        real = torch.view_as_real(comp)
        real = torch.moveaxis(real, -1, 0).contiguous()
        return real
    

class ViewAsReal_combine(Transform):
    """After splitting the complex type Tensor into real numbers, merge them into the dimensions of the channel."""
    @for_tuple
    def do(self, comp : torch.Tensor):
        if isinstance(comp, list):
            res = []
            for x in comp:
                real = x.real
                imag = x.imag
                x = torch.concat([real,imag],dim = -3)
                res.append(x)
            return res
        x = comp
        real = x.real
        imag = x.imag
        x = torch.concat([real,imag],dim = -3)
        return x
    

class ViewAsComplex(Transform):
    def do(self, real : torch.Tensor):
        if isinstance(real, list):
            res = []
            for x in real:
                x = torch.moveaxis(x, 0, -1).contiguous()
                x = torch.view_as_complex(x)
                res.append(x)
            return res
        real = torch.moveaxis(real, 0, -1).contiguous()
        comp = torch.view_as_complex(real)
        return comp


class ViewAsComplex_combine(Transform):
    """Regard the first half and the second half of the channel dimension as real and imaginary components, respectively, and convert the float type Tensor to complex type"""
    @for_tuple
    def do(self, comp : torch.Tensor):
        if isinstance(comp, list):
            res = []
            for x in comp:
                x = x.moveaxis(-3, 0)
                channels = x.shape[0]
                real = x[:channels].moveaxis(0, -3)
                imag = x[channels:].moveaxis(0, -3)
                x = torch.complex(real, imag)
                res.append(x)
            return res
        x = comp
        x = x.moveaxis(-3, 0)
        channels = x.shape[0]
        real = x[:channels].moveaxis(0, -3)
        imag = x[channels:].moveaxis(0, -3)
        x = torch.complex(real, imag)
        return x


def crc(func):
    """Complex to Real (to do something) to Complex."""
    # Decorator
    view_as_real = ViewAsReal()
    view_as_complex = ViewAsComplex()

    @wraps(func)
    def wrapper(x):
        x = view_as_real(x)
        x = func(x)
        x = view_as_complex(x)
        return x
    return wrapper
