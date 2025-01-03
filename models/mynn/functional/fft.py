from sympy import fft
import torch
from torch.fft import fftshift, fftn, ifftshift, ifftn

from typing import List


def itok(tensor: torch.Tensor, dim : List[int] = [-2,-1]) -> torch.Tensor:
    return fftshift(fftn(ifftshift(tensor, dim = dim), dim = dim), dim = dim)

fft = itok

def ktoi(tensor: torch.Tensor, dim : List[int] = [-2,-1]) -> torch.Tensor:
    return fftshift(ifftn(ifftshift(tensor, dim = dim), dim = dim), dim = dim)

ifft = ktoi