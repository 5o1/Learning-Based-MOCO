from typing import List
from torch import nn

from .functional.fft import itok, ktoi


class FFT2(nn.Module):
    def forward(self, tensor):
        return itok(tensor)

class IFFT2(nn.Module):
    def forward(self, tensor):
        tensor = ktoi(tensor)
        return tensor
    

class FFTn(nn.Module):
    def __init__(self, dim : List[int] = [-2,-1]) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        return itok(tensor, self.dim)

class IFFTn(nn.Module):
    def __init__(self, dim : List[int] = [-2,-1]) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        tensor = ktoi(tensor, self.dim)
        return tensor
