# fft functions

from torch.fft import fftshift, fftn, ifftshift, ifftn
from utils.nptorch import ndarray_to_tensor, tensor_to_ndarray
import torch

@ndarray_to_tensor
def itok(I) -> torch.Tensor: 
    return fftshift(fftn(ifftshift(I),dim=[-1,-2]))

@ndarray_to_tensor
def ktoi(K) -> torch.Tensor:
    return fftshift(ifftn(ifftshift(K),dim=[-1,-2]))