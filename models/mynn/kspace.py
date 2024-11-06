"""
Common transforms used in MoCo, such as normalization, random crop, random flip, etc.
"""


import torch
from torch import nn

from typing import List, Any, Literal, Tuple
from abc import abstractmethod

import functional as myf


class KlineHider(nn.Module):
    def __init__(
            self,
            mask: torch.Tensor = None, 
            dim : List[int] = [0], 
            hiderate : float = 0.2,
            *args, **kwargs
            ) -> Any:
        super().__init__(*args,**kwargs)
        self._mask = mask
        self.dim = dim
        self.hiderate = hiderate

    def forward(self, x):
        # fourier transform
        K = myf.itok(x)
        
        if self._mask is None:
            self._mask = torch.ones_like(K)
            for dim in self.dim:
                Nlines = K.shape[dim]
                Nshuffle = torch.randperm(Nlines)
                submask = torch.ones(Nlines)
                submask[Nshuffle[:int(Nlines * self.hiderate)]] = 0
                # submask.reshape([1] * dim + [-1] + [1] * (len(K.shape) - dim - 1))
                print(submask.shape)
                print(Nlines)
                print(K.shape)
                self._mask *= submask
        
        # multiply Ki with masks[i] and sum 
        K *= self._mask
        
        # inverse fourier transform
        I = torch.fft.ifftshift(torch.fft.ifftn(K))
        return { 'x' : x , 'y' : I , 'mask' : self._mask}
    

def sinc(x):
    return torch.where(x == 0, torch.tensor(1.0), torch.sin(x) / (x)) 

def kspace_normalization_sinc(x: torch.Tensor, pattern : Literal['circular', 'cross'], dim : Tuple[int] | List[int] = (-2, -1)):
    """
    Normalize k-space data to [0, 1] using sinc function.
    Args:
        x: input k-space data
        pattern: normalization pattern, 'circular' or 'cross'
        start_dim: the first dimension to apply normalization
    """
    assert pattern in ['circular', 'cross'], f"Invalid pattern: {pattern}"
    num_dim = len(dim)

    # minmax normalization
    x = myf.complexnorm_abs(x, dim=dim)

    # construct the position matrix, each dimension is in the range of [-shape[dim]//2, shape[dim]//2]
    # The shape of pos_map is [dim1, dim2, ..., num_dim]
    pos = torch.meshgrid([torch.arange(x.shape[d], device=x.device) - x.shape[d] // 2 for d in dim]) # [dim1, dim2, ...]
    pos = torch.stack(pos, dim=-1) # [dim1, dim2, ..., num_dim]


if __name__ == '__main__':
    # test sinc
    print(sinc(torch.tensor(3.0)))
    

