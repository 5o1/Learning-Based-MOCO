import torch

from typing import Literal


def _minmax(x: torch.Tensor):
    minv = x.min(dim=[-1,-2],keepdim=True)
    maxv = x.max(dim=[-1,-2],keepdim=True)
    return minv,maxv

def _meanstd(x: torch.Tensor):
    mean = x.mean(dim=[-1,-2],keepdim=True)
    std = x.std(dim=[-1,-2],keepdim=True)
    return mean,std

def _get_statistics(x: torch.Tensor, mode: Literal['meanstd','minmax'] = 'meanstd'):
    if mode == 'meanstd':
        mean,std = _meanstd(x)
        return mean,std
    elif mode == 'minmax':
        minv,maxv = _minmax(x)
        return minv,maxv-minv
    else:
        raise NotImplementedError
    

def _viewx(x: torch.Tensor, view: None | Literal['abs'] = None):
    if view is None:
        return x
    elif view == 'abs':
        return x.abs()
    else:
        raise NotImplementedError

def normalize(x : torch.Tensor, view: None | Literal['abs'] = None, mode: Literal['meanstd','minmax'] = 'meanstd', requires_ps: bool = False):
    x1 = _viewx(x, view)
    miu,sigma = _get_statistics(x1, mode)
    x = (x - miu) / sigma
    if requires_ps:
        return x, miu, sigma
    else:
        return x