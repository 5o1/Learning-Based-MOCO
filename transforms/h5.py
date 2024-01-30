import h5py as h5
from typing import List

import numpy
import torch

from .debug import Transform

def complexNorm(x : torch.Tensor, return_magnitude = False):
    """Normalize."""
    x1 = x.abs()
    x2 = (x1 - x1.mean(dim=[-1,-2],keepdim=True)) / x1.std(dim = [-1,-2], keepdim = True)
    magnitude = x2 / x1
    x = x * magnitude

    if return_magnitude:
        return x, magnitude
    return x


class H52tensor(Transform):
    """Convert H5file format image to tensor."""

    def __init__(self, for_keys : List = ['kspace'], return_magnitude = False, device = None):
        super().__init__()
        self.for_keys = for_keys
        self.return_magnitude = return_magnitude
        self.device = device

    def __call__(self, file: h5.File):
        res = {}
        for key in self.for_keys:
            data = file[key][:]
            tensor = torch.from_numpy(data)
            if self.device is not None:
                tensor = tensor.to(self.device)
            res[key] = tensor
        if len(res) == 1:
            return res[self.for_keys[0]]
        return res


