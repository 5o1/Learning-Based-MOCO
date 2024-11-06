"""
Toolkits for h5 file format.
"""


from torch import nn
import torch

import h5py as h5
from typing import List

class H52tensor(nn.Module):
    """Convert H5file format image to tensor."""

    def __init__(self, key : str = 'kspace') -> None:
        super().__init__()
        self.key = key

    def forward(self, file: h5.File) -> torch.Tensor:

        data = file[self.key][:]
        tensor = torch.from_numpy(data)
        return tensor


