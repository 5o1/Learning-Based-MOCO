"""
Toolkits for h5 file format.
"""

import torch

import h5py as h5
from typing import List


def h5_to_tensor(h5file: h5.File, for_keys : List[str] = ['kspace']) -> torch.Tensor:
    """
    Convert h5 file to tensor. 

    Parameters:
        file (h5.File): The input h5 file.
        for_keys (List[str]): The keys for the data to be converted.

    Returns:
        torch.Tensor: The converted tensor.
    """
    data = []
    for key in for_keys:
        data.append(torch.tensor(h5file[key][()]))
    return data


