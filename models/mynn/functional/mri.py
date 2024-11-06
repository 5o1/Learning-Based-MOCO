"""
This file contains the utility functions for MRI data.
"""


from typing import Tuple, List

import torch

def gen_line_mask(
        shape: Tuple[int, int],
        indices: List[int],
        dim: int,
        device: torch.device
        ) -> torch.Tensor:
    """Generate a mask for k-space data."""
    mask = torch.zeros(shape, device=device)
    if dim %  2 == 0:
        mask[indices, :] = 1
    else:
        mask[:, indices] = 1
    return mask

def gen_undersampling_mask_normal(
        *shape: int,
        accfactor: float = 4,
        sigma:float = 0.35,
        percenter: float = 0.08,
        dim: int = -1,
        device: torch.device = torch.device('cpu')
        ):
    """Generate a mask for k-space data.

    Args:
        shape: the shape of the mask.  
        accfactor: the acceleration factor.  
        sigma: the sigma of the Gaussian distribution.  
        percenter: the percentage of the center lines.  
        dim: the dimension to apply the mask.  
        device: the device to store the mask.  

    Returns:  
        mask: the generated mask.
    """

    mask = torch.zeros(shape, device=device)
    nline = shape[dim]


    ncenter= int(nline*percenter)
    inds = torch.randn(nline * 4, device=device)
    inds = inds * sigma
    inds = inds.clip(-1,1-1e-6)
    
    inds = inds * (nline // 2)
    inds = torch.where(inds>0, torch.floor(inds), torch.ceil(inds))
    inds = inds.to(torch.int64)
    inds = torch.unique(inds)
    inds = inds[torch.randperm(inds.size(0))]
    inds = inds[:int(nline / accfactor)]
    inds = inds[(inds < -ncenter/2) | (inds > ncenter/2)]
    cens = torch.arange(-ncenter/2, ncenter/2, device=device, dtype=torch.int64)
    inds = torch.cat([inds, cens])
    inds = inds + nline // 2

    if dim == 0 or dim == -2:
        mask[inds,:] = 1
    else:
        mask[:, inds] = 1
    return mask


def gen_undersampling_mask_uniform(
        *shape: int,
        accfactor: float = 4,
        sigma:float = 0.35,
        percenter: float = 0.08,
        dim: int = -1,
        device: torch.device = torch.device('cpu')
        ):
    """Generate a mask for k-space data.

    Args:
        shape: the shape of the mask.  
        accfactor: the acceleration factor.  
        sigma: the sigma of the Gaussian distribution.  
        percenter: the percentage of the center lines.  
        dim: the dimension to apply the mask.  
        device: the device to store the mask.  

    Returns:  
        mask: the generated mask.
    """

    mask = torch.zeros(shape[-2:], device=device)
    nline = shape[dim]

    inds = torch.arange(nline, device=device)
    # inds: [0, 1, 2, ..., nline-1]
    # cen_inds: [nline*(0.5-percenter/2), ... , nline // 2 - 1, nline //2 , nline //2 + 1, ... , nline*(0.5+percenter/2)]
    cen_inds = torch.arange(int(nline*(0.5-percenter/2)), int(nline*(0.5+percenter/2)), device=device, dtype=torch.int64)
    ncenter = cen_inds.size(0)
    # from inds remove the cen_inds
    inds = inds[torch.isin(inds, cen_inds, invert=True)]

    # Uniformly sample the indices
    inds = inds[torch.randperm(inds.size(0))]
    inds = inds[:int(nline / accfactor)]

    # Add the center lines
    inds = torch.cat([inds, cen_inds])

    if dim == 0 or dim == -2:
        mask[inds,:] = 1
    else:
        mask[:, inds] = 1
    return mask


def gen_mixtures_mask(shape: Tuple[int, int], n_mixtures: int, distrub: str, ratios: List[float], device: torch.device) -> List[torch.Tensor]:
    """Generate a set of mixtures for MRI data.

    Args:
        shape: the shape of the mask.  
        n_mixtures: the number of mixtures.  
        distrub: the distribution of the mixtures.  
        ratios: the ratios of the mixtures.  
        device: the device to store the mask.

    Returns:
        masks: the generated mask.
    """
    masks = []
    if distrub == 'uniform':
        for i in range(n_mixtures):
            mask = torch.zeros(shape, device=device)
            mask = mask.bernoulli_(ratios[i])
            masks.append(mask)
    elif distrub == 'gaussian':
        for i in range(n_mixtures):
            mask = torch.randn(shape, device=device)
            mask = mask > ratios[i]
            masks.append(mask)
    else:
        raise ValueError('Unsupported distribution: {}'.format(distrub))
    return masks

def rsos(x: torch.Tensor, dim: int = -3) -> torch.Tensor:
    """Root sum of squares."""
    return torch.sqrt((x ** 2).sum(dim))
