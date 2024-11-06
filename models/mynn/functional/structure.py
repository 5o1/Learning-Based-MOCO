from einops import rearrange
import torch
from typing import Tuple


def bit2traj(bit: torch.Tensor, start_dim = -2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert bit to trajectory

    Args:
        bit (Tensor): (B, C, [D, ...], H, W)
        start_dim (int): start dimension to flatten

    Returns:
        pts_seq (Tensor): (B, H * W [ * D * ...], C)
        pos_seq (Tensor): (B, H * W [ * D * ...], len({H, W, [D, ...]}))
    """
    # construct pos map of Cartesian coordinate system from -1 to 1 in each dimension [..., H, W]
    # for example, if H = 4, W = 4, then pos_map = torch.Tensor[[[-1, -1], [-1, 0], [-1, 1], [-1, 2]], ...]
    num_dim = (len(bit.shape) - start_dim) % len(bit.shape) # number of dimensions whatever the start_dim is positive or negative
    num_dim = len(bit.shape) if num_dim == 0 else num_dim
    pos_map = torch.meshgrid([torch.linspace(-1, 1, bit.shape[len(bit.shape) - num_dim + dim], device = bit.device) for dim in range(num_dim)], indexing='ij') # [..., H, W] 
    pos_map = torch.stack(pos_map, dim = -1) # [..., H, W, num_dim]

    # flatten pos_map to (B, H * W [ * D * ...], num_dim)
    pos_seq = torch.flatten(pos_map, start_dim = 0, end_dim = -2) # (B, H * W [ * D * ...], num_dim)

    # flatten bit to (B, C, H * W [ * D * ...])
    pts_seq = rearrange(bit, 'b c ... h w -> b (h w ...) c', h = bit.shape[-2], w = bit.shape[-1]) # (B, H * W [ * D * ...], C)

    return pts_seq, pos_seq


if __name__ == '__main__':
    bit = torch.arange(8).reshape(1, 2, 2, 2)
    print(bit)
    res = bit2traj(bit, start_dim=-3)
    print(res)