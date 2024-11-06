"""
We need to change the structure of the input data, 
such as selecting one layer from multi-layer images. 
The functions that change the shape of the input data 
are all in this file.
"""


from torch import nn
import torch


class RandSlice(nn.Module):
    """Randomly take out a slice of the image"""
    def __init__(self, lower : float = 0.25, upper : float = 0.75, return_index = False, keepdim = False):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.return_index = return_index
        self.keepdim = keepdim

    def forward(self, x : torch.Tensor):
        z = x.shape[0]
        i = torch.randint(int(z * 0.25), int(z * 0.75), (1,))
        if self.keepdim:
            x = x[i]
        else:
            x = x[i].squeeze(0)
        if self.return_index:
            return x, i
        return x