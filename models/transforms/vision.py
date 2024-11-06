"""
Includes vision-related transforms such as cropping, padding, etc.
"""


from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import transforms as T

from typing import List, Callable



class CircularDecorator(nn.Module):
    """Cyclic padding of the image (to simulate the periodicity of the cyclic convolution), then perform func(x), then restore to the original dimensions."""

    def __init__(self, func: Callable | List[Callable], readoutdim : int = 1, for_keys = None):
        super().__init__(for_keys=for_keys)
        self.func = func
        self.readoutdim = readoutdim

    def forward(self, image : Tensor):
        shape = [
            image.shape[-2],
            image.shape[-1]
            ]
        padshape = [
            shape[-1] // 2 if self.readoutdim in [1,-1] else 0, 
            shape[-1] // 2 if self.readoutdim in [1,-1] else 0, 
            shape[-2] // 2 if self.readoutdim in [0,-2] else 0, 
            shape[-2] // 2 if self.readoutdim in [0,-2] else 0
            ]
        image = F.pad(image, padshape, mode='circular')
        if isinstance(self.func, List):
            for f in self.func:
                image = f(image)
        else:
            image = self.func(image)
        crop = T.CenterCrop(shape)
        image = crop(image)
        return image
    

class SquareCrop(nn.Module):
    """Crop an image of any size into square shape."""

    def __init__(self, for_keys = None):
        super().__init__(for_keys=for_keys)

    def forward(self, image : Tensor):
        shape = [
            image.shape[-2],
            image.shape[-1]
            ]
        shape = [
            min(shape),
            min(shape)
        ]
        crop = T.CenterCrop(shape)
        image = crop(image)
        return image
    