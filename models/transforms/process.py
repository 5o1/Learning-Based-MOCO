"""
Process transforms are used to process the input data in a way that is not directly related to the model.
"""


import torch
from typing import List
import random
from .functional.common import for_tuple

from .debug import Transform

class RandSlice(Transform):
    """Randomly take out a slice of the image"""
    def __init__(self, lower : float = 0.25, upper : float = 0.75, for_keys = None):
        super().__init__(for_keys=for_keys)
        self.lower = lower
        self.upper = upper

    @for_tuple
    def do(self, x : torch.Tensor):
        z = x.shape[0]
        i = random.randint(int(z * 0.25), int(z * 0.75))
        x = x[i]
        return x
    

class CollectToList(Transform):
    """Turn dict into list in order of for_keys."""
    def __init__(self, x_keys:List[str], y_keys = List[str], for_keys = None):
        super().__init__(for_keys=for_keys)
        self.x_keys = x_keys
        self.y_keys = y_keys


    def do(self, input : dict):
        x = {key:input[key] for key in self.x_keys}
        y = {key:input[key] for key in self.y_keys}
        return [x,y]

class DoubleX(Transform):
    def __init__(self, deepcopy = True, for_keys = None):
        super().__init__(for_keys=for_keys)
        self.deepcopy = deepcopy


    def do(self, x : torch.Tensor):
        if self.deepcopy:
            return x, x.clone()
        return x, x
    

class Todevice(Transform):
    def __init__(self, deepcopy = True, for_keys = None):
        super().__init__(for_keys=for_keys)


    def do(self, x : torch.Tensor):
        if self.deepcopy:
            return x, x.clone()
        return x, x