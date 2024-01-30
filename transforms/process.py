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
    def __init__(self, keys:List[str], for_keys = None):
        super().__init__(for_keys=for_keys)
        self.keys = keys


    def do(self, x : dict):
        res = []
        for key in self.keys:
            res.append(x[key])
        return x

class DoubleX(Transform):
    def __init__(self, deepcopy = True, for_keys = None):
        super().__init__(for_keys=for_keys)
        self.deepcopy = deepcopy


    def do(self, x : torch.Tensor):
        if self.deepcopy:
            return x, x.clone()
        return x, x
    

