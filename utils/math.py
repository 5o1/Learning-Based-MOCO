# math functions

import torch
from .nptorch import ndarray_to_tensor

def mse(a,b):
    a = torch.abs(a)
    b = torch.abs(b)
    return torch.mean( torch.square(a-b))

def norm(x):
    x = torch.abs(x)
    lower = torch.min(x)
    upper = torch.max(x)
    return (x- lower) / (upper - lower)

@ndarray_to_tensor
def fix(x):
    x[torch.isnan(x)] = 0
    x[torch.isinf(x)] = 0
    return x

class mseloss:
    def __init__(self, y) :
        self.y = y


    def __call__(self, y_hat) :
        return mse(y_hat, self.y)