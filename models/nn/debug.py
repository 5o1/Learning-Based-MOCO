from torch import any,isnan,isinf
from torch.nn import Module

class CheckNan(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, tag):
        if any(isnan(x)):
            raise ValueError(f'Runtime with NaN. Tag: {tag}')
        return x
    

class CheckInf(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, tag):
        if any(isinf(x)):
            raise ValueError(f'Runtime with Inf.Tag: {tag}')
        return x