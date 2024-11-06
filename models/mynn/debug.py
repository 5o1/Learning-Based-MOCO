from torch import any,isnan,isinf
from torch import nn

class CheckNan(nn.Module):
    """Check if there is any NaN in the tensor."""
    def forward(self, tensor, tag):
        if any(isnan(tensor)):
            raise ValueError(f'Runtime with NaN. Tag: {tag}')
        return tensor
    

class CheckInf(nn.Module):
    """Check if there is any Inf in the tensor."""
    def forward(self, tensor, tag):
        """Check if there is any Inf in the tensor."""
        if any(isinf(tensor)):
            raise ValueError(f'Runtime with Inf. Tag: {tag}')
        return tensor