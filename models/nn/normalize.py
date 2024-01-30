from torch.nn import Module,BatchNorm2d
from torch import concat, Tensor, view_as_real, moveaxis, view_as_complex, is_complex

class ComplexBatchNorm2d(Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = BatchNorm2d(channels)
        self.cat = concat

    def forward(self, x: Tensor):
        if is_complex(x):
            x = view_as_real(x)
            x = moveaxis(x, -1, 0).contiguous()
            x = self.cat([self.bn(x[0]).unsqueeze(0),self.bn(x[1]).unsqueeze(0)])
            x = moveaxis(x, 0, -1).contiguous()
            x = view_as_complex(x)
            return x
        else:
            x = self.bn(x)
            return x