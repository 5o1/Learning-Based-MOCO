from torch.nn import Module, ReLU
from torch import concat, Tensor, view_as_real, moveaxis, view_as_complex, is_complex

class ComplexReLU(Module):


    def __init__(self):
        super().__init__()
        self.relu = ReLU()


    def forward(self, x: Tensor):
        if is_complex(x):
            x = view_as_real(x)
            x = moveaxis(x, -1, 0).contiguous()
            x = self.relu(x)
            x = moveaxis(x, 0, -1).contiguous()
            x = view_as_complex(x)
            return x
        else:
            x = self.relu(x)
            return x