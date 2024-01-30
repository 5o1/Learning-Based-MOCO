from torch.fft import fftshift, fftn, ifftshift, ifftn
import torch

from .functional import normalize
from .debug import Transform

class Ifft(Transform):
    def do(self, x: torch.Tensor):
        x = fftshift(ifftn(ifftshift(x),dim=[-1,-2]))
        x = normalize(x,'abs','meanstd')
        return x
    

class Fft(Transform):
    def do(self, x:torch.Tensor):
        x = fftshift(fftn(ifftshift(x),dim=[-1,-2]))
        x = normalize(x,'abs','meanstd')
        return 