from torch.fft import fftn,fftshift,ifftn,ifftshift
from torch.nn import Module


class FFT(Module):
    def __init__(self):
        super(FFT, self).__init__()
    
    def forward(self, x):
        x = fftshift(fftn(ifftshift(x, dim = [-1,-2]),dim=[-1,-2]), dim = [-1,-2])
        return x
    


class IFFT(Module):
    def __init__(self):
        super(IFFT, self).__init__()
    
    def forward(self, x):
        x = fftshift(ifftn(ifftshift(x, dim = [-1,-2]),dim=[-1,-2]), dim = [-1,-2])
        return x