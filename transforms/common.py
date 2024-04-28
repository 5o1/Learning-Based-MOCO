from abc import abstractmethod

import torch

class Transform:
    """Super class of all Transforms, used to share some common functions."""
    def __init__(self, for_keys = None):
        super().__init__()
        self.for_keys = for_keys

    @abstractmethod
    def do(self, x):
        pass

    def __call__(self, x):
        if self.for_keys is None:
            return self.do(x)
        else:
            if isinstance(x, dict):
                res = {}
                for key in self.for_keys:
                    res[key] = self.do(x[key])
                return res
            else:
                res = []
                for key in self.for_keys:
                    res.append(self.do(x[key]))
                res = type(x)(res)
                return res

class exp1(Transform):
    def __init__(self, for_keys = None, transforms = None, masks = None):
        super().__init__(for_keys)
        self.transforms = transforms
        self._masks = masks

    def do(self, x):
        # transform
        Ii = []
        for transform in self.transforms:
            Ii.append(transform(x))
        
        # fourier transform
        Ki = []
        for i in Ii:
            Ki.append(torch.fft.fftn(torch.fft.fftshift(i)))

        if self._masks is None:
            Nmasks = len(Ki)
            Nlines = Ki[0].shape[0]
            Nshuffle = torch.randperm(Nlines)
            self._masks = [torch.zeros_like(Ki[0]) for _ in range(len(Ki))]
            for i in range(Nmasks):
                self._masks[i][Nshuffle[i * Nlines // Nmasks : (i + 1) * Nlines // Nmasks]] = 1
            self._masks[-1][Nshuffle[Nmasks * Nlines // Nmasks :]] = 1
        
        # multiply Ki with masks[i] and sum 
        K = torch.zeros_like(Ki[0])
        for i in range(len(Ki)):
            K += Ki[i] * self._masks[i]
        
        # inverse fourier transform
        I = torch.fft.ifftshift(torch.fft.ifftn(K))
        return { 'x' : x , 'y' : x, 'Ii' : Ii }