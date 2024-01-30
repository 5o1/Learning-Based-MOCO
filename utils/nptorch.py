import torch
import numpy as np

def ndarray_to_tensor(func):
    # Decorator
    def wrapper(*args, **kwargs):
        newargs = encoder(args)
        newkwargs = encoder(kwargs)

        result = func(*newargs, **newkwargs)
        return result

    def encoder(obj):
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (tuple, list)):
            newobj = []
            for item in obj:
                newobj.append(encoder(item))
            if isinstance(obj, tuple):
                newobj = tuple(newobj)
            return newobj
        if isinstance(obj, dict):
            newobj = {}
            for key, value in obj.items():
                newobj[key] = encoder(value)
            return newobj
        return obj # fallback
    
    return wrapper



def tensor_to_ndarray(func):
    # Decorator
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        result = decoder(result)
        return result

    def decoder(obj):
        if isinstance(obj, torch.Tensor):
            return obj.numpy()
        if isinstance(obj, np.ndarray):
            return obj
        if isinstance(obj, (tuple, list)):
            newobj = []
            for item in obj:
                obj.append(decoder(item))
            if isinstance(obj, tuple):
                newobj = tuple(obj)
            return newobj
        if isinstance(obj, dict):
            for key, value in obj:
                obj[key] = decoder(value)
            return obj
        return obj # fallback
    
    return wrapper