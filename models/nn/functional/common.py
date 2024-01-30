import torch

def is_complex_dtype(dtype: torch.dtype):
    dummy = torch.zeros(1, dtype=dtype)
    return dummy.is_complex()


def dict_filter(d : dict, dtypes: list = [float, int, torch.dtype, str, list, tuple, bool, set, dict, type(None)], tostr : list = [torch.dtype]):
    res = {}
    for key,value in d.items():
        if str.startswith(key,'_'):
            continue
        if type(value) in dtypes:
            if type(value) in tostr:
                res[key] = str(value)
            else:
                res[key] = value
    return res