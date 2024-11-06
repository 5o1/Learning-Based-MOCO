"""
Toolkits provided for any function to use.
"""

import torch

from functools import wraps, update_wrapper
from typing import List, Dict, Literal


class packed:
    """A decorator that packs the forward function of a class, allowing for keys to be specified for the input dict, and mapping to be specified for the output dict.

    # Functions:

    - for_keys: Pack the forward function of the class, allowing for keys to be specified for the input dict, and mapping to be specified for the output dict.
    """
    def __init__(self, cls):
        self.cls = cls

    def __call__(self, *args, **kwargs):
        class dummy(self.cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, '__already_packed__'), "forward class is already packed"
                self.__already_packed = False
                self.__raw_foward = self.forward

                self.__input_keys = None
                self.__input_mapping = None
                self.__input_remaining = None
                self.__input_raw_mapping = None

                self.__output_mapping = None

            
            def __packed_forward(*args, **kwargs):
                assert args or kwargs, "input must be specified"
                if args:
                    x = args.pop(0)
                else:
                    first_key = next(iter(kwargs))
                    x = kwargs.pop(first_key)

                if self.__input_raw_mapping:
                    return {self.raw_mapping: self.__raw_foward(x, *args, **kwargs)}
                
                assert isinstance(x, dict), "input must be a dict"

                result = {}
                for key, value in x.items():
                    if self.__packed_input_keys__ and (self.__packed_input_keys__ == 'all' or key in self.__packed_input_keys__):
                        res_of_key = self.__packed_raw_foward__(value, *args, **kwargs)
                        # output mapping
                        # Todo
                        result[self.__packed_input_mapping__[key] if key in self.__packed_input_mapping__ else key] = self.__packed_raw_foward__(value, *args, **kwargs)
                    elif key in self.__packed_input_mapping__:
                        result[self.__packed_input_mapping__[key]] = value
                    elif self.__packed_input_remaining__:
                        result[key] = value

            def input(self, keys : List[str] | Literal['all'] = [], mapping : Dict[str, str] = {}, remaining: bool = True, raw_mapping : str | None = None) -> self.cls:
                """Pack the forward function of the class, allowing for keys to be specified for the input dict, and mapping to be specified for the output dict.
                
                Args:
                    keys (List[str] | Literal['all'], optional): The keys to be used for the input dict. Defaults to [].
                    mapping (Dict[str, str], optional): The mapping for the output dict. Defaults to {}.
                    remaining (bool, optional): Whether to include the remaining keys in the output dict. Defaults to True.
                    raw_mapping (str | None, optional): The key for the output dict. Defaults to None.
                    
                Returns:
                    cls: The packed obj.
                """
                assert not self.__already_packed, "forward function is already packed"
                if self.__packed_func__ is None:
                    self.__packed_func__ = self.forward
                @wraps(self.__packed_func__)
                def pack(input, *args, **kwargs):
                    if raw_mapping:
                        return {raw_mapping: self.__packed_func__(input, *args, **kwargs)}
                    assert isinstance(input, dict), "input must be a dict"
                    res = {}
                    for key, value in input.items():
                        if keys and (keys == 'all' or key in keys):
                            res[mapping[key] if key in mapping else key] = self.__packed_func__(value, args, kwargs)
                        elif key in mapping:
                            res[mapping[key]] = value
                        elif remaining:
                            res[key] = value
                    return res
                self.forward = pack
                self.__already_packed = True
                return self
        update_wrapper(dummy, self.cls, updated=())
        return dummy

def for_tuple(func):
    """A decorator that executes the function separately for each element of the input tulpe."""
    @wraps(func)
    def wrapper(self, x):
        if isinstance(x, tuple):
            res = []
            x = list(x)
            for i in x:
                res.append(func(self, i))
            return tuple(res)

        else:
            return func(self, x)
        
    
    return wrapper


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