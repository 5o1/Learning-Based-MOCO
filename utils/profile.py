# IO interface for experiment profile


import yaml
import os
from typing import Callable, Iterable
import re
import inspect

class DictFilter:
    """
    Filter for dict data.
    Dict: dictfilter -> keyfilter -> valuefilter -> Dict / List / Object
    List: iterfilter -> Dict / List / Object
    Object: objectfilter -> vars -> Dict
    """
    def __init__(self):
        self.keyfilters = [
            lambda key: None if str.startswith(key,'_') else key, # Hide private key
        ]
        self.valuefilters = [
            # Type filter
            lambda value: None if str(type(value)) == "<class 'torch.Tensor'>" else value,
            lambda value: None if str(type(value)) == "<class 'numpy.ndarray'>" else value,
            # Other rules
            lambda value: str(value) if bool(re.match(r"<class '((numpy|torch)\.dtype\.(.*)|type)'>", str(type(value)))) else value,
            lambda value: inspect.getsource(value) if inspect.isfunction(value) else value,
        ]


    def add_keyfilter(self, filter : callable):
        self.keyfilters.append(filter)


    def add_valuefilter(self, filter : callable):
        self.valuefilters.append(filter)


    def __distributor(self, data):
        data = self.objectfilter(data)
        if isinstance(data, dict):
            return self.dictfilter(data)
        if isinstance(data, (list, tuple)):
            return self.iterfilter(data)
        return self.valuefilter(data)


    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError("Input data should be a dict.")
        return self.dictfilter(data)


    def dictfilter(self, data : dict):
        res = {}
        for key, value in data.items():
            key = self.keyfilter(key)
            if key is None:
                continue
            value = self.__distributor(value)
            if value is None:
                continue
            res[key] = value
        return res


    def iterfilter(self, data : Iterable):
        res = []
        for item in data:
            item = self.__distributor(item)
            if item is None:
                continue
            res.append(item)
        return res

    
    def objectfilter(self, obj):
        if hasattr(obj,"__dict__") and not inspect.isfunction(obj):
            return {obj.__class__.__name__: vars(obj)}
        return obj


    def keyfilter(self, key):
        for filter in self.keyfilters:
            key = filter(key)
            if key is None:
                return None
            if isinstance(key, Exception):
                raise key
        return key


    def valuefilter(self, value):
        for filter in self.valuefilters:
            value = filter(value)
            if value is None:
                return None
            if isinstance(value, Exception):
                raise value
        return value


class Profile:
    """Super class for dumping and loading profile data."""

    def __init__(self, profile : dict | str, load_from_file = False, filter : Callable = None, flow_style :str = 'stereoscopic'):
        if load_from_file:
            self.load(profile)
        else:
            self.profile = profile

        if filter is not None:
            self.profile = filter(self.profile)

        self.flow_style = flow_style


    def get(self):
        return self.profile
    
    
    @staticmethod
    def stereoscopic_flow_style(dumper, data):
        if isinstance(data, list) and all(isinstance(item, (int, float, str)) for item in data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

    
    def dump(self, save_to : str, cover_if_exist : bool = False):
        # Check is exist
        if os.path.exists(save_to) and not cover_if_exist:
            raise FileExistsError(f"File {save_to} already exists.")
        # Dump
        with open(save_to, 'w') as f:
            # If profile is a dict, dump with yaml format, else dump with str format
            if isinstance(self.profile, dict):
                # Construct yaml dumper
                dumper = yaml.Dumper
                if self.flow_style == 'stereoscopic':
                    dumper.add_representer(list, self.stereoscopic_flow_style)
                yaml.dump(self.profile, f, Dumper=dumper, default_flow_style= False, sort_keys= False)
            else:
                print(str(self.profile), file=f)
    
    def load(self, load_from : str):
        # Todo
        self.profile = 1
    

