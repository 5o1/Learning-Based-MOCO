# IO interface for experiment profile


import yaml
from typing import Callable, Iterable, Union
import re
import inspect
import os

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

    def __init__(self, profile: Union[dict, str], load_from_file=False, filter: Callable = None, flow_style: str = 'stereoscopic'):
        if load_from_file:
            self.load_from_file(profile)
        else:
            self.profile = profile

        if filter is not None:
            self.profile = filter(self.profile)

        self.flow_style = flow_style

    def get(self):
        return self.profile
    
    class LiteralString(str): 
        pass

    class CustomDumper(yaml.Dumper):
        pass

    class CustomLoader(yaml.SafeLoader):
        pass

    def literal_str_representer(dumper: yaml.Dumper, data: str):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    
    def literal_str_constructor(loader: yaml.Loader, node: yaml.Node):
        # Load the node as a standard Python string
        value = loader.construct_scalar(node)
        # Wrap it in LiteralString
        return Profile.LiteralString(value)

    CustomDumper.add_representer(LiteralString, literal_str_representer)
    CustomLoader.add_constructor('tag:yaml.org,2002:str', literal_str_constructor)

    # def stereoscopic_flow_style(dumper, data):
    #     if isinstance(data, list) and all(isinstance(item, (int, float, str)) for item in data):
    #         return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    #     return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

    def dump_to_string(self) -> str:            
        def convert_to_literal_string(d):
            for key, value in d.items():
                if isinstance(value, str) and ('\n' in value or len(value) > 40):
                    d[key] = Profile.LiteralString(value)
                elif isinstance(value, dict):
                    convert_to_literal_string(value)
        profile_copy = self.profile.copy()
        convert_to_literal_string(profile_copy)
        return yaml.dump(profile_copy, Dumper=Profile.CustomDumper, default_flow_style=False, sort_keys=False)
    
    def load_from_string(yaml_string):
        # Load the YAML string and convert it back into a dictionary
        data = yaml.load(yaml_string, Loader=Profile.CustomLoader)
        
        # If necessary, further processing of data can be done here
        return data

    
    def __repr__(self) -> str:
        return self.dump_to_string()

    def dump_to_file(self, save_to: str):
        """Return YAML dump of the profile as a string."""
        with open(save_to, 'w') as f:
            f.write(repr(self))

    def load_from_file(self, load_from: str):
        """Loads profile data from a YAML file at the specified path."""
        if not os.path.exists(load_from):
            raise FileNotFoundError(f"File {load_from} does not exist.")
        
        with open(load_from, 'r') as f:
            self.profile = yaml.load(f, Loader=Profile.CustomLoader)