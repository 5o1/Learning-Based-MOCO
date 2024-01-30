import re
import inspect

class DictFilter:
    def __init__(self):
        self.error = ValueError('Bad key or value.')
        self.keyfilters = [
            lambda key: self.error if str.startswith(key,'_') else key,
        ]
        self.valuefilters = [
            lambda value: str(value) if bool(re.match(r"<class '((numpy|torch)\.dtype\.(.*)|type)'>", str(type(value)))) else value,
            lambda value: self._list(vars(value)["transforms"]) if hasattr(value, '__dict__') and "transforms" in vars(value) else value,
            lambda value: {value.__class__.__name__:self(vars(value))} if hasattr(value, '__dict__') else value,
            lambda value: inspect.getsource(value) if callable(value) else value,
        ]

    def add_keyfilter(self, filter : callable):
        self.keyfilters.append(filter)

    def add_valuefilter(self, filter : callable):
        self.valuefilters.append(filter)

    def _list(self, l : list):
        for i, item in enumerate(l):
            l[i] = {item.__class__.__name__:self(vars(item))}
        return l
    
    def _dict(self, d : dict):
        res = {}
        for key, value in d.items():
            for filter in self.keyfilters:
                key = filter(key)
                if key is self.error:
                    break
            if key is self.error:
                continue
            for filter in self.valuefilters:
                value = filter(value)
                if value is self.error:
                    break
            if value is self.error:
                continue
            res[key] = value
        return res

    def __call__(self, x : dict | list):
        if isinstance(x, list):
            return self._list(x)
        if isinstance(x, dict):
            return self._dict(x)
