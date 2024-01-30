from abc import abstractmethod

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
            if isinstance(res, dict):
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

    @property
    def args(self):
        args = {self.__class__.__name__:self.__dict__}
        return args

    def __repr__(self):
        return str(self.args)