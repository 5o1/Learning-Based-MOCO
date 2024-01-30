from functools import wraps


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