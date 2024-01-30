from torch import any,isnan,isinf

from .functional import for_tuple
from .common import Transform



class CheckNan(Transform):
    def __init__(self, tag, for_keys = None):
        super().__init__(for_keys=for_keys)
        self.tag = tag

    @for_tuple
    def do(self, x):
        if any(isnan(x)):
            raise ValueError(f'Runtime with NaN. Tag: {self.tag}')
        return x
    

class CheckInf(Transform):
    def __init__(self, tag, for_keys = None):
        super().__init__(for_keys=for_keys)
        self.tag = tag

    @for_tuple
    def do(self, x):
        if any(isinf(x)):
            raise ValueError(f'Runtime with Inf.Tag: {self.tag}')
        return x
    

class DoubleCheck(Transform):
    """Combine ChechNan and CheckInf."""
    def __init__(self, tag, for_keys = None):
        super().__init__(for_keys=for_keys)
        self.chechnan = CheckNan(tag=tag)
        self.chechinf = CheckInf(tag=tag)
        self.tag = tag

    @for_tuple
    def do(self, x):
        x = self.chechnan(x)
        x = self.chechinf(x)
        return x