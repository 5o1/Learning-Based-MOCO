import time
from datetime import datetime


def get_time_diff(start_time: datetime, end_time: datetime) -> str:
    """
    计算两个时间点之间的差异，并格式化为天、小时、分钟、秒的字符串形式，
    如果某个单位为0则不显示该部分。

    Parameters:
    - start_time (datetime): 开始时间
    - end_time (datetime): 结束时间

    Returns:
    - str: 格式化后的时间差字符串
    """
    elapsed_time = end_time - start_time
    
    days = elapsed_time.days
    hours, remainder = divmod(elapsed_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_diff_parts = []
    
    if days > 0:
        time_diff_parts.append(f"{days}d")
    if hours > 0:
        time_diff_parts.append(f"{hours}h")
    if minutes > 0:
        time_diff_parts.append(f"{minutes}m")
    if seconds > 0 or not time_diff_parts:
        time_diff_parts.append(f"{seconds}s")
    
    return " ".join(time_diff_parts)


class clocker:
    __available = True

    def __init__(self, logfunc : callable = print, historyfunc : callable = None):
        self.logfunc = logfunc
        self.historyfunc = historyfunc


    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if not self.__available:
                return func(*args, **kwargs)
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            if self.historyfunc:
                self.historyfunc(end - start)
            if self.logfunc:
                self.logfunc(f'{func.__name__} took {end - start} seconds')
            return result
        return wrapper
    
    @staticmethod
    def on():
        clocker.__available = True

    @staticmethod
    def off():
        clocker.__available = False



if __name__ == '__main__':
    his = []
    @clocker(history=his)
    def test():
        time.sleep(1)
        print('test')

    clocker.on()

    test()

    print(his)