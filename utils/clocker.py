import time
    

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