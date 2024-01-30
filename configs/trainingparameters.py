import argparse
import datetime

def str2bool(x : str):
    lx = x.lower()
    if lx in ['false','off','no']:
        return False
    if lx in ['true','on','yes']:
        return True
    try:
        numeric_value = int(lx)
        if numeric_value == 0:
            return False
        else:
            return True
    except ValueError:
        raise(ValueError(f"Expected to receive a bool type, but received value {x}"))

class TrainingParameters():
    def __init__(self):
            self.parser = argparse.ArgumentParser(description='')
            self.parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime(r'%Y-%m-%d-%H-%M-%S'))
            self.parser.add_argument('--outdir', type=str, default='exp')
            self.parser.add_argument('--basedir', type=str,default='./')
            self.parser.add_argument('--trainpath', type=str, default=None)
            self.parser.add_argument('--valpath', type=str, default=None)
            self.parser.add_argument('--batchsize', type=int, default=16)
            self.parser.add_argument('--epoch', type=int, default=300)
            self.parser.add_argument('--depth', type=int, default=5)
            self.parser.add_argument('--topchannels', type=int, default=32)
            self.parser.add_argument('--lr', type=float, default=5e-4)
            self.parser.add_argument('--cache', type=str2bool, default=True)
            self.parser.add_argument('--nsample', type=int, default=None)
            self.parser.add_argument('--nsampletrain', type=int, default=None)
            self.parser.add_argument('--nsampleval', type=int, default=None)
            self.parser.add_argument('--ncpu', type=int, default=8)
            self.parser.add_argument('--cropres', type=str2bool, default=True)

    @property
    def options(self):
        return self.parser.parse_args()