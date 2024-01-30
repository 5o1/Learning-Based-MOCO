
import argparse
import os
import sys
import datetime
import time

from torchvision.transforms import Compose, RandomAffine

from datasets.fastmri_brain import Fastmri_brain
from transforms import *

from torch.utils.data import DataLoader


sys.stdout = sys.stderr

if __name__ == '__main__':

    # options
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--indir', type=str)
    parser.add_argument('--outdir', type=str, default=None)

    options = parser.parse_args()

    transforms = Compose([
        H52tensor(),
        DoubleCheck(tag = 0),
        Ifft(),
        DoubleCheck(tag = 1),
        RandSlice(),
        DoubleCheck(tag = 2),
        FSE_readout(ETL=8, probability=0.25,movefunc=RandomAffine(degrees = [3,3], translate = [1e-3,1e-3])),
        DoubleCheck(tag = 3),
        ViewAsReal_combine(),
        DoubleCheck(tag = 4)
    ])


    dataset = Fastmri_brain(path = options.indir, transforms=transforms)
    if options.outdir is not None:
        OUTDIR = options.outdir
        os.makedirs(OUTDIR, exist_ok=False)

    loader = DataLoader(dataset, num_workers=8, shuffle=False)

    pretime = time.process_time()
    rounds = 100
    for r in range(rounds):
        print(f'Round {r}:')
        for i,_ in enumerate(iter(loader)):
            nowtime = time.process_time()
            print(f'Processing of the {i}^th sample is complete. Time spent: {nowtime-pretime}')
            pretime = nowtime

    print('End.')