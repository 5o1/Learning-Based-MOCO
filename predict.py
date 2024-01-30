from typing import AnyStr

import PIL.Image as I

import torch
from ignite.engine import Engine, create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss, SSIM, PSNR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomRotation, ToTensor
from torchvision.utils import save_image

import logging

import argparse
import datetime
import os

import datasets
from models.unetComplex import UnetComplex
from transforms.fft import Ifft
from transforms.mri import FSE_readout
from transforms.h5 import H52tensor
from transforms.complex import ViewAsComplex, ViewAsReal
from transforms.process import RandSlice


best = None

def predict(model : torch.nn.Module, image: torch.Tensor , name = 'exp1', outdir = './exp/'):
    """main train steps"""
    OUTDIR = os.path.join(outdir,'predict', name)
    os.makedirs(OUTDIR, exist_ok=False)

    # configure logger
    FORMAT = '[%(asctime)s][%(name)s][%(levelname)s]%(message)s'
    LOGPATH = os.path.join(OUTDIR, 'train.log')
    DATEFORMAT =  '%Y/%m/%d %H:%M:%S'
    logging.basicConfig(format=FORMAT, datefmt = DATEFORMAT)
    logger = logging.getLogger('predict')
    logger.setLevel(logging.DEBUG)
    logfile_handler = logging.FileHandler(LOGPATH)
    logfile_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(logfile_handler)
    logger.addHandler(console_handler)

    # get divice
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image.to(device)

    # # load datasets
    # transform = Compose([
    #     H52tensor(),
    #     RandSlice(),
    #     Ifft(),
    #     FSE_randrot(movefunc = RandomRotation([0,30])),
    # ])


    # # Lost, optimizer,

    # def criterion(y_hat: torch.Tensor, y : torch.Tensor):
    #     x = y_hat - y
    #     x = x * x.conj()
    #     x = x.abs()
    #     x = x.mean()
    #     return x


    # val_metrics = {
    #     "SSIM": SSIM(1.0),
    #     "PSNR": PSNR(1.0),
    #     "loss": Loss(criterion)
    # }

    OUTIMAGEPATH = os.path.join(OUTDIR, 'outimage.bmp')
    outimage = model(image)
    save_image(outimage, OUTIMAGEPATH)

 

if __name__ == '__main__':

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # options
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--input', type=str,default=None)
    parser.add_argument('--name', type=str, default=timestamp)
    parser.add_argument('--basedir', type=str,default='./')
    parser.add_argument('--outdir', type=str, default='exp')

    options = parser.parse_args()

    outdir = os.path.join(options.basedir, options.outdir)
    expname = options.name

    image = I.open(options.input)
    image = ToTensor()(image)

    predict(
        model=UnetComplex(in_channels=4, out_channels=4, dtype=torch.complex64),
        image=image,
        name = expname,
        outdir = outdir
    )
