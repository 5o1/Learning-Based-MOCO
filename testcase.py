"""生成一张运动受损的图片"""

from typing import AnyStr

import torch
from ignite.engine import Engine, create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss, SSIM, PSNR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomRotation, RandomAffine
import torchvision.transforms as tf
import PIL.Image as I

import logging

from utils.fft import ktoi

import argparse
import datetime
import os

import datasets
from models.unet import UNet
from transforms import *

from glob import glob

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

def to255(image :torch.Tensor):
    image = image - image.min()
    image = image / image.max()
    image = image * 255
    return image


def mse(y_hat: torch.Tensor, y : torch.Tensor):
    x = y_hat - y
    x = x * x.conj()
    x = x.abs()
    x = x.mean()
    return x


def makecase(input_path : str, name = 'exp1', outdir = './exp/'):
    OUTDIR = os.path.join(outdir,'makecase', name)
    try:
        os.makedirs(OUTDIR, exist_ok=False)
    except FileExistsError:
        OUTDIR = os.path.join(outdir,'makecase', name + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        os.makedirs(OUTDIR, exist_ok=False)

    # load datasets
    transforms = Compose([
        H52tensor(),
        Ifft(),
        RandSlice(),
        SquareCrop(),
        FSE_readout(ETL=8, probability=0.25,movefunc=RandomAffine(degrees = [-3,3], translate = [1e-3,1e-3])),
        Complex2Real(),
    ])


    train_loader = DataLoader(
    datasets.Fastmri_brain(path=input_path, transforms=transforms), batch_size=1, shuffle=True
    )

    
    x, label = next(iter(train_loader))

    # savefile
    torch.save(x, os.path.join(OUTDIR, 'x.pt'))
    torch.save(label, os.path.join(OUTDIR, 'label.pt'))

    toimage = tf.ToPILImage()

    # saveimage
    for i in range(x.shape[1]):
        casex = x[0][i]
        casex = casex.abs()
        casex = to255(casex)
        casex = casex.to(torch.uint8)
        casex = toimage(casex)
        caselabel = label[0][i]
        caselabel = caselabel.abs()
        caselabel = to255(caselabel)
        caselabel = caselabel.to(torch.uint8)
        caselabel = toimage(caselabel)
        casex.save(os.path.join(OUTDIR, f'x_{i}.bmp'))
        caselabel.save(os.path.join(OUTDIR, f'label_{i}.bmp'))

    return x, label, OUTDIR

def testcase(x, label, model : torch.nn.Module, input_path : str, weight_path :str, name = 'exp1', outdir = './exp/'):
    OUTDIR = os.path.join(outdir,'testcase', name)
    try:
        os.makedirs(OUTDIR, exist_ok=False)
    except FileExistsError:
        OUTDIR = os.path.join(outdir,'testcase', name + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        os.makedirs(OUTDIR, exist_ok=False)

    # get divice
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location = device)['model_state_dict'])

    # saveimage
    toimage = tf.ToPILImage()
    for i in range(x.shape[1]):
        casex = x[0][i]
        casex = casex.abs()
        casex = to255(casex)
        casex = casex.to(torch.uint8)
        casex = toimage(casex)
        caselabel = label[0][i]
        caselabel = caselabel.abs()
        caselabel = to255(caselabel)
        caselabel = caselabel.to(torch.uint8)
        caselabel = toimage(caselabel)
        casex.save(os.path.join(OUTDIR, f'x_{i}.bmp'))
        caselabel.save(os.path.join(OUTDIR, f'label_{i}.bmp'))

    model.eval()
    with torch.no_grad():
        y_hat = model(x)

    for i in range(y_hat.shape[1]):
        casey_hat = y_hat[0][i]
        casey_hat = casey_hat.abs()
        casey_hat = to255(casey_hat)
        casey_hat = casey_hat.to(torch.uint8)
        casey_hat = toimage(casey_hat)
        casey_hat.save(os.path.join(OUTDIR, f'y_hat_{i}.bmp'))

    # metrix
    ssim = SSIM(data_range=1.0)
    ssim.attach(default_evaluator, 'ssim')
    psnr = PSNR(data_range=1.0)
    psnr.attach(default_evaluator, 'psnr')
    state = default_evaluator.run([[y_hat, label]])
    print(f'SSIM : {state.metrics["ssim"]}, PSNR : {state.metrics["psnr"]}, MSE : {mse(y_hat, label)}')

if __name__=="__main__":
    x, label, _ = makecase('/mnt/nfs_datasets/fastMRI_brain/multicoil_train_sorted/size_320_640_4/')
    model = UNet(
        in_channels=8,
        out_channels=8,
        depth = 6,
        top_channels = 32,
        dtype=torch.float,
        crop_res=False
    )
    testcase(x,label,model,'','/home/assaneko/studio/moco_fastmri/notebook/resources/baseline_33/best.model')