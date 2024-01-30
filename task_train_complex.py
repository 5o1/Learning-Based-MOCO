# sys
import argparse
import datetime
import os
from configs import TrainingParameters

# project
from models.unetComplex import UnetComplex
from train import train

# torch
import torch


if __name__ == '__main__':

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # options
    options = TrainingParameters().options

    outdir = os.path.join(options.basedir, options.outdir)

    train(
        model=UnetComplex(in_channels=4, out_channels=4, dtype=torch.complex64),
        train_path=options.trainpath,
        val_path=options.valpath,
        name = options.name,
        outdir = outdir
    )
