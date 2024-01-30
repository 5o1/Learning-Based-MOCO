# sys
import os

# project
from models import Unet
from models.nn import nConv2d
from datasets import Fastmri_brain
from transforms import *
from configs import TrainingParameters
from train import train

# torch
import torch
from torchvision.transforms import Compose, RandomAffine

if __name__ == '__main__':
    # options
    options = TrainingParameters().options

    outdir = os.path.join(options.basedir, options.outdir)

    transforms = Compose([
        H52tensor(),
        Ifft(),
        RandSlice(),
        SquareCrop(),
        FSE_readout(ETL=8, probability=0.25,movefunc=RandomAffine(degrees = [-3,3], translate = [1e-3,1e-3])),
        ViewAsReal_combine(),
    ])

    train_set = Fastmri_brain(
        path=options.trainpath,
        transforms=transforms, 
        n_subset = options.nsampletrain if options.nsampletrain is not None else options.nsample,
        load_from_memory=options.cache,
        n_jobs=options.ncpu
        )
    
    val_set = Fastmri_brain(
        path=options.valpath,
        transforms=transforms,
        n_subset = options.nsampleval if options.nsampleval is not None else options.nsample // 4 if options.nsample is not None else None,
        load_from_memory=options.cache,n_jobs=options.ncpu
        )

    model = Unet(
        in_channels=8,
        out_channels=8,
        depth = options.depth,
        top_channels = options.topchannels,
        dtype=torch.float,
        crop_res=options.cropres
        )
    
    # model = nConv2d(
    #     in_channels=8,
    #     out_channels=8,
    #     n = 1,
    #     dtype=torch.float,
    #     padding = 'same'
    #     )

    train(
        model = model,
        train_dataset = train_set,
        val_dataset = val_set,
        batch_size = options.batchsize,
        epoch = options.epoch,
        name = options.name,
        outdir = outdir,
        learning_rate=options.lr,
        options=options
    )
