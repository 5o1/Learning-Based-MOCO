# sys
import os

# project
from models import Unet,Exp1_model 
from models.nn import nConv2d
from datasets import Fastmri_brain
from transforms import *
from configs import TrainingParameters
from train import train

# torch
import torch
from torchvision.transforms import Compose, RandomAffine

# metric
from torch.nn import MSELoss

import ignite.engine.engine

if __name__ == '__main__':
    # options
    options = TrainingParameters().options

    outdir = os.path.join(options.basedir, options.outdir, 'train', options.name)
    # if exists then append index to filename
    if os.path.exists(outdir):
        i = 1
        while os.path.exists(outdir + f'_{i}'):
            i += 1
        outdir = outdir + f'_{i}'
    os.makedirs(outdir)

    transforms = Compose([
        H52tensor(),
        Ifft(),
        RandSlice(),
        SquareCrop(),
        ViewAsReal_combine(),
        exp1(transforms = [
            RandomAffine(degrees = [-3,3], translate = [1e-3,1e-3]),
            RandomAffine(degrees = [-3,3], translate = [1e-3,1e-3])
        ]),
        CollectToList(x_keys=['x'], y_keys=['y', 'Ii'])
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

    model = Exp1_model(
        in_channels=8,
        out_channels=8,
        depth = options.depth,
        top_channels = options.topchannels,
        dtype=torch.float,
        crop_res=options.cropres,
        n_input = 2
        )
    
    mse_func = MSELoss()
    
    # model = nConv2d(
    #     in_channels=8,
    #     out_channels=8,
    #     n = 1,
    #     dtype=torch.float,
    #     padding = 'same'
    #     )

    def metric_transform(y_pred, y):
        # return y_pred, y
        return y_pred['y_hat'], y['y']

    def loss_fn(y_pred, y):
        loss = mse_func(y['y'], y_pred['y_hat'])
        for i in range(len(y['Ii'])):
            loss += mse_func(y['Ii'][i], y_pred['Ii_hat'][i])
        return loss              
    train(
        model = model,
        loss_fn = loss_fn,
        train_dataset = train_set,
        val_dataset = val_set,
        batch_size = options.batchsize,
        epoch = options.epoch,
        exp_name = options.name,
        save_to = outdir,
        learning_rate=options.lr,
        metric_transform=metric_transform
        # options=options
    )
