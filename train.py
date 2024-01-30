import logging
import datetime
import os
import yaml

from ignite.engine import Engine, create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss, SSIM, PSNR
from ignite.handlers import Checkpoint, global_step_from_engine

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import MSELoss
from torchinfo import summary

best = None


def train(model : torch.nn.Module = None, train_dataset: Dataset = None, val_dataset: Dataset = None, 
          batch_size: int = 16, epoch: int = 30, learning_rate :float = 5e-5 , name = 'exp1', outdir = './exp/', options = None):
    """main train steps"""
    OUTDIR = os.path.join(outdir,'train', name)
    try:
        os.makedirs(OUTDIR, exist_ok=False)
    except FileExistsError:
        OUTDIR = os.path.join(outdir,'train', name + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        os.makedirs(OUTDIR, exist_ok=False)

    # configure logger
    FORMAT = '[%(asctime)s][%(name)s][%(levelname)s]%(message)s'
    DATEFORMAT =  '%Y/%m/%d %H:%M:%S'
    formatter = logging.Formatter(fmt = FORMAT, datefmt=DATEFORMAT)
    LOGPATH = os.path.join(OUTDIR, 'train.log')

    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)

    logfile_handler = logging.FileHandler(LOGPATH)
    logfile_handler.setLevel(logging.INFO)
    logfile_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.addHandler(logfile_handler)
    logger.addHandler(console_handler)

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'|||||Trainer now using device: {device}|||||')
    model.to(device)

    # prepare dataloader
    train_loader = DataLoader(train_dataset, num_workers=8)
    val_loader = DataLoader(val_dataset, num_workers=8)
    lentrain = len(train_loader)
    lenval = len(val_loader)
    logger.info(f'Total samples: train_loader{lentrain}, val_loader{lenval}')


    # Lost, optimizer
    # def loss_fn(y_hat: torch.Tensor, y : torch.Tensor):
    #     x = y_hat - y
    #     x = x * x.conj()
    #     x = x.abs()
    #     x = x.mean()
    #     return x

    loss_fn = MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
 
    # ignite configure
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)

    val_metrics = {
        "SSIM": SSIM(1.0),
        "PSNR": PSNR(1.0),
        "loss": Loss(loss_fn)
    }



    # save args
    ARGSFILEPATH = os.path.join(OUTDIR, 'args.yaml')
    with open(ARGSFILEPATH, 'w') as f:
        args = {
            'train':{
                'name': name,
                'batch_size': batch_size,
                'epoch': epoch,
                'learning_rate': learning_rate,
                'outdir': outdir
                },
            # 'model':model.args,
            'trainset':train_dataset.args,
            'valset':val_dataset.args
        }
        args_yaml = yaml.dump(args)
        print(args_yaml)
        print(args_yaml, file = f)

    # save options
    OPTIONSPATH = os.path.join(OUTDIR, 'options.yaml')
    with open(OPTIONSPATH, 'w') as f:
        options_yaml = yaml.dump(vars(options))
        print(options_yaml)
        print(options_yaml, file = f)

    # save network architecture
    NETWORKARCHPATH = os.path.join(OUTDIR, 'model.torchinfo')
    with open(NETWORKARCHPATH, 'w') as f:
        print(
            summary(
                model = model,
                input_size=(1,8,320,320),
                depth=5,
                device=device,
                dtypes=[model.dtype]
                ),file = f)

    train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch[0].to(device), batch[1].to(device)
        # logger.debug(f"iter={engine.state.iteration}")
        # logger.debug(f"x.shape={x.shape}, y.shape={y.shape}")
        # logger.debug(f"x.isnan={torch.any(torch.isnan(x))}, y.isnan={torch.any(torch.isnan(y))}")
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        # logger.debug(f"x#y_hat={loss_fn(x,y_pred)}, x#y={loss_fn(x,y)}, y#y_hat={loss}")
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_step)

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = model(x)
            return y_pred, y

    train_evaluator = Engine(validation_step)
    val_evaluator = Engine(validation_step)

    # Attach metrics to the evaluators
    for name, metric in val_metrics.items():
        metric.attach(train_evaluator, name)

    for name, metric in val_metrics.items():
        metric.attach(val_evaluator, name)

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(engine):
        logger.debug(f"Epoch[{engine.state.epoch}/{epoch}], Iter[{engine.state.iteration}/{lentrain * engine.state.epoch}] Loss: {engine.state.output}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        logger.info(f"Training Results - Epoch[{trainer.state.epoch}/{epoch}] SSIM: {metrics['SSIM']} PSNR:{metrics['PSNR']} Avg loss: {metrics['loss']}")


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        logger.info(f"Validation Results - Epoch[{trainer.state.epoch}/{epoch}] SSIM: {metrics['SSIM']} PSNR:{metrics['PSNR']} Avg loss: {metrics['loss']}")

    # to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
    # checkpoint_dir = "checkpoints/"

    # checkpoint = Checkpoint(
    #     to_save=to_save,
    #     save_handler=OUTDIR,
    #     n_saved=1,
    #     global_step_transform=global_step_from_engine(trainer),
    # )  
    # train_evaluator.add_event_handler(Events.COMPLETED, checkpoint)

        

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def save_model(trainer):
        epoch = trainer.state.epoch
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        metrics = train_evaluator.state.metrics
        path = os.path.join(OUTDIR,f'epoch{epoch}.checkpoint')

        savepoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'loss': metrics['loss'],
            }

        torch.save(savepoint, path)
        # update last
        path = os.path.join(OUTDIR,f'last.model')
        torch.save(savepoint, path)
        
        # update best
        global best
        if best == None or best['loss'] > metrics['loss']:
            best = savepoint
            path = os.path.join(OUTDIR, f'best.model')
            torch.save(best, path)
            


    trainer.run(train_loader, max_epochs=epoch)

