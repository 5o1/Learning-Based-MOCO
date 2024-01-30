import logging
import os
import yaml

from ignite.engine import Engine, create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss, SSIM, PSNR
from ignite.handlers import Checkpoint, global_step_from_engine

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchinfo

from utils import DictFilter

LOG_LEVEL = logging.DEBUG
LOG_CONSOLE_LEVEL = logging.DEBUG
LOG_FILE_LEVEL = logging.INFO
LOG_FORMAT = '[%(asctime)s][%(name)s][%(levelname)s]%(message)s'
LOG_DATEFORMAT =  '%Y/%m/%d %H:%M:%S'

TASK_NAME = 'train'

best = None


def train(
        exp_name : str, batch_size: int, epoch: int, learning_rate :float, save_to : str,
        loss_fn: callable,
        model : torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset, 
        for_dump: dict = None,
        x_keys : list = [0],
        y_keys : list = [1],
        ):
    """main train steps"""
    # configure logger
    formatter = logging.Formatter(fmt = LOG_FORMAT, datefmt=LOG_DATEFORMAT)
    logger = logging.getLogger(TASK_NAME)
    logger.setLevel(LOG_LEVEL)
    # console log handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_CONSOLE_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # file log handler
    LOG_PATH = os.path.join(save_to, f'{TASK_NAME}.log')
    if os.path.exists(LOG_PATH):
        msg = f'File {LOG_PATH} exists!'
        logger.error(msg)
        raise FileExistsError(msg)
    logfile_handler = logging.FileHandler(LOG_PATH)
    logfile_handler.setLevel(LOG_FILE_LEVEL)
    logfile_handler.setFormatter(formatter)
    logger.addHandler(logfile_handler)

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f'|||||Trainer now using device: {device}|||||')
    model.to(device)

    # prepare dataloader
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=8, batch_size=batch_size, shuffle=True)
    logger.debug(f'Total samples: train_loader{len(train_loader)}, val_loader{len(val_loader)}')

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
 
    # ignite configure
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)

    val_metrics = {
        "SSIM": SSIM(1.0, device=device),
        "PSNR": PSNR(1.0, device=device),
        "loss": Loss(loss_fn, device=device)
    }

    def get_data(batch: torch.Tensor | dict, keys: list | int | str):
        """get data from batch"""
        if isinstance(keys, (str, int)):
            return batch[keys].to(device)
        if isinstance(keys, list):
            if len(keys) == 1 and isinstance(keys[0], (str, int)):
                return batch[keys[0]].to(device)
            if all(isinstance(key, int) for key in keys):
                return [get_data(batch, key) for key in keys]
            if all(isinstance(key, str) for key in keys):
                return {key:get_data(batch, key) for key in keys}
        logger.error(f'keys={keys} is not supported.')
        raise NotImplementedError(f'keys={keys} is not supported.')

    dictFilter = DictFilter()

    # save args.yaml to file
    ARGS_PATH = os.path.join(save_to, 'args.yaml')
    if os.path.exists(ARGS_PATH):
        msg = f'File {ARGS_PATH} exists!'
        logger.error(msg)
        raise FileExistsError(msg)
    with open(ARGS_PATH, 'w') as f:
        args = {
            TASK_NAME:{
                'exp_name':exp_name,
                'batch_size':batch_size,
                'epoch':epoch,
                'learning_rate':learning_rate,
                'device':device,
                'save_to':save_to,
                'loss_fn':loss_fn,
                'model':model,
                'train_dataset':train_dataset,
                'val_dataset':val_dataset,
            }
        }
        args = dictFilter(args)
        args_yaml = yaml.dump(args, default_flow_style= False, sort_keys= False)
        logger.debug(args_yaml)
        print(args_yaml, file = f)

    # save model.torchinfo to file
    NETWORKARCH_PATH = os.path.join(save_to, 'model.torchinfo')
    if os.path.exists(NETWORKARCH_PATH):
        msg = f'File {NETWORKARCH_PATH} exists!'
        logger.error(msg)
        raise FileExistsError(msg)
    with open(NETWORKARCH_PATH, 'w') as f:
        summary = torchinfo.summary(
                model = model,
                input_data=get_data(next(iter(train_loader)), x_keys),
                depth=5,
                device=device,
        )
        logger.debug(summary)
        print(summary, file = f)

    # save for_dump[keys] to file
    if for_dump is not None:
        for key, value in for_dump:
            DUMP_PATH = os.path.join(save_to, key)
            if os.path.exists(DUMP_PATH):
                msg = f'File {DUMP_PATH} exists!'
                logger.error(msg)
                raise FileExistsError(msg)
            with open(DUMP_PATH, 'w') as f:
                if isinstance(value, dict):
                    value = dictFilter(value)
                    value = yaml.dump(value, default_flow_style= False, sort_keys= False)
                print(value, file = f)

    train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = get_data(batch, x_keys), get_data(batch, y_keys)
        if isinstance(x, torch.Tensor):
            y_pred = model(x)
        elif isinstance(x, list):
            y_pred = model(*x)
        elif isinstance(x, dict):
            y_pred = model(**x)
        else:
            logger.error(f'type(x)={type(x)} is not supported.')
            raise NotImplementedError(f'type(x)={type(x)} is not supported.')
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_step)

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = get_data(batch, x_keys), get_data(batch, y_keys)
            if isinstance(x, torch.Tensor):
                y_pred = model(x)
            elif isinstance(x, list):
                y_pred = model(*x)
            elif isinstance(x, dict):
                y_pred = model(**x)
            else:
                logger.error(f'type(x)={type(x)} is not supported.')
                raise NotImplementedError(f'type(x)={type(x)} is not supported.')
            return y_pred, y

    train_evaluator = Engine(validation_step)
    val_evaluator = Engine(validation_step)

    # Attach metrics to the evaluators
    for name, metric in val_metrics.items():
        metric.attach(train_evaluator, name)

    for name, metric in val_metrics.items():
        metric.attach(val_evaluator, name)

    # @trainer.on(Events.ITERATION_COMPLETED(every=10))
    # def log_training_loss(engine):
    #     logger.debug(f"Epoch[{engine.state.epoch}/{epoch}], Iter[{engine.state.iteration}/{len(train_loader) * engine.state.epoch}] Loss: {engine.state.output}")

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

