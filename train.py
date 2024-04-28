"""The main steps in training a model

This file should contain the code that implements the decoupling of the model architecture.
I.e., there is no need to change the internal code of train() to achieve compatibility for training any model.

# Author: 5o1
"""


import logging
from typing import Callable
import os

from ignite.engine import Engine, create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss, SSIM, PSNR
from ignite.handlers import Checkpoint, global_step_from_engine

import torch
from torch.utils.data import DataLoader, Dataset
import torchinfo

from utils import Profile, DictFilter

from matplotlib import pyplot as plt


# Configurations
LOG_LEVEL = logging.DEBUG
LOG_CONSOLE_LEVEL = logging.DEBUG
LOG_FILE_LEVEL = logging.INFO
LOG_FORMAT = '[%(asctime)s][%(name)s][%(levelname)s]%(message)s'
LOG_DATEFORMAT =  '%Y/%m/%d %H:%M:%S'

TASK_NAME = 'train'


def train(
        exp_name : str, batch_size: int, epoch: int, learning_rate :float, save_to : str,
        loss_fn: Callable,
        model : torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset, 
        for_dump: dict = None,
        x_keys : list = [0],
        y_keys : list = [1],
        metric_transform : Callable = lambda x : x,
        testshow_transform : Callable = lambda x : x,
        tensorboard_on : bool = True
        ):
    
    #########################
    # Configuring the logger begin
    # Configure logger
    formatter = logging.Formatter(fmt = LOG_FORMAT, datefmt=LOG_DATEFORMAT)
    logger = logging.getLogger(TASK_NAME)
    logger.setLevel(LOG_LEVEL)
    # Console log handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_CONSOLE_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # File log handler
    LOG_PATH = os.path.join(save_to, f'{TASK_NAME}.log')
    if os.path.exists(LOG_PATH):
        msg = f'File {LOG_PATH} exists!'
        logger.error(msg)
        raise FileExistsError(msg)
    logfile_handler = logging.FileHandler(LOG_PATH)
    logfile_handler.setLevel(LOG_FILE_LEVEL)
    logfile_handler.setFormatter(formatter)
    logger.addHandler(logfile_handler)
    # Configuring the logger end
    #########################

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f'|||||Trainer now using device: {device}|||||')
    model.to(device)

    # Prepare dataloader
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=8, batch_size=batch_size, shuffle=True)
    logger.debug(f'Total samples: train_loader{len(train_loader)}, val_loader{len(val_loader)}')


    #########################
    # Data Processing Utilities begin
    def output_transform(output):
        """Transform output to metric input."""
        return metric_transform(output[0], output[1])
    
    def to_device(data, device):
        """Recursively move data to device."""
        if isinstance(data, dict):
            return {key: to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [to_device(item, device) for item in data]
        elif torch.is_tensor(data):
            return data.to(device)
        else:
            return data

    def get_data(batch: torch.Tensor | dict, keys: list | int | str):
        """Get data from batch"""
        if isinstance(keys, (str, int)):
            return batch[keys]
        if isinstance(keys, list):
            if len(keys) == 1 and isinstance(keys[0], (str, int)):
                return batch[keys[0]]
            if all(isinstance(key, int) for key in keys):
                return [get_data(batch, key) for key in keys]
            if all(isinstance(key, str) for key in keys):
                return {key:get_data(batch, key) for key in keys}
        logger.error(f'keys={keys} is not supported.')
        raise NotImplementedError(f'keys={keys} is not supported.')
    # Data Processing Utilities end
    #########################

    # Prepare metrics
    metrics = {
        "SSIM": SSIM(1.0, output_transform = output_transform, device=device),
        "PSNR": PSNR(1.0, output_transform = output_transform, device=device),
        "loss": Loss(loss_fn, device=device)
    }

    #########################
    # Saving Experiment Configurations begin
    # Save args.yaml to file
    ARGS_PATH = os.path.join(save_to, 'args.yaml')
    args_profile = Profile(
        profile={
            TASK_NAME:{
            'exp_name':exp_name,
            'batch_size':batch_size,
            'epoch':epoch,
            'learning_rate':learning_rate,
            'device':device,
            'save_to':save_to,
            'loss_fn':loss_fn,
            # 'model':model,
            'train_dataset':train_dataset,
            'val_dataset':val_dataset,
            }
        },
        filter=DictFilter()
        )
    args_profile.dump(save_to=ARGS_PATH)
    logger.debug(args_profile.get())

    # Save model.torchinfo to file
    NETWORKARCH_PATH = os.path.join(save_to, 'model_architecture.yaml')
    summary_profile = Profile(profile=str(torchinfo.summary(
            model = model,
            input_data=[get_data(next(iter(train_loader)), x_keys)],
            depth=5,
            device=device,
    )))
    summary_profile.dump(save_to=NETWORKARCH_PATH)
    logger.debug(summary_profile.get())

    # Save for_dump[keys] to file
    if for_dump is not None:
        for key, value in for_dump:
            DUMP_PATH = os.path.join(save_to, key)
            value_profile = Profile(profile=value,filter=DictFilter())
            value_profile.dump(save_to=DUMP_PATH)
            logger.debug(value_profile.get())

    # Saving Experiment Configurations end
    #########################

    train_history = []
    val_history = []

    #########################
    # Main training steps begin
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
 
    # Ignite configure
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    def train_step(engine, batch):
        """Train step."""
        model.train()
        optimizer.zero_grad()
        x, y = get_data(batch, x_keys), get_data(batch, y_keys)
        x = to_device(x, device)
        y = to_device(y, device)

        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()
        # return y_pred, y

    trainer = Engine(train_step)
    # metric = Loss(loss_fn=loss_fn, device=device)
    # metric.attach(train_step, "Loss")

    def validation_step(engine, batch):
        """Validation step."""
        model.eval()
        with torch.no_grad():
            x, y = get_data(batch, x_keys), get_data(batch, y_keys)
            x = to_device(x, device)
            y = to_device(y, device)
            y_pred = model(x)
            return y_pred, y

    train_evaluator = Engine(validation_step)
    val_evaluator = Engine(validation_step)

    # Attach metrics to the evaluators
    for name, metric in metrics.items():
        # metric.attach(train_evaluator, name)
        if name == 'loss':
            metric.attach(train_evaluator, name)

    for name, metric in metrics.items():
        metric.attach(val_evaluator, name)

    # @trainer.on(Events.ITERATION_COMPLETED(every=10))
    # def log_training_loss(engine):
    #     logger.debug(f"Epoch[{engine.state.epoch}/{epoch}], Iter[{engine.state.iteration}/{len(train_loader) * engine.state.epoch}] Loss: {engine.state.output}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        """Log training results."""
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        # logger.info(f"Training Results - Epoch[{trainer.state.epoch}/{epoch}] SSIM: {metrics['SSIM']} PSNR:{metrics['PSNR']} Avg loss: {metrics['loss']}")
        logger.info(f"Training Results - Epoch[{trainer.state.epoch}/{epoch}] Avg loss: {metrics['loss']}")
        train_history.append(metrics)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        """Log validation results."""
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        logger.info(f"Validation Results - Epoch[{trainer.state.epoch}/{epoch}] SSIM: {metrics['SSIM']} PSNR:{metrics['PSNR']} Avg loss: {metrics['loss']}")
        val_history.append(metrics)

    # to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
    # checkpoint_dir = "checkpoints/"

    # checkpoint = Checkpoint(
    #     to_save=to_save,
    #     save_handler=OUTDIR,
    #     n_saved=1,
    #     global_step_transform=global_step_from_engine(trainer),
    # )  
    # train_evaluator.add_event_handler(Events.COMPLETED, checkpoint)

        
    best = None
    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def save_model(trainer):
        """Save model."""
        epoch = trainer.state.epoch
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        metrics = train_evaluator.state.metrics
        path = os.path.join(save_to,f'epoch{epoch}.checkpoint')

        savepoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'loss': metrics['loss'],
            }

        torch.save(savepoint, path)
        # Update last
        path = os.path.join(save_to,f'last.model')
        torch.save(savepoint, path)
        
        # Update best
        nonlocal best
        if best == None or best['loss'] > metrics['loss']:
            best = savepoint
            path = os.path.join(save_to, f'best.model')
            torch.save(best, path)
        
    # Main training steps end
    #########################

    #########################
    # Tensor board logging begin

    # Todo
    if tensorboard_on:
        from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler, WeightsScalarHandler, WeightsHistHandler
        TB_LOGS_PATH = os.path.join(save_to, 'tb_logs')
        tb_logger = TensorboardLogger(log_dir= TB_LOGS_PATH)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", output_transform=lambda loss: {'loss': loss}), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(trainer, log_handler=WeightsScalarHandler(model), event_name=Events.ITERATION_COMPLETED)
        # tb_logger.attach(trainer, log_handler=WeightsHistHandler(model), event_name=Events.EPOCH_COMPLETED)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", output_transform=lambda x: {'loss': x}), event_name=Events.EPOCH_COMPLETED)
        tb_logger.attach(val_evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

        import multiprocessing, subprocess
        TB_PORT = 56006
        tb_process = multiprocessing.Process(target=lambda tb_logs, port : subprocess.run(['tensorboard', '--logdir', tb_logs, '--port', str(port)]), args=(TB_LOGS_PATH, TB_PORT))
        tb_process.daemon = True
        tb_process.start()
    # Tensor board logging end
    #########################

    # Trian
    trainer.run(train_loader, max_epochs=epoch)


    #########################
    # Training results and evaluations begin

    # Todo : loss curve, metric curve, testcase(image test) and metrics

    # loss curve
    fig, axs = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    for i, key, value in enumerate(metrics.items()):
        if key in train_history[0].keys():
            axs[i].plot([x[key] for x in train_history], label='train', color='darkorange')
        if key in val_history[0].keys():
            axs[i].plot([x[key] for x in val_history], label='val', color = 'dodgerblue')
        axs[i].legend()
        axs[i].set_title(f'{key}')
    
    fig.savefig(os.path.join(save_to, 'loss_curve.png'))

    # Testcase
    with torch.no_grad():
        x, y = get_data(next(iter(val_loader)), x_keys), get_data(next(iter(val_loader)), y_keys)
        x = to_device(x, device)
        y = to_device(y, device)
        y_pred = model(x)
        y_pred = y_pred.cpu().numpy()
        y = y.cpu().numpy()
        y, y_pred = testshow_transform(y), testshow_transform(y_pred)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(y[0])
        axs[0].set_title('Ground Truth')
        axs[1].imshow(y_pred[0])
        axs[1].set_title('Prediction')
        fig.savefig(os.path.join(save_to, 'testcase.png'))


    # Training results and evaluations begin
    #########################