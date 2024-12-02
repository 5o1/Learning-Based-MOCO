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
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver

import torch
from torch.utils.data import DataLoader, Dataset

from utils import Profile, DictFilter

from matplotlib import pyplot as plt

from datetime import datetime


# Configurations
LOG_LEVEL = logging.DEBUG
LOG_CONSOLE_LEVEL = logging.DEBUG
LOG_FILE_LEVEL = logging.INFO
LOG_FORMAT = '[%(asctime)s][%(name)s][%(levelname)s]%(message)s'
LOG_DATEFORMAT =  '%Y/%m/%d %H:%M:%S'

TASK_NAME = 'train'


#########################
# Data Processing Utilities begin
# def output_transform(output):
#     """Transform output to metric input."""
#     return output_transform(output[0], output[1])

def _to_device(data, device):
    """Recursively move data to device."""
    if isinstance(data, dict):
        return {key: _to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [_to_device(item, device) for item in data]
    elif torch.is_tensor(data):
        return data.to(device)
    else:
        return data

# def get_data(batch: torch.Tensor | dict, keys: list | int | str):
#     """Get data from batch"""
#     if isinstance(keys, (str, int)):
#         return batch[keys]
#     if isinstance(keys, list):
#         if len(keys) == 1 and isinstance(keys[0], (str, int)):
#             return batch[keys[0]]
#         if all(isinstance(key, int) for key in keys):
#             return [get_data(batch, key) for key in keys]
#         if all(isinstance(key, str) for key in keys):
#             return {key:get_data(batch, key) for key in keys}
#     logger.error(f'keys={keys} is not supported.')
#     raise NotImplementedError(f'keys={keys} is not supported.')
# Data Processing Utilities end
#########################



def train(
        exp_name : str, batch_size: int, epoch: int, learning_rate :float, exp_basedir : str,
        loss_fn: Callable,
        model : torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset, 
        epoch_length: int = None,
        eval_length_train: int = None,
        eval_length_val: int = None,
        loader_num_workers: int = 8,
        # for_dump: dict = None, # Todo
        enhance_transform : Callable = lambda x : x,
        input_transform : Callable = lambda x : x,
        output_transform : Callable = lambda x : x,
        show_transform : Callable = None,
        checkpoint_every_epoch : int = 10,
        maxnum_checkpoints : int = 10,
        validate_every_epoch : int = 1,
        device : torch.device | str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        to_device : Callable = _to_device,
        tensorboard_basedir : str = None,
        # continue_from : str = None, # Todo
        ):
    #########################
    # parameter check begin
    if epoch_length is None:
        epoch_length = len(train_dataset)

    if eval_length_train is None:
        eval_length_train = len(train_dataset)

    if eval_length_val is None:
        eval_length_val = len(val_dataset)

    #########################
    # Path
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    save_to = os.path.join(exp_basedir, f"{exp_name}_{TASK_NAME}_{timestamp}")

    assert not os.path.exists(save_to), f'{save_to} already exists!'
    os.makedirs(save_to, exist_ok=True)

    LOG_PATH = os.path.join(save_to, f'{TASK_NAME}.log')
    ARGS_PATH = os.path.join(save_to, 'args.yaml')
    NETWORKARCH_PATH = os.path.join(save_to, 'model_architecture.yaml')
    DUMP_PATH = os.path.join(save_to, key)
    TENSORBOARD_LOG_PATH = tensorboard_basedir
    
    #########################
    # logger configuration begin
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
    if os.path.exists(LOG_PATH):
        msg = f'File {LOG_PATH} exists!'
        logger.error(msg)
        raise FileExistsError(msg)
    logfile_handler = logging.FileHandler(LOG_PATH)
    logfile_handler.setLevel(LOG_FILE_LEVEL)
    logfile_handler.setFormatter(formatter)
    logger.addHandler(logfile_handler)
    # logger configuration end
    #########################

    #########################
    # tensorboard configuration begin
    if tensorboard_basedir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning('torch.utils.tensorboard is not installed. Please install torch.utils.tensorboard to use tensorboard.')
            SummaryWriter = None
        else:
            writer = SummaryWriter(log_dir=TENSORBOARD_LOG_PATH)
    # tensorboard configuration end
    #########################

    # Get device
    logger.debug(f'|||||Trainer now using device: {device}|||||')
    model.to(device)

    # Prepare dataloader
    train_loader = DataLoader(train_dataset, num_workers=loader_num_workers, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=loader_num_workers, batch_size=batch_size, shuffle=True)
    logger.debug(f'Total samples: train_loader{len(train_loader)}, val_loader{len(val_loader)}')

    # Prepare metrics
    metrics = {
        "SSIM": SSIM(1.0, output_transform = output_transform, device=device),
        "PSNR": PSNR(1.0, output_transform = output_transform, device=device),
        "loss": Loss(loss_fn, device=device)
    }

    #########################
    # Saving Experiment Configurations begin
    # Save args.yaml to file
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

    # If torchinfo is installed, save model.torchinfo to file
    try:
        import torchinfo
    except ImportError:
        logger.warning('torchinfo is not installed. Please install torchinfo to get model summary.')
        torchinfo = None
    else:
        # Save model.torchinfo to file
        summary_profile = Profile(profile=str(torchinfo.summary(
                model = model,
                input_data=[input_transform(next(iter(train_loader))[0])],
                depth=5,
                device=device,
        )))
        summary_profile.dump(save_to=NETWORKARCH_PATH)
        logger.debug(summary_profile.get())

    # # Save for_dump[keys] to file
    # if for_dump is not None:
    #     for key, value in for_dump:
    #         value_profile = Profile(profile=value,filter=DictFilter())
    #         value_profile.dump(save_to=DUMP_PATH)
    #         logger.debug(value_profile.get())

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
        if not model.training:
            model.train()

        batch = to_device(batch, device)
        # gpu
        with torch.no_grad(): # prepare
            batch = enhance_transform(batch)
            x, y = input_transform(batch)

        optimizer.zero_grad()

        pred = model(x)

        loss_score = loss_fn(pred, y)
        loss_score.backward()
        optimizer.step()

        if tensorboard_basedir is not None:
            writer.add_scalar('iteration/loss', loss_score.item(), engine.state.iteration)

        return loss_score.item()

    trainer = Engine(train_step)

    def validation_step(engine, batch):
        """Validation step. Note that this function and train_step are not the difference between the training set and the test set, that is, train_step is used for gradient descent and validation_step is used to calculate metrics"""
        if model.training:
            model.eval()
        
        batch = to_device(batch, device)
        # gpu
        with torch.no_grad(): # prepare
            batch = enhance_transform(batch)
            x, y = input_transform(batch)

            pred = model(x)
            return pred, y

    train_evaluator = Engine(validation_step) # Note that metrics and loss are not the same thing.
    val_evaluator = Engine(validation_step)

    # Attach metrics to the evaluators
    for name, metric in metrics.items():
        metric.attach(train_evaluator, name)

    for name, metric in metrics.items():
        metric.attach(val_evaluator, name)

    # @trainer.on(Events.ITERATION_COMPLETED(every=10))
    # def log_training_loss(engine):
    #     logger.debug(f"Epoch[{engine.state.epoch}/{epoch}], Iter[{engine.state.iteration}/{len(train_loader) * engine.state.epoch}] Loss: {engine.state.output}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        """Log training results."""
        train_evaluator.run(train_loader, epoch_length=eval_length_train)
        metrics = train_evaluator.state.metrics
        logger.info(f"Training Results - Epoch[{trainer.state.epoch}/{epoch}] SSIM: {metrics['SSIM']} PSNR:{metrics['PSNR']} Avg loss: {metrics['loss']}")

        if tensorboard_basedir is not None:
            writer.add_scalar('training/SSIM', metrics['SSIM'], trainer.state.epoch)
            writer.add_scalar('training/PSNR', metrics['PSNR'], trainer.state.epoch)
            writer.add_scalar('training/loss', metrics['loss'], trainer.state.epoch)

        train_history.append(metrics.items())


    @trainer.on(Events.EPOCH_COMPLETED(every=validate_every_epoch))
    def log_validation_results(trainer):
        """Log validation results."""
        val_evaluator.run(val_loader, epoch_length=eval_length_val)
        metrics = val_evaluator.state.metrics
        logger.info(f"Validation Results - Epoch[{trainer.state.epoch}/{epoch}] SSIM: {metrics['SSIM']} PSNR:{metrics['PSNR']} Avg loss: {metrics['loss']}")
        if tensorboard_basedir is not None:
            writer.add_scalar('validation/SSIM', metrics['SSIM'], trainer.state.epoch)
            writer.add_scalar('validation/PSNR', metrics['PSNR'], trainer.state.epoch)
            writer.add_scalar('validation/loss', metrics['loss'], trainer.state.epoch)

        val_history.append(metrics.items())

    # checkpoint
    _to_save_dict = {'model': model, 'optimizer': optimizer}

    every_n_checkpointer = Checkpoint(
        to_save=_to_save_dict,
        save_handler=DiskSaver(save_to,create_dir=True, require_empty=False),
        filename_prefix='checkpoint',
        n_saved=maxnum_checkpoints,
        global_step_transform=global_step_from_engine(trainer),
    )

    best_loss_checkpointer = Checkpoint(
        to_save=_to_save_dict,
        save_handler=DiskSaver(save_to,create_dir=True, require_empty=False),
        filename_prefix='best',
        n_saved=1,
        score_name='loss',
        score_function=lambda engine: -engine.state.metrics['loss'],
        global_step_transform=global_step_from_engine(trainer),
    )

    best_ssim_checkpointer = Checkpoint(
        to_save=_to_save_dict,
        save_handler=DiskSaver(save_to,create_dir=True, require_empty=False),
        filename_prefix='best',
        n_saved=1,
        score_name='ssim',
        score_function=lambda engine: engine.state.metrics['ssim'],
        global_step_transform=global_step_from_engine(trainer),
    )
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_every_epoch), every_n_checkpointer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, best_loss_checkpointer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, best_ssim_checkpointer)
        
    # best_score = None
    # @trainer.on(Events.EPOCH_COMPLETED(every=checkpoint_every_epoch))
    # def save_model(trainer):
    #     """Save model."""
    #     epoch = trainer.state.epoch
    #     model_state_dict = model.state_dict()
    #     optimizer_state_dict = optimizer.state_dict()
    #     metrics = train_evaluator.state.metrics
    #     path = os.path.join(save_to,f'epoch{epoch}.checkpoint')

    #     savepoint = {
    #         'epoch': epoch,
    #         'model_state_dict': model_state_dict,
    #         'optimizer_state_dict': optimizer_state_dict,
    #         'loss': metrics['loss'],
    #         }

    #     torch.save(savepoint, path)
    #     # Update last
    #     path = os.path.join(save_to,f'last.model')
    #     torch.save(savepoint, path)
        
    #     # Update best
    #     nonlocal best_score
    #     if best_score == None or best_score['loss'] > metrics['loss']:
    #         best_score = savepoint
    #         path = os.path.join(save_to, f'best.model')
    #         torch.save(best_score, path)
        
    # Main training steps end
    #########################

    # Trian
    trainer.run(train_loader, max_epochs=epoch, epoch_length=epoch_length)


    #########################
    # Training results and evaluations begin

    # Todo : loss curve, metric curve, testcase(image test) and metrics

    # loss curve
    fig, axs = plt.subplots(1, len(metrics), figsize=(10*len(metrics), 10))
    for i, key in enumerate(metrics.keys()):
        if key in train_history[0].keys():
            axs[i].plot([x[key] for x in train_history], label='train', color='darkorange')
        if key in val_history[0].keys():
            axs[i].plot([x[key] for x in val_history], label='val', color = 'dodgerblue')
        axs[i].legend()
        axs[i].set_title(f'{key}')
    
    fig.savefig(os.path.join(save_to, 'loss_curve.png'))

    # Testcase
    train_loader_shown = DataLoader(train_dataset, num_workers=loader_num_workers, batch_size=1, shuffle=True)
    val_loader_shown = DataLoader(val_dataset, num_workers=loader_num_workers, batch_size=1, shuffle=True)

    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader_shown))
        batch = to_device(batch, device)
        batch = enhance_transform(batch)
        x, y = input_transform(batch)
        pred = model(x)

        x, pred, y = map(lambda x: x.cpu().detach().numpy(), show_transform(x, pred, y))

        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        axs[0].imshow(x[0])
        axs[0].set_title('input')
        axs[0].axis('off')
        axs[1].imshow(pred[0])
        axs[1].set_title('pred')
        axs[0].axis('off')
        axs[2].imshow(y[0])
        axs[2].set_title('gt')
        axs[0].axis('off')

        fig.savefig(os.path.join(save_to, 'testcase.png'))
    # Training results and evaluations end
    #########################