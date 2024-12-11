"""The main steps in training a model

This file should contain the code that implements the decoupling of the model architecture.
I.e., there is no need to change the internal code of train() to achieve compatibility for training any model.

# Author: 5o1

# TODO:
- [ ] Implement continue from checkpoint

"""


import logging
from typing import Callable
import os
import pandas as pd

from ignite.engine import Engine, create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss, SSIM, PSNR
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver, TensorboardLogger

from matplotlib.pylab import f
import torch
from torch.utils.data import DataLoader, Dataset

from utils import Profile, DictFilter, get_time_diff

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
        model : torch.nn.Module,
        train_dataset: Dataset,
        loss_fn: Callable,
        epochs: int,
        lr :float = 0.01,
        exp_name : str = 'task',
        batch_size: int = 8,  
        exp_basedir : str = 'exp',
        val_dataset: Dataset = None, 
        epoch_length: int = None,
        eval_length_train: int = None,
        eval_length_val: int = None,
        accumulation_steps : int = 1,
        loader_num_workers: int = 8,
        augmentation_transform : Callable = lambda batch : batch,
        input_transform : Callable = lambda batch : (batch, batch),
        output_transform : Callable = lambda output : (output[0], output[1]),
        show_transform : Callable = lambda batch, pred, gt : (batch, pred, gt),
        checkpoint_every_epoch : int = 10,
        maxnum_checkpoints : int = 10,
        validate_every_epoch : int = 1,
        device : torch.device | str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        to_device : Callable = _to_device,
        tensorboard_basedir : str = '.tb_logs',
        # continue_from : str = None, # Todo
        ):


    #########################
    # parameter check begin
    if val_dataset is None:
        val_dataset = train_dataset

    if epoch_length is None:
        epoch_length = len(train_dataset)

    if eval_length_train is None:
        eval_length_train = len(train_dataset)

    if eval_length_val is None:
        eval_length_val = len(val_dataset)



    #########################
    # Path
    start_time = datetime.now()

    UNINAME = f"{exp_name}_{TASK_NAME}_{start_time.strftime('%Y%m%d%H%M%S')}"

    save_to = os.path.join(exp_basedir, UNINAME)

    assert not os.path.exists(save_to), f'{save_to} already exists!'
    os.makedirs(save_to, exist_ok=True)

    LOG_PATH = os.path.join(save_to, f'{TASK_NAME}.log')
    ARGS_PATH = os.path.join(save_to, 'args.yaml')
    NETWORKARCH_PATH = os.path.join(save_to, 'model.torchinfo')
    TENSORBOARD_LOG_PATH = os.path.join(tensorboard_basedir, UNINAME) if tensorboard_basedir is not None else None
    TRAINHISTORY_PATH = os.path.join(save_to, 'train_history.csv')
    VALHISTORY_PATH = os.path.join(save_to, 'val_history.csv')

    
    #########################
    # logger configuration begin
    # Configure logger
    formatter = logging.Formatter(fmt = LOG_FORMAT, datefmt=LOG_DATEFORMAT)
    # logger = logging.getLogger(TASK_NAME)
    logger = logging.getLogger(UNINAME)
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

    logger.debug(f'logger {UNINAME} configuration done. Log file: {LOG_PATH}, Log level: {LOG_LEVEL}, Console log level: {LOG_CONSOLE_LEVEL}, File log level: {LOG_FILE_LEVEL}.')
    # logger configuration end
    #########################

    #########################
    # # tensorboard configuration begin
    # if TENSORBOARD_LOG_PATH is not None:
    #     try:
    #         from torch.utils.tensorboard import SummaryWriter
    #     except ImportError:
    #         logger.warning('torch.utils.tensorboard is not installed. Please install torch.utils.tensorboard to use tensorboard.')
    #         SummaryWriter = None
    #     else:
    #         writer = SummaryWriter(log_dir=TENSORBOARD_LOG_PATH)
    # # tensorboard configuration end
    # #########################

    #########################
    # ignite tensorboard logger configuration begin
    if TENSORBOARD_LOG_PATH is not None:
        tb_logger = TensorboardLogger(log_dir=TENSORBOARD_LOG_PATH)
    # ignite tensorboard logger configuration end
    #########################

    # Get device
    logger.debug(f'|||||Trainer now using device: {device}|||||')
    model.to(device)

    # Prepare dataloader
    train_loader = DataLoader(train_dataset, num_workers=loader_num_workers, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=loader_num_workers, batch_size=batch_size, shuffle=True)
    logger.debug(f'Total samples: train_loader {len(train_loader)} , val_loader {len(val_loader)}')

    # Prepare metrics
    global_metrics = {
        "ssim": SSIM(1.0, output_transform = output_transform, device=device),
        "psnr": PSNR(1.0, output_transform = output_transform, device=device),
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
            'epoch':epochs,
            'learning_rate':lr,
            'device':str(device),
            'save_to':save_to,
            'loss_fn':loss_fn,
            # 'model':model,
            'train_dataset':train_dataset,
            'val_dataset':val_dataset,
            }
        },
        filter=DictFilter()
        )
    args_profile.dump_to_file(save_to=ARGS_PATH)
    logger.debug( '|-\n' + str(args_profile))

    # If torchinfo is installed, save model.torchinfo to file
    try:
        import torchinfo
    except ImportError:
        logger.warning('torchinfo is not installed. Please install torchinfo to get model summary.')
        torchinfo = None
    else:
        # Save model.torchinfo to file
        summary = torchinfo.summary(
            model = model,
            input_data=[input_transform(next(iter(train_loader)))[0]],
            depth=6,
            device=device,
        )
        assert not os.path.exists(NETWORKARCH_PATH), f'{NETWORKARCH_PATH} already exists!'
        with open(NETWORKARCH_PATH, 'w') as f:
            f.write(str(summary))
        logger.debug( '|-\n' + str(summary))

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
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
 
    # Ignite configure
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)

    train_evaluator = create_supervised_evaluator(model, metrics=global_metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=global_metrics, device=device)

    def train_step(engine, batch):
        """Train step."""
        if not model.training:
            model.train()

        batch = to_device(batch, device)
        # gpu
        with torch.no_grad(): # prepare
            batch = augmentation_transform(batch)
            x, y = input_transform(batch)

        pred = model(x)

        loss_score = loss_fn(pred, y) / accumulation_steps         
        loss_score.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return loss_score.item()

    trainer = Engine(train_step)

    def validation_step(engine, batch):
        """Validation step. Note that this function and train_step are not the difference between the training set and the test set, that is, train_step is used for gradient descent and validation_step is used to calculate metrics"""
        if model.training:
            model.eval()
        
        batch = to_device(batch, device)
        # gpu
        with torch.no_grad(): # prepare
            batch = augmentation_transform(batch)
            x, y = input_transform(batch)

            pred = model(x)
            return pred, y

    train_evaluator = Engine(validation_step) # Note that metrics and loss are not the same thing.
    val_evaluator = Engine(validation_step)

    # Attach metrics to the evaluators
    for name, metric in global_metrics.items():
        metric.attach(train_evaluator, name)

    for name, metric in global_metrics.items():
        metric.attach(val_evaluator, name)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        """Log training results."""
        train_evaluator.run(train_loader, epoch_length=eval_length_train)
        metrics_results = train_evaluator.state.metrics
        logger.info(f"Training Results - Epoch[{trainer.state.epoch}/{epochs}] {' '.join([f'{k}: {v}' for k, v in metrics_results.items()])}")
        train_history.append({'epoch': trainer.state.epoch, **metrics_results})

    @trainer.on(Events.EPOCH_COMPLETED(every=validate_every_epoch))
    def log_validation_results(trainer):
        """Log validation results."""
        val_evaluator.run(val_loader, epoch_length=eval_length_val)
        metrics_results = val_evaluator.state.metrics
        logger.info(f"Validation Results - Epoch[{trainer.state.epoch}/{epochs}] {' '.join([f'{k}: {v}' for k,v in metrics_results.items()])}")
        val_history.append({'epoch': trainer.state.epoch, **metrics_results})


    if TENSORBOARD_LOG_PATH is not None:
        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda loss: {"batch_loss": loss},
        )

        for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names="all",
                global_step_transform=global_step_from_engine(trainer),
            )


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
        score_name='-loss',
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
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_loss_checkpointer)
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_ssim_checkpointer)
        
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
    trainer.run(train_loader, max_epochs=epochs, epoch_length=epoch_length)
    logger.info(f'Training finished. Time cost: {get_time_diff(start_time, datetime.now())}')

    #########################
    # Training results and evaluations begin

    # dump train_history and val_history to csv
    pd.DataFrame(train_history).to_csv(TRAINHISTORY_PATH, sep = '\t', index=False)
    pd.DataFrame(val_history).to_csv(VALHISTORY_PATH, sep = '\t', index=False)

    # loss curve
    fig, axs = plt.subplots(1, len(global_metrics), figsize=(10*len(global_metrics), 10))
    for i, key in enumerate(global_metrics.keys()):
        if key in train_history[0].keys():
            axs[i].plot([x['epoch'] for x in train_history], [x[key] for x in train_history], label='train', color = 'darkorange')
        if key in val_history[0].keys():
            axs[i].plot([x['epoch'] for x in val_history], [x[key] for x in val_history], label='val', color = 'dodgerblue')
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
        batch = augmentation_transform(batch)
        x, y = input_transform(batch)
        pred = model(x)

        x, pred, y = map(lambda x: x.cpu().detach().numpy(), show_transform(x, pred, y))

        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        axs[0].imshow(x[0].transpose(1,2,0))
        axs[0].set_title('input')
        axs[0].axis('off')
        axs[1].imshow(pred[0].transpose(1,2,0))
        axs[1].set_title('pred')
        axs[0].axis('off')
        axs[2].imshow(y[0].transpose(1,2,0))
        axs[2].set_title('gt')
        axs[0].axis('off')

        fig.savefig(os.path.join(save_to, 'testcase.png'))
    # Training results and evaluations end
    #########################

    #########################
    # release resources begin

    # close logger
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()

    # close tensorboard
    if TENSORBOARD_LOG_PATH is not None:
        tb_logger.close()

    # release resources end
    #########################