"""The main steps in training a model

This file should contain the code that implements the decoupling of the model architecture.
I.e., there is no need to change the internal code of train() to achieve compatibility for training any model.

# Author: 5o1

# TODO:
- [ ] Implement continue from checkpoint

"""


import glob
import logging
from typing import Callable
import os
from matplotlib.pylab import f
import pandas as pd

from ignite.engine import Engine, Events
from ignite.metrics import Loss, SSIM, PSNR
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver, TensorboardLogger, EarlyStopping

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
# Data Processing Utilities end
#########################


def train(
        model : torch.nn.Module,
        train_dataset: Dataset,
        loss_fn: Callable,
        epochs: int,
        val_dataset: Dataset = None,
        lr = 1e-3,
        optimizer: torch.optim.Optimizer = None,
        batch_size: int = 8, 
        exp_name : str = 'task',
        exp_basedir : str = 'exp',
        epoch_length: int = None,
        eval_length_train: int = None,
        eval_length_val: int = None,
        accumulation_steps : int = 1,
        loader_num_workers: int = 16,
        prefetch_factor=16,
        input_transform : Callable = lambda batch : (batch, batch),
        metric_transform : Callable = lambda x : x,
        show_transform_input : Callable = lambda batch : (batch, batch, None),
        show_transform_output : Callable = lambda pred, gt, x, ps: (pred, gt, x),
        global_metrics : dict = None,
        checkpoint_every_epoch : int = 10,
        maxnum_checkpoints : int = 10,
        eval_every_epoch : int = 1,
        early_stopping_metric : str = 'ssim',
        early_stopping_patience : int = 10,
        early_stopping_min_delta : float = 1e-4,
        early_stopping_after : int = None,
        device : torch.device | str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        to_device : Callable = _to_device,
        imshow_dataset: Dataset = None,
        extra_info : dict = {},
        eps : float = 1e-13
        # continue_from : str = None, # Todo
        ):
    

    #########################
    # parameters check begin
    if val_dataset is None:
        val_dataset = train_dataset

    if global_metrics is None:
        def output_transform(output):
            """Transform output to metric input."""
            return metric_transform(output[0]), metric_transform(output[1])
        
        global_metrics = {
            "ssim": SSIM(1.0, output_transform = output_transform, device=device),
            "psnr": PSNR(1.0, output_transform = output_transform, device=device),
            "loss": Loss(loss_fn, device=device)
        }

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, eps = eps)
    else:
        lr = "setting with optimizer"

    for key in global_metrics.keys():
        if key.lower() == 'loss':
            loss_name = key

    if early_stopping_after is None:
        early_stopping_after = epochs // 5

    # parameters check end
    #########################


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
    TRAINHISTORY_PATH = os.path.join(save_to, 'train_history.csv')
    VALHISTORY_PATH = os.path.join(save_to, 'val_history.csv')

    #########################
    
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

    try:
        import tensorboard
    except ImportError:
        TENSORBOARD_LOG_PATH = None
        logger.warning('tensorboard is not installed. Please install tensorboard to use tensorboard.')
    else:
        TENSORBOARD_LOG_PATH = save_to

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


    #########################
    # Prepare torch components begin
    # Get device
    logger.debug(f'|||||Trainer now using device: {device}|||||')
    model.to(device)

    # Prepare dataloader
    train_loader = DataLoader(train_dataset, num_workers=loader_num_workers, batch_size=batch_size, shuffle=True, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, num_workers=loader_num_workers, batch_size=batch_size, shuffle=True, prefetch_factor=prefetch_factor)
    logger.debug(f'Total batches: train_loader {len(train_loader)} , val_loader {len(val_loader)}, and epoch length is {epoch_length}.')

    # Prepare imshow loader
    if imshow_dataset is not None:
        imshow_loader = DataLoader(imshow_dataset, batch_size=1, shuffle=False)

    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    # Prepare torch components end
    #########################

    #########################
    # Prepare training steps begin

    def train_step(engine, batch):
        """Train step."""
        if not model.training:
            model.train()

        batch = to_device(batch, device)
        # gpu
        with torch.no_grad(): # prepare
            x, y = input_transform(batch)

        pred = model(x)

        loss_score = loss_fn(pred, y) / accumulation_steps         
        loss_score.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return loss_score.item()


    def eval_step(engine, batch):
        """Evaluation step. Note that this function and train_step are not the difference between the training set and the test set, that is, train_step is used for gradient descent and validation_step is used to calculate metrics"""
        if model.training:
            model.eval()
        
        batch = to_device(batch, device)
        # gpu
        with torch.no_grad(): # prepare
            x, y = input_transform(batch)
            pred = model(x)
        return pred, y
        

    def predict_step(engine, batch):
        if model.training:
            model.eval()

        batch = to_device(batch, device)
        with torch.no_grad():
            x, gt, ps = show_transform_input(batch)
            pred = model(x)
    
        return pred, gt, x, ps
    

    def imshow_process(loader, filename, all=False, suptitle = None):
        print(f'imshowing {filename} device: {device}')
        for i, batch in enumerate(loader):
            if i == 5:
                break
            pred, gt, x, ps = predict_step(None, batch)

            with torch.no_grad():
                pred, gt, x = map(lambda item: item.cpu().detach().numpy(), show_transform_output(pred, gt, x, ps))

            cmaps = list(map(lambda image: 'gray' if image.shape[1] == 1 else None, [pred, gt, x]))

            fig, axs = plt.subplots(1, 4, figsize=(40, 10), constrained_layout=True)
            if suptitle is not None:
                if all:
                    plt.suptitle(suptitle + f"_{i}", fontsize=30)
                else:
                    plt.suptitle(suptitle, fontsize=30)
            axs[0].imshow(x[0].transpose(1,2,0), cmap=cmaps[0])
            axs[0].set_title('input', fontsize=20)
            axs[0].axis('off')
            axs[1].imshow(pred[0].transpose(1,2,0), cmap=cmaps[1])
            axs[1].set_title('pred', fontsize=20)
            axs[1].axis('off')
            axs[2].imshow(gt[0].transpose(1,2,0), cmap=cmaps[2])
            axs[2].set_title('gt', fontsize=20)
            axs[2].axis('off')
            im = axs[3].imshow(((pred[0]-gt[0])/(gt.max()-gt.min())).transpose(1,2,0), cmap = "viridis", vmin = -0.5, vmax = 0.5)
            axs[3].set_title('(pred - gt) / (gt.max - gt.min)', fontsize=20)
            axs[3].axis('off')
            # 添加 colorbar
            cbar = plt.colorbar(im, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label('Difference (value)', rotation=270, labelpad=20, fontsize=20)  # 可选：设置标签

            fig.subplots_adjust(wspace=0.3, hspace=0.4)

            if all:
                fig.savefig(os.path.join(save_to, f'{filename}_{i}.png'))
            else:
                fig.savefig(os.path.join(save_to, f'{filename}.png'))
                break
        plt.show()
    
    # Prepare training steps end
    #########################

    #########################
    # Saving Experiment Configurations begin
    # Save args.yaml to file
    config = {
        TASK_NAME:{
        'exp_name':exp_name,
        'batch_size':batch_size,
        'epoch':epochs,
        'learning_rate':lr,
        'device':str(device),
        'save_to':save_to,
        'loss_fn':loss_fn,
        # 'model':model,
        # 'train_dataset':train_dataset,
        # 'val_dataset':val_dataset,
        }
    }
    config.update({
        "extra_info": extra_info
    })
    config_profile = Profile(
        profile=config,
        filter=DictFilter()
        )
    config_profile.dump_to_file(save_to=ARGS_PATH)
    logger.debug( '|-\n' + str(config_profile))

    # If torchinfo is installed, save model.torchinfo to file
    try:
        import torchinfo
    except ImportError:
        logger.warning('torchinfo is not installed. Please install torchinfo to get model summary.')
        torchinfo = None
    else:
        # Save model.torchinfo to file
        summary_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_num_workers)
        summary = torchinfo.summary(
            model = model,
            input_data=[input_transform(to_device(next(iter(summary_loader)), device=device))[0]],
            depth=6,
            device=device,
        )
        assert not os.path.exists(NETWORKARCH_PATH), f'{NETWORKARCH_PATH} already exists!'
        with open(NETWORKARCH_PATH, 'w', encoding='utf-8') as f:
            f.write(str(summary))
        logger.debug( '|-\n' + str(summary))


    # imshow before training
    if imshow_dataset is not None:
        logger.info('imshow before training')
        imshow_process(imshow_loader, 'showset_before', all=True, suptitle='showset before training')

    # Saving Experiment Configurations end
    #########################


    #########################
    # Main training steps begin 
    train_begin_time = datetime.now()
    logger.info(f'Training started. Time: {train_begin_time}')

    train_history = []
    val_history = []

    # Ignite configure
    trainer = Engine(train_step)

    train_evaluator = Engine(eval_step) # Note that metrics and loss are not the same thing.
    val_evaluator = Engine(eval_step)

    # Attach metrics to the evaluators
    for name, metric in global_metrics.items():
        metric.attach(train_evaluator, name)

    for name, metric in global_metrics.items():
        metric.attach(val_evaluator, name)


    @trainer.on(Events.EPOCH_COMPLETED(every=eval_every_epoch))
    def log_training_results(trainer):
        """Log training results."""
        logger.debug(f"Epoch[{trainer.state.epoch}/{epochs}] run train evaluator.")
        train_evaluator.run(train_loader, epoch_length=eval_length_train)
        metrics_results = train_evaluator.state.metrics
        logger.info(f"Training Results - Epoch[{trainer.state.epoch}/{epochs}] {' '.join([f'{k}: {v}' for k, v in metrics_results.items()])}")
        train_history.append({'epoch': trainer.state.epoch, **metrics_results})

    @trainer.on(Events.EPOCH_COMPLETED(every=eval_every_epoch))
    def log_validation_results(trainer):
        """Log validation results."""
        logger.debug(f"Epoch[{trainer.state.epoch}/{epochs}] run val evaluator.")
        val_evaluator.run(val_loader, epoch_length=eval_length_val)
        metrics_results = val_evaluator.state.metrics
        logger.info(f"Validation Results - Epoch[{trainer.state.epoch}/{epochs}] {' '.join([f'{k}: {v}' for k,v in metrics_results.items()])}")
        val_history.append({'epoch': trainer.state.epoch, **metrics_results})


    if TENSORBOARD_LOG_PATH is not None:
        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda loss: {"iteration_loss": loss},
        )

        for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names="all",
                global_step_transform=global_step_from_engine(trainer),
            )
    
    # tb show image
    if TENSORBOARD_LOG_PATH is not None and imshow_dataset is not None:
        tb_imshow_evaluator = Engine(predict_step)

        @trainer.on(Events.EPOCH_COMPLETED(every=eval_every_epoch))
        def tb_evaluator_run(trainer):
            logger.debug(f"Epoch[{trainer.state.epoch}/{epochs}] run tb_imshow_evaluator.")
            tb_imshow_evaluator.run(imshow_loader)

        @tb_imshow_evaluator.on(Events.ITERATION_COMPLETED)
        def tb_imshow_log(engine):
            logger.debug(f"Epoch[{trainer.state.epoch}/{epochs}] tb_iter {engine.state.iteration}, run imshow in tensorboard.")
            pred, _, _ = show_transform_output(*engine.state.output) # B C H W
            global_step = trainer.state.epoch
            tb_logger.writer.add_image(f'pred_{engine.state.iteration}', pred[0], global_step, dataformats='CHW')

    # checkpoint    
    _to_save_dict = {'model': model, 'optimizer': optimizer}

    every_n_checkpointer = Checkpoint(
        to_save=_to_save_dict,
        save_handler=DiskSaver(save_to,create_dir=True, require_empty=False),
        filename_prefix='checkpoint',
        n_saved=maxnum_checkpoints,
        global_step_transform=global_step_from_engine(trainer),
    )

    metrics_checkpointers = []
    for metric_name in global_metrics.keys():
        if metric_name.lower() == 'loss':
            metrics_checkpointers.append(Checkpoint(
                to_save=_to_save_dict,
                save_handler=DiskSaver(save_to,create_dir=True, require_empty=False),
                filename_prefix=f'best',
                n_saved=1,
                score_name=metric_name,
                score_function=lambda engine: -engine.state.metrics[loss_name],
                global_step_transform=global_step_from_engine(trainer),
            ))
        else:
            metrics_checkpointers.append(Checkpoint(
                to_save=_to_save_dict,
                save_handler=DiskSaver(save_to,create_dir=True, require_empty=False),
                filename_prefix=f'best',
                n_saved=1,
                score_name=metric_name,
                global_step_transform=global_step_from_engine(trainer),
            ))
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_every_epoch), every_n_checkpointer)
    for metrics_checkpoint in metrics_checkpointers:
        val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, metrics_checkpoint)

    # early stopping
    if early_stopping_metric is not None and early_stopping_patience is not None and early_stopping_min_delta is not None and early_stopping_metric in global_metrics.keys():
        if early_stopping_metric.lower() == 'loss':
            early_stopping_handler = EarlyStopping(
                patience=early_stopping_patience,
                score_function=(lambda engine: -engine.state.metrics[early_stopping_metric]),
                trainer=trainer,
                min_delta=early_stopping_min_delta,
            )
        else:
            early_stopping_handler = EarlyStopping(
                patience=early_stopping_patience,
                score_function=lambda engine: engine.state.metrics[early_stopping_metric],
                trainer=trainer,
                min_delta=early_stopping_min_delta,
            )
        early_stopping_handler.logger = logger
        val_evaluator.add_event_handler(Events.EPOCH_COMPLETED(event_filter=lambda *args: True if trainer.state.epoch >= early_stopping_after else False), early_stopping_handler)
        logger.info(f'Early stopping is enabled. Early stopping metric: {early_stopping_metric}, patience: {early_stopping_patience}, min_delta: {early_stopping_min_delta}, after: {early_stopping_after}')
    else:
        logger.warning('Early stopping is not enabled.')
    # Main training steps end
    #########################

    # Training
    trainer.run(train_loader, max_epochs=epochs, epoch_length=epoch_length)
    logger.info(f'Training finished. Time cost: {get_time_diff(train_begin_time, datetime.now())}')

    #########################
    # Training results and evaluations begin

    # load best model
    def get_best_loss_model():
        # best_checkpoint_6_loss=-0.0009.pt
        pt_list = glob.glob(os.path.join(save_to, f'best*{loss_name}*.pt'))
        if len(pt_list) == 0:
            logger.warning(f'No best model found in {save_to}.')
            return None
        best_model_name = pt_list[0]
        logger.info(f'Best model found: {best_model_name}')
        return torch.load(best_model_name, weights_only=False)['model']
    model.load_state_dict(get_best_loss_model())

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
    train_loader_shown = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader_shown = DataLoader(val_dataset, batch_size=1, shuffle=False)


    imshow_process(train_loader_shown, 'testcase_trainset', all=False, suptitle='trainset')
    imshow_process(val_loader_shown, 'testcase_valset', all=False, suptitle='valset')
    if imshow_dataset is not None:
        imshow_process(imshow_loader, 'testcase_tb', all=True, suptitle='showset')
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