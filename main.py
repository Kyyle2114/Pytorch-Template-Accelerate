import os
import math
import argparse
import torchinfo
import time
import datetime
import json
import yaml
from pathlib import Path
from accelerate import Accelerator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config.schemas import Config
from utils import misc, datasets, lr_sched
from engines import engine_train
from models import cnn


def get_args_parser() -> argparse.ArgumentParser:
    """
    Create and return argument parser for training configuration.
    
    All configuration is handled through the YAML config file.
    Override config values with --set KEY=VALUE
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(add_help=False)
    
    # --- Config file path ---
    parser.add_argument(
        '-c', '--config', 
        type=str, 
        default='config/default.yaml',
        help='path to YAML configuration file'
    )
    
    # --- Show help for config ---
    parser.add_argument(
        '--help_config', 
        action='store_true',
        help='show detailed help for configuration parameters'
    )
    
    # --- Override config values ---
    parser.add_argument(
        '--set', 
        action='append', 
        nargs='+', 
        metavar='KEY=VALUE',
        help='override any config value (e.g., --set general.seed=42 --set training.lr=0.001)'
    )
    
    return parser


def main(args: argparse.Namespace) -> None:
    """
    Main function for model training.

    Args:
        args: Parsed command line arguments containing:
            - config: Path to YAML configuration file
            - set: Optional config overrides in KEY=VALUE format
            - help_config: Show config help and exit
    """
    # --- Load configuration ---
    try:
        config: Config = misc.load_config(args.config, args)
        
        # print config help and exit
        if args.help_config:
            misc.print_config_help(config)
            exit(0)
            
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")
    
    # --- Seed & Output setup ---
    misc.seed_everything(config.general.seed)
    
    output_path = Path(config.general.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # --- Accelerate & WandB setting ---
    try:
        accelerator = Accelerator(
            gradient_accumulation_steps=config.training.accum_iter,
            mixed_precision='fp16',
            log_with='wandb',
            project_dir=str(output_path)
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Accelerator: {e}\n")
    
    # wandb initialization through accelerator
    if accelerator.is_main_process:
        try:
            accelerator.init_trackers(
                project_name=config.wandb.project_name,
                config=config.model_dump(),
                init_kwargs={"wandb": {"name": config.wandb.run_name}}
            )
            
        except Exception as e:
            accelerator.print(f"Warning: Failed to initialize WandB tracking: {e}")
            accelerator.print("Continuing without WandB logging...")
    
    if accelerator.is_main_process:
        accelerator.print('\njob dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        accelerator.print(f'config: {args.config}\n')
        
        # save config as YAML file
        config_file = output_path / 'config.yaml'
        try:
            with open(config_file, mode="w", encoding="utf-8") as f:
                yaml.dump(config.model_dump(), f, indent=4, sort_keys=False)
        
        except IOError as e:
            accelerator.print(f"Warning: Failed to save config to file: {e}\n")
    
    # --- Dataset & Dataloader ---
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # CIFAR10 for testing
    try:
        train_set = datasets.make_cifar10_dataset(
            dataset_path=config.data.dataset_path,
            train=True,
            transform=transform
        )

        val_set = datasets.make_cifar10_dataset(
            dataset_path=config.data.dataset_path,
            train=False,
            transform=transform
        )
    
    except Exception as e:
        raise RuntimeError(f"Failed to create datasets: {e}\n")
    
    try:
        num_workers = config.data.num_workers
        train_loader = DataLoader(
            train_set, 
            batch_size=config.data.batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=misc.seed_worker,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_set, 
            batch_size=config.data.batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
    
    except Exception as e:
        raise RuntimeError(f"Failed to create data loaders: {e}\n")
    
    # --- Model config ---
    try:
        model = cnn.SimpleCNNforCIFAR10()
        n_parameters = model.num_parameters
    
    except Exception as e:
        raise RuntimeError(f"Failed to create model: {e}\n")
    
    # print model info 
    if accelerator.is_main_process:
        accelerator.print()
        accelerator.print('=== MODEL INFO ===')
        torchinfo.summary(model)
        accelerator.print()    

    # --- Training config (loss, optimizer, scheduler) ---
    criterion = nn.CrossEntropyLoss()

    eff_batch_size = config.data.batch_size * config.training.accum_iter * accelerator.num_processes
    abs_lr = config.training.lr * eff_batch_size / 256
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model, config.training.weight_decay)
    
    optimizer = torch.optim.AdamW(
        param_groups, 
        lr=0.0  # small lr for warm-up
    )
    
    # calculate total number of steps for the scheduler
    num_update_steps_per_epoch = math.ceil(len(train_loader) / config.training.accum_iter)
    num_training_steps = config.training.epoch * num_update_steps_per_epoch
    num_warmup_steps = config.training.warmup_epochs * num_update_steps_per_epoch
    
    try:
        # use per-step lr scheduler
        scheduler = lr_sched.CosineAnnealingWarmUpRestarts(
            optimizer, 
            T_0=num_training_steps, 
            T_mult=1, 
            eta_max=abs_lr, 
            T_up=num_warmup_steps, 
            gamma=1.0
        )
    
    except Exception as e:
        raise RuntimeError(f"Failed to create learning rate scheduler: {e}\n")

    # --- Prepare everything with accelerator ---
    try:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
    
    except Exception as e:
        raise RuntimeError(f"Failed to prepare components with Accelerator: {e}\n")

    # --- WandB logging ---
    if accelerator.is_main_process:
        try:
            # log additional configurations to wandb through accelerator
            accelerator.log({
                'config/batch_size_per_gpu': config.data.batch_size,
                'config/gradient_accumulation_steps': config.training.accum_iter,
                'config/num_processes': accelerator.num_processes,
                'config/effective_batch_size': eff_batch_size,
                'config/num_parameters': n_parameters
            }, step=0)
        
        except Exception as e:
            accelerator.print(f"Warning: Failed to log to WandB: {e}\n")
    
    # --- Model training ---
    start_time = time.time()
    max_accuracy = 0.0
    min_loss = float('inf')
    
    # best model metrics
    best_epoch = 0
    best_acc1 = 0.0
    
    # early stopping: lower is better ('min' mode)
    es = misc.DistributedEarlyStopping(
        patience=config.training.patience, 
        delta=0.0, 
        mode='min', 
        verbose=True
    )
    
    metrics_tracker = misc.MetricTracker()
    
    # training loop
    try:
        for epoch in range(config.training.epoch):
            train_stats = engine_train.train_one_epoch(
                model=model, 
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=accelerator,
                epoch=epoch,
                clip_grad=config.training.clip_grad,
                metrics_tracker=metrics_tracker
            )
                
            # model evaluation (validation set)
            eval_stats = engine_train.evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                accelerator=accelerator,
                metrics_tracker=metrics_tracker
            )
            
            # update best validation metrics on main process
            if accelerator.is_main_process:
                accelerator.print(f"[INFO] Accuracy of the network on the {len(val_set)} test images: {eval_stats['acc1']:.4f}%")
                max_accuracy = max(max_accuracy, eval_stats["acc1"])
                val_loss = eval_stats['loss']
                accelerator.print(f'[INFO] Current max validation accuracy: {max_accuracy:.4f}%')
                
                # save best model based on validation loss
                if val_loss < min_loss:
                    accelerator.print(f'[INFO] Validation loss improved from {min_loss:.5f} to {val_loss:.5f}. Saving best model.')
                    min_loss = val_loss
                    best_epoch = epoch
                    best_acc1 = eval_stats['acc1']
                    
                    try:
                        save_dir = output_path / "best_model"
                        accelerator.save_state(save_dir)
                        accelerator.print(f"[INFO] Best model saved to {save_dir}")

                    except Exception as e:
                        accelerator.print(f"Warning: Failed to save best model: {e}")

                # stats logging to file
                if config.general.output_dir:
                    log_stats = {
                        **{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in eval_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters,
                        'max_accuracy': max_accuracy
                    }
                    
                    try:
                        log_file_path = output_path / "log.txt"
                        with open(log_file_path, mode="a", encoding="utf-8") as f:
                            f.write(json.dumps(log_stats) + "\n")
                    
                    except IOError as e:
                        accelerator.print(f"Warning: Failed to write log file: {e}\n")
                
                # wandb logging
                try:
                    accelerator.log(
                        {
                            'train/loss': train_stats['loss'],
                            'train/learning_rate': train_stats['lr'],
                            'eval/loss': eval_stats['loss'],
                            'eval/acc1': eval_stats['acc1'],
                            'eval/max_accuracy': max_accuracy,
                            'epoch': epoch
                        }, step=epoch
                    )
                
                except Exception as e:
                    accelerator.print(f"Warning: Failed to log to WandB: {e}\n")
                        
            # check early stopping
            should_stop = es(eval_stats['loss'], accelerator)

            if should_stop:
                accelerator.print(f"[INFO] Early stopping triggered at epoch {epoch}\n")
                break
                
    except KeyboardInterrupt:
        accelerator.print("\n[INFO] Training interrupted by user")
        
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")
    
    finally:
        # save last model
        if accelerator.is_main_process:
            try:
                save_dir = output_path / "last_model"
                accelerator.save_state(save_dir)
                accelerator.print(f"[INFO] Last model saved to {save_dir}")
            except Exception as e:
                accelerator.print(f"Warning: Failed to save last model: {e}")
        
        # print final metrics
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        if accelerator.is_main_process:
            accelerator.print(f'\n[INFO] Training completed in {total_time_str}')
            accelerator.print(f'[INFO] Best validation epoch: {best_epoch}')
            accelerator.print(f'[INFO] Best validation loss: {min_loss:.5f}')
            accelerator.print(f'[INFO] Best validation accuracy: {best_acc1:.4f}%')
            accelerator.print(f'[INFO] Max validation accuracy: {max_accuracy:.4f}%\n')
        
        # log final metrics to wandb
        try:
            accelerator.log({
                'final/best_loss': min_loss,
                'final/best_epoch': best_epoch,
                'final/best_acc1': best_acc1,
                'final/max_accuracy': max_accuracy,
                'final/training_time': total_time_str
            })
        except Exception as e:
            accelerator.print(f"Warning: Failed to log final metrics: {e}")

        # end training
        if accelerator.is_main_process:
            try:
                accelerator.end_training()
            except Exception as e:
                accelerator.print(f"Warning: Failed to properly end WandB tracking: {e}")
    

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('Model-Training', parents=[get_args_parser()])
    args = parser.parse_args() 
    
    # check if this is the main process (LOCAL_RANK=0 or not set in single GPU)
    is_main_process = int(os.environ.get('LOCAL_RANK', 0)) == 0
    
    try:
        main(args)
        if is_main_process:
            print('\n=== Training Complete ===\n')
    
    except Exception as e:
        if is_main_process:
            print(f'\n=== Training Failed ===\n')
            print(f'Error: {e}\n')
        exit(1)