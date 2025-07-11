import os
import argparse
import torchinfo
import time
import datetime
import json
import numpy as np
from pathlib import Path
from accelerate import Accelerator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.utils.data import DataLoader

from utils import misc, datasets, lr_sched
from engines import engine_train
from models import cnn

def get_args_parser() -> argparse.ArgumentParser:
    """
    Create and return argument parser for training configuration.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(add_help=False)
    
    # --- Initial config ---
    parser.add_argument('--seed', type=int, default=21, 
                        help='random seed for reproducibility')
    
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save checkpoints and logs')
    
    # --- Training config ---
    parser.add_argument('--dataset_path', type=str, default='./dataset', 
                        help='dataset root path')
    
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='batch size per GPU')
    
    parser.add_argument('--epoch', type=int, default=10, 
                        help='total number of training epochs')
    
    parser.add_argument('--patience', type=int, default=50, 
                        help='patience for early stopping')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of data loading workers')
    
    # --- Optimizer config ---
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='base learning rate')
    
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='weight decay for optimizer')
    
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='gradient accumulation steps to increase effective batch size')
    
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='number of warmup epochs for learning rate scheduler')
    
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gradient clipping norm (None for no clipping)')
    
    # --- WandB config ---
    parser.add_argument('--project_name', type=str, default='Model-Training', 
                        help='WandB project name')
    
    parser.add_argument('--run_name', type=str, default='Model-Training', 
                        help='WandB run name')
    
    return parser


def main(args: argparse.Namespace) -> None:
    """
    Main function for model training with Accelerate.

    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Raises:
        RuntimeError: If training encounters unrecoverable errors
        ValueError: If invalid arguments are provided
    """
    # --- Input validation ---
    if args.epoch <= 0:
        raise ValueError(f"Number of epochs must be positive, got {args.epoch}\n")
    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {args.batch_size}\n")
    if args.lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {args.lr}\n")
    
    # --- Accelerate & WandB setting ---
    misc.seed_everything(args.seed)
    
    # initialize accelerator
    try:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.accum_iter,
            mixed_precision='fp16',
            log_with='wandb',
            project_dir=args.output_dir
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Accelerator: {e}\n")
    
    # wandb initialization through accelerator
    if accelerator.is_main_process:
        try:
            accelerator.init_trackers(
                project_name=args.project_name,
                config=vars(args),
                init_kwargs={"wandb": {"name": args.run_name}}
            )
            
        except Exception as e:
            print(f"Warning: Failed to initialize WandB tracking: {e}")
            print("Continuing without WandB logging...")
    
    if accelerator.is_main_process:
        print('\njob dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print('args: ', args, '\n')
        
        # ensure output directory exists
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # save args as JSON file with timestamp
        args_dict = vars(args)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args_file_path = output_path / f'args_{timestamp}.json'
        
        try:
            with open(args_file_path, mode="w", encoding="utf-8") as f:
                json.dump(args_dict, f, indent=4)
        
        except IOError as e:
            print(f"Warning: Failed to save arguments to file: {e}\n")
    
    # --- Dataset & Dataloader ---
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # CIFAR10 for testing
    try:
        train_set = datasets.make_cifar10_dataset(
            dataset_path=args.dataset_path,
            train=True,
            transform=transform
        )

        val_set = datasets.make_cifar10_dataset(
            dataset_path=args.dataset_path,
            train=False,
            transform=transform
        )
    
    except Exception as e:
        raise RuntimeError(f"Failed to create datasets: {e}\n")
    
    try:
        train_loader = DataLoader(
            train_set, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=misc.seed_worker,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_set, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None
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
        print()
        print('=== MODEL INFO ===')
        torchinfo.summary(model)
        print()    

    # --- Training config (loss, optimizer, scheduler) ---
    criterion = nn.CrossEntropyLoss()

    eff_batch_size = args.batch_size * args.accum_iter * accelerator.num_processes
    args.abs_lr = args.lr * eff_batch_size / 256
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model, args.weight_decay)
    
    optimizer = torch.optim.AdamW(
        param_groups, 
        lr=1e-7  # small lr for warm-up
    )
    
    if accelerator.is_main_process:
        print('Optimizer:')
        print(optimizer, '\n')
    
    try:
        scheduler = lr_sched.CosineAnnealingWarmUpRestarts(
            optimizer, 
            T_0=args.epoch, 
            T_mult=1, 
            eta_max=args.abs_lr, 
            T_up=args.warmup_epochs, 
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
                'optimizer': type(optimizer).__name__,
                'scheduler': type(scheduler).__name__,
                'batch_size_accumulated': args.batch_size * args.accum_iter,
                'effective_batch_size': eff_batch_size,
                'num_parameters': n_parameters
            }, step=0)
        
        except Exception as e:
            print(f"Warning: Failed to log to WandB: {e}\n")
    
    # --- Model training ---
    start_time = time.time()
    max_accuracy = 0.0  # for evaluation
    max_loss = np.inf   # for evaluation 
    
    # early stopping: distributed-aware version
    es = misc.DistributedEarlyStopping(patience=args.patience, delta=0, mode='min', verbose=True)
    
    # initialize reusable metrics tracker for training
    metrics_tracker = misc.MetricTracker()
    
    # training loop
    try:
        for epoch in range(args.epoch):
            # model train with reusable metrics tracker
            train_stats = engine_train.train_one_epoch(
                model=model, 
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                accelerator=accelerator,
                epoch=epoch,
                args=args,
                metrics_tracker=metrics_tracker
            )
            
            # save model checkpoint periodically
            if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epoch):
                try:
                    if accelerator.is_main_process:
                        save_dir = Path(args.output_dir) / f"checkpoint-{epoch}"
                        accelerator.save_state(save_dir)
                        accelerator.print(f"Periodic checkpoint saved to {save_dir}")
                    
                except Exception as e:
                    accelerator.print(f"Warning: Failed to save periodic checkpoint: {e}")
                
            scheduler.step()
                
            # model evaluation (validation set)
            eval_stats = engine_train.evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                accelerator=accelerator
            )
            
            # validation on main process
            if accelerator.is_main_process:
                print(f"[INFO] Accuracy of the network on the {len(val_set)} test images: {eval_stats['acc1']:.1f}%")
                max_accuracy = max(max_accuracy, eval_stats["acc1"])
                val_loss = eval_stats['loss']
                print(f'[INFO] Current max validation accuracy: {max_accuracy:.2f}%')
                
                # save best model based on validation loss
                if val_loss < max_loss:
                    print(f'[INFO] Validation loss improved from {max_loss:.5f} to {val_loss:.5f}. Saving best model.')
                    max_loss = val_loss
                    
                    try:
                        save_dir = Path(args.output_dir) / "best_model"
                        accelerator.save_state(save_dir)
                        accelerator.print(f"Best model saved to {save_dir}")

                    except Exception as e:
                        accelerator.print(f"Warning: Failed to save best model: {e}")

                # stats logging to file
                if args.output_dir:
                    log_stats = {
                        **{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in eval_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters,
                        'max_accuracy': max_accuracy
                    }
                    
                    try:
                        log_file_path = Path(args.output_dir) / "log.txt"
                        with open(log_file_path, mode="a", encoding="utf-8") as f:
                            f.write(json.dumps(log_stats) + "\n")
                    
                    except IOError as e:
                        print(f"Warning: Failed to write log file: {e}\n")
                
                # wandb logging
                try:
                    accelerator.log(
                        {
                            'Training loss': train_stats['loss'],
                            'Training learning rate': train_stats['lr'],
                            'Evaluation loss': eval_stats['loss'],
                            'Evaluation top-1 accuracy': eval_stats['acc1'],
                            'Max accuracy': max_accuracy
                        }, step=epoch+1
                    )
                
                except Exception as e:
                    print(f"Warning: Failed to log to WandB: {e}\n")
                        
                # check early stopping with distributed synchronization
                should_stop = es(val_loss, accelerator)
                if should_stop:
                    print(f'[INFO] Early stopping triggered at epoch {epoch+1}\n')
                    # sync is already handled by DistributedEarlyStopping
                    break
                
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")
    
    finally:
        pass  # Python's automatic garbage collection is sufficient
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training completed in {total_time_str}')
    print(f'Best validation accuracy: {max_accuracy:.2f}%')
    
    # end wandb tracking
    if accelerator.is_main_process:
        try:
            accelerator.end_training()
        
        except Exception as e:
            print(f"Warning: Failed to properly end WandB tracking: {e}")
    
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('Model-Training', parents=[get_args_parser()])
    args = parser.parse_args() 
    
    try:
        main(args)
        print('\n=== Training Complete ===\n')
    
    except Exception as e:
        print(f'\n=== Training Failed ===')
        print(f'Error: {e}\n')
        raise