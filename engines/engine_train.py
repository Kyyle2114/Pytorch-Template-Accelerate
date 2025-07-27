import math
import argparse
from typing import Iterable, Dict, Any, Optional
from timm.utils import accuracy
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler

import torch

from utils.misc import MetricTracker

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable, 
    criterion: torch.nn.Module,
    optimizer: AcceleratedOptimizer,
    scheduler: AcceleratedScheduler,
    accelerator: Accelerator, 
    epoch: int, 
    args: Optional[argparse.Namespace] = None,
    metrics_tracker: Optional[MetricTracker] = None
) -> Dict[str, float]:
    """
    Train the model for one epoch using Accelerate.

    Args:
        model: PyTorch model to train
        data_loader: PyTorch DataLoader for training data
        criterion: PyTorch loss function 
        optimizer: An `AcceleratedOptimizer` instance from `accelerator.prepare()`.
        scheduler: An `AcceleratedScheduler` instance from `accelerator.prepare()`.
        accelerator: Accelerator object for distributed training
        epoch: Current epoch number
        args: Parsed arguments containing gradient clipping and batch size info
        metrics_tracker: Optional reusable MetricTracker instance

    Returns:
        Dictionary containing the global average for each metric:
        - loss: Average training loss across all processes
        - lr: Current learning rate
    
    Raises:
        RuntimeError: If loss becomes infinite or NaN
        ValueError: If invalid gradient clipping value is provided
        TypeError: If arguments have wrong types
        
    Example:
        ```python
        # standard distributed training with reusable metrics tracker
        metrics_tracker = MetricTracker()
        for epoch in range(num_epochs):
            # The optimizer and scheduler must be prepared by `accelerate`
            prepared_optimizer, prepared_scheduler = accelerator.prepare(optimizer, scheduler)
            train_stats = train_one_epoch(
                model=model,
                data_loader=train_loader, 
                criterion=criterion,
                optimizer=prepared_optimizer,
                scheduler=prepared_scheduler,
                accelerator=accelerator,
                epoch=epoch,
                args=args,
                metrics_tracker=metrics_tracker
            )
            print(f"Training Loss: {train_stats['loss']:.4f}")
        ```
    """
    # input validation
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be a torch.nn.Module, got {type(model)}")
    if not isinstance(criterion, torch.nn.Module):
        raise TypeError(f"criterion must be a torch.nn.Module, got {type(criterion)}")
    if not isinstance(optimizer, AcceleratedOptimizer):
        raise TypeError(f"optimizer must be an `accelerate.optimizer.AcceleratedOptimizer`, got {type(optimizer)}")
    if not isinstance(scheduler, AcceleratedScheduler):
        raise TypeError(f"scheduler must be an `accelerate.scheduler.AcceleratedScheduler`, got {type(scheduler)}")
    if not isinstance(accelerator, Accelerator):
        raise TypeError(f"accelerator must be an Accelerator, got {type(accelerator)}")
    if not isinstance(epoch, int):
        raise TypeError(f"epoch must be an integer, got {type(epoch)}")
    
    try:
        model.train()
        
        # initialize or reset metrics tracker
        if metrics_tracker is None:
            metrics_tracker = MetricTracker()
        else:
            metrics_tracker.reset()
        
        progress_bar = tqdm(
            range(len(data_loader)), 
            disable=not accelerator.is_main_process,
            desc=f"Epoch {epoch}",
            dynamic_ncols=True,
            ncols=100
        )

        optimizer.zero_grad()
            
        for _, (samples, targets) in enumerate(data_loader):
            batch_size = samples.size(0)
            
            with accelerator.accumulate(model):
                outputs = model(samples)
                loss = criterion(outputs, targets)
                
                # store with batch size for accurate averaging
                metrics_tracker.update({'loss': loss}, batch_size=batch_size)
                
                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    error_msg = f"Loss is {loss_value}, stopping training"
                    accelerator.print(error_msg)
                    raise RuntimeError(error_msg)

                # backward pass with accelerator
                accelerator.backward(loss)
                
                # gradient clipping
                if args and hasattr(args, 'clip_grad') and args.clip_grad is not None:
                    if args.clip_grad <= 0:
                        raise ValueError(f"clip_grad must be positive, got {args.clip_grad}")
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            # progress bar shows current local averages
            if accelerator.is_main_process:
                progress_bar.update(1)
                lr = optimizer.param_groups[0]["lr"]
                current_averages = metrics_tracker.get_current_averages()
                progress_bar.set_postfix({
                    'loss': f"{current_averages.get('loss', 0.0):.4f}",
                    'lr': f"{lr:.6f}"
                })
        
        # clean up progress bar
        if accelerator.is_main_process:
            progress_bar.close()
        
        # compute global averages only once at epoch end
        avg_stats = metrics_tracker.compute_epoch_averages(accelerator)
        avg_stats['lr'] = optimizer.param_groups[0]["lr"]
        
        if accelerator.is_main_process:
            loss = avg_stats.get('loss', 0.0)
            lr = avg_stats['lr']
            accelerator.print(f"Training Epoch {epoch} - Loss: {loss:.4f}, Learning Rate: {lr:.6f} \n")
        
        return avg_stats
        
    except Exception as e:
        # clean up resources on error
        if 'progress_bar' in locals() and accelerator.is_main_process:
            progress_bar.close()
        raise RuntimeError(f"Training failed: {e}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable, 
    criterion: torch.nn.Module,
    accelerator: Accelerator
) -> Dict[str, float]:
    """
    Evaluate the model using Accelerate with improved accuracy calculation.

    Args:
        model: PyTorch model to evaluate
        data_loader: PyTorch DataLoader for validation/test set
        criterion: PyTorch loss function
        accelerator: Accelerator object for distributed training

    Returns:
        Dictionary containing the global average for each metric:
        - loss: Average validation loss across all processes
        - acc1: Top-1 accuracy percentage
        - acc5: Top-5 accuracy percentage
    
    Raises:
        RuntimeError: If evaluation encounters unexpected errors
        TypeError: If arguments have wrong types
        
    Example:
        ```python
        eval_stats = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            accelerator=accelerator
        )
        print(f"Validation Acc@1: {eval_stats['acc1']:.2f}%")
        print(f"Validation Loss: {eval_stats['loss']:.4f}")
        ```
    """
    # input validation
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be a torch.nn.Module, got {type(model)}")
    if not isinstance(criterion, torch.nn.Module):
        raise TypeError(f"criterion must be a torch.nn.Module, got {type(criterion)}")
    if not isinstance(accelerator, Accelerator):
        raise TypeError(f"accelerator must be an Accelerator, got {type(accelerator)}")

    try:
        model.eval()

        # initialize metrics tracker for evaluation
        metrics_tracker = MetricTracker()
        
        progress_bar = tqdm(
            range(len(data_loader)), 
            disable=not accelerator.is_main_process,
            desc="Evaluating",
            dynamic_ncols=True,
            ncols=100
        )

        for _, (images, targets) in enumerate(data_loader):
            batch_size = images.size(0)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # accuracy calculation - returns percentages
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            # update metrics tracker with batch results
            metrics_tracker.update({
                'loss': loss,
                'acc1': acc1,  # timm's accuracy already returns percentages
                'acc5': acc5
            }, batch_size=batch_size)
            
            if accelerator.is_main_process:
                progress_bar.update(1)
                # show current running averages
                current_averages = metrics_tracker.get_current_averages()
                progress_bar.set_postfix({
                    'loss': f"{current_averages.get('loss', 0.0):.4f}",
                    'acc1': f"{current_averages.get('acc1', 0.0):.2f}%"
                })

        # clean up progress bar
        if accelerator.is_main_process:
            progress_bar.close()

        # compute global averages using MetricTracker
        avg_stats = metrics_tracker.compute_epoch_averages(accelerator)

        if accelerator.is_main_process:
            acc1 = avg_stats.get("acc1", 0.0)
            acc5 = avg_stats.get("acc5", 0.0)
            loss = avg_stats.get("loss", 0.0)
            accelerator.print(f'* Acc@1: {acc1:.3f} Acc@5: {acc5:.3f} Loss: {loss:.3f} \n')
            
        return avg_stats
        
    except Exception as e:
        # clean up resources on error
        if 'progress_bar' in locals() and accelerator.is_main_process:
            progress_bar.close()
        
        error_msg = f"Evaluation failed: {e}"
        accelerator.print(error_msg)
        raise RuntimeError(error_msg)