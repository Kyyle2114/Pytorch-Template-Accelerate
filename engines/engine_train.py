import sys
import math
from typing import Iterable, Dict, Optional
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
    clip_grad: float | None = None,
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
        clip_grad: Gradient clipping norm (None for no clipping)
        metrics_tracker: Optional reusable MetricTracker instance

    Returns:
        Dictionary containing the global average for each metric:
        - loss: Average training loss across all processes
        - lr: Current learning rate
    
    Raises:
        RuntimeError: If loss becomes infinite or NaN
        ValueError: If invalid gradient clipping value is provided
        TypeError: If arguments have wrong types
    """
    try:
        model.train()
        
        if metrics_tracker is None:
            metrics_tracker = MetricTracker()
        else:
            metrics_tracker.reset()
        
        progress_bar = tqdm(
            data_loader, 
            disable=not accelerator.is_main_process or not sys.stdout.isatty(),
            desc=f"Epoch {epoch}",
            dynamic_ncols=True
        )
            
        for samples, targets in progress_bar:
            batch_size = samples.size(0)
            
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                
                outputs = model(samples)
                loss = criterion(outputs, targets)
                
                metrics_tracker.update({'loss': loss}, batch_size=batch_size)
                
                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    error_msg = f"Loss is {loss_value}, stopping training"
                    accelerator.print(error_msg)
                    raise RuntimeError(error_msg)

                # backward pass with accelerator
                accelerator.backward(loss)
                
                # gradient clipping
                if clip_grad is not None:
                    if clip_grad <= 0:
                        raise ValueError(f"clip_grad must be positive, got {clip_grad}")
                    accelerator.clip_grad_norm_(model.parameters(), clip_grad)
                
                optimizer.step()
                scheduler.step()
                
            # progress bar shows current local averages
            if accelerator.is_main_process:
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
    accelerator: Accelerator,
    metrics_tracker: Optional[MetricTracker] = None 
) -> Dict[str, float]:
    """
    Evaluate the model using Accelerate.

    Args:
        model: PyTorch model to evaluate
        data_loader: PyTorch DataLoader for validation/test set
        criterion: PyTorch loss function
        accelerator: Accelerator object for distributed training
        metrics_tracker: Optional reusable MetricTracker instance
    
    Returns:
        Dictionary containing the global average for each metric:
        - loss: Average validation loss across all processes
        - acc1: Top-1 accuracy percentage
    
    Raises:
        RuntimeError: If evaluation encounters unexpected errors
        TypeError: If arguments have wrong types
    """
    try:
        model.eval()

        if metrics_tracker is None:
            metrics_tracker = MetricTracker()
        else:
            metrics_tracker.reset()
        
        progress_bar = tqdm(
            data_loader, 
            disable=not accelerator.is_main_process or not sys.stdout.isatty(),
            desc="Evaluating",
            dynamic_ncols=True
        )

        for images, targets in progress_bar:
            batch_size = images.size(0)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # accuracy calculation - returns percentages
            acc1, _ = accuracy(outputs, targets, topk=(1, 5))
            
            # update metrics tracker with batch results
            metrics_tracker.update({
                'loss': loss,
                'acc1': acc1,  # timm's accuracy already returns percentages
            }, batch_size=batch_size)
            
            if accelerator.is_main_process:
                # show current running averages
                current_averages = metrics_tracker.get_current_averages()
                progress_bar.set_postfix({
                    'loss': f"{current_averages.get('loss', 0.0):.4f}",
                    'acc1': f"{current_averages.get('acc1', 0.0):.4f}%"
                })

        # clean up progress bar
        if accelerator.is_main_process:
            progress_bar.close()

        # compute global averages using MetricTracker
        avg_stats = metrics_tracker.compute_epoch_averages(accelerator)

        if accelerator.is_main_process:
            acc1 = avg_stats.get("acc1", 0.0)
            loss = avg_stats.get("loss", 0.0)
            accelerator.print(f'* Acc@1: {acc1:.4f} Loss: {loss:.4f} \n')
            
        return avg_stats
        
    except Exception as e:
        # clean up resources on error
        if 'progress_bar' in locals() and accelerator.is_main_process:
            progress_bar.close()
        
        error_msg = f"Evaluation failed: {e}"
        accelerator.print(error_msg)
        raise RuntimeError(error_msg)
