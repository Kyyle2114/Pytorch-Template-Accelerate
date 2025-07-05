import math
from typing import Iterable, Dict, Any, Optional
from timm.utils import accuracy
from tqdm.auto import tqdm
from accelerate import Accelerator

import torch

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable, 
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator, 
    epoch: int, 
    args: Optional[Any] = None
) -> Dict[str, float]:
    """
    Train the model for one epoch using Accelerate.

    Args:
        model (torch.nn.Module): PyTorch Model
        data_loader (Iterable): PyTorch DataLoader
        criterion (torch.nn.Module): PyTorch loss function
        optimizer (torch.optim.Optimizer): PyTorch optimizer
        accelerator (Accelerator): Accelerator object for distributed training
        epoch (int): Current epoch number
        args (Optional[Any]): Parsed arguments

    Returns:
        Dict[str, float]: Dictionary containing the global average for each metric,
                          such as training loss and learning rate.
                         
    Raises:
        RuntimeError: If loss becomes infinite or NaN
        ValueError: If invalid gradient clipping value is provided
    """
    model.train()
    
    total_loss = 0.0
    
    # setup progress bar
    progress_bar = tqdm(
        range(len(data_loader)), 
        disable=not accelerator.is_main_process,
        desc=f"Epoch {epoch}",
        dynamic_ncols=True
    )

    optimizer.zero_grad()
        
    for step, (samples, targets) in enumerate(data_loader):
        with accelerator.accumulate(model):
            outputs = model(samples)
            loss = criterion(outputs, targets)
            
            # we gather the loss before backward pass
            # to avoid blocking the main process
            avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
            total_loss += avg_loss.item() / args.accum_iter
            
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                error_msg = f"Loss is {loss_value}, stopping training\n"
                accelerator.print(error_msg)
                raise RuntimeError(error_msg)

            # backward pass with accelerator
            accelerator.backward(loss)
            
            # gradient clipping
            if args and args.clip_grad is not None:
                if args.clip_grad <= 0:
                    raise ValueError(f"clip_grad must be positive, got {args.clip_grad}")
                accelerator.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            optimizer.step()
            optimizer.zero_grad()
            
        # update progress bar
        if accelerator.is_main_process:
            progress_bar.update(1)
            lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                loss=total_loss / (step + 1),
                lr=f"{lr:.6f}"
            )
        
    return {
        'loss': total_loss / len(data_loader),
        'lr': optimizer.param_groups[0]["lr"]
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable, 
    criterion: torch.nn.Module,
    accelerator: Accelerator
) -> Dict[str, float]:
    """
    Evaluate the model using Accelerate.

    Args:
        model (torch.nn.Module): PyTorch Model
        data_loader (Iterable): PyTorch DataLoader for validation/test set
        criterion (torch.nn.Module): PyTorch loss function
        accelerator (Accelerator): Accelerator object for distributed training

    Returns:
        Dict[str, float]: Dictionary containing the global average for each metric,
                          such as validation accuracy and loss.
                         
    Raises:
        RuntimeError: If evaluation encounters unexpected errors
    """
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    progress_bar = tqdm(
        range(len(data_loader)), 
        disable=not accelerator.is_main_process,
        desc="Evaluating",
        dynamic_ncols=True
    )

    try:
        for _, (images, targets) in enumerate(data_loader):
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # gather predictions and labels from all processes
            gathered_preds = accelerator.gather_for_metrics(outputs)
            gathered_labels = accelerator.gather_for_metrics(targets)
            
            all_preds.append(gathered_preds.cpu())
            all_labels.append(gathered_labels.cpu())

            # calculate the sum of losses for the current global batch and add to total_loss
            local_loss_sum = loss * images.size(0)
            total_loss += accelerator.gather(local_loss_sum).sum().item()
            
            if accelerator.is_main_process:
                progress_bar.update(1)
                
    except Exception as e:
        error_msg = f"Error during evaluation: {e}\n"
        accelerator.print(error_msg)
        raise RuntimeError(error_msg)

    # close the progress bar to ensure it is finalized before printing the metrics
    progress_bar.close()

    # concatenate all gathered tensors
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # calculate metrics on the main process
    if accelerator.is_main_process:
        # calculate average loss over all samples
        avg_loss = total_loss / len(all_labels)
        acc1, acc5 = accuracy(all_preds, all_labels, topk=(1, 5))
        
        # use standard print after closing the progress bar
        print(f'* Acc@1 {acc1.item():.3f} Acc@5 {acc5.item():.3f} loss {avg_loss:.3f} \n')
        
        return {
            'loss': avg_loss,
            'acc1': acc1.item(),
            'acc5': acc5.item()
        }
    
    # for other processes, return empty dict or sync results
    # returning results from main process is sufficient for most cases
    return {}