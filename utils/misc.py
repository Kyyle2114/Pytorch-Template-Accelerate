import os 
import random
import numpy as np
from typing import Any, Dict, List

import torch

def seed_everything(seed: int = 21) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value. Defaults to 21.
    """
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.use_deterministic_algorithms(True)


def seed_worker(_worker_id: int) -> None:
    """
    Worker initialization function for DataLoader to ensure reproducible results.
    
    Args:
        _worker_id (int): Worker ID (unused but required by DataLoader)
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_weight_decay(model: torch.nn.Module, weight_decay: float = 1e-5, skip_list: tuple = ()) -> List[Dict[str, Any]]:
    """
    Add weight decay to parameters that don't have it.
    Set weight decay to 0 for bias and 1D weight tensors.
    Following timm's implementation.
    
    Args:
        model (torch.nn.Module): Model to add weight decay to
        weight_decay (float): Weight decay value
        skip_list (tuple): List of parameter names to skip weight decay
    
    Returns:
        List[Dict[str, Any]]: List of parameter groups with weight decay
    """
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
            
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
            
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


class MetricTracker:
    """
    Distributed metrics tracker for Accelerate.
    
    Uses running averages instead of storing all values to minimize memory usage.
    Computes accurate global averages in distributed settings.
    """
    def __init__(self) -> None:
        """Initialize metric tracker."""
        self.running_metrics: Dict[str, float] = {}
        self.running_counts: Dict[str, int] = {}
        self.total_samples: int = 0
    
    def update(self, metrics_dict: Dict[str, torch.Tensor], batch_size: int = 1) -> None:
        """
        Update metrics using running averages for memory efficiency.
        
        Args:
            metrics_dict: Dictionary of metrics (key: metric name, value: tensor)
            batch_size: Number of samples in the current batch
            
        Raises:
            TypeError: If metrics_dict is not a dictionary
            ValueError: If batch_size <= 0
        """
        if not isinstance(metrics_dict, dict):
            raise TypeError(f"metrics_dict must be a dictionary, got {type(metrics_dict)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
            
        self.total_samples += batch_size
        
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                current_value = value.item()
            else:
                current_value = float(value)
            
            if key not in self.running_metrics:
                self.running_metrics[key] = current_value
                self.running_counts[key] = batch_size
            else:
                # weighted running average
                total_count = self.running_counts[key] + batch_size
                self.running_metrics[key] = (
                    self.running_metrics[key] * self.running_counts[key] + 
                    current_value * batch_size
                ) / total_count
                self.running_counts[key] = total_count
    
    def compute_epoch_averages(self, accelerator) -> Dict[str, float]:
        """
        Compute accurate global averages across all processes.
        
        Args:
            accelerator: Accelerator instance for distributed operations
            
        Returns:
            Dictionary of globally averaged metrics
        """
        if self.total_samples == 0 or not self.running_metrics:
            return {}
            
        averages = {}
        
        if accelerator.use_distributed:
            # optimize: gather all metrics at once instead of separate calls
            metric_keys = list(self.running_metrics.keys())
            local_data = []
            
            for key in metric_keys:
                local_avg = self.running_metrics[key]
                local_count = self.running_counts[key]
                local_sum = local_avg * local_count
                local_data.extend([local_sum, local_count])
            
            if not local_data:
                return {}
                
            # single gather call for all metrics
            try:
                gathered_data = accelerator.gather_for_metrics(
                    torch.tensor(local_data, device=accelerator.device, dtype=torch.float32)
                )
                
                # reshape and compute averages
                num_processes = gathered_data.shape[0]
                num_metrics = len(metric_keys)
                gathered_data = gathered_data.view(num_processes, num_metrics * 2)
                
                for i, key in enumerate(metric_keys):
                    sum_idx = i * 2
                    count_idx = i * 2 + 1
                    
                    total_sum = gathered_data[:, sum_idx].sum()
                    total_count = gathered_data[:, count_idx].sum()
                    global_avg = total_sum / total_count if total_count > 0 else 0.0
                    averages[key] = global_avg.item()
                    
            except Exception as e:
                # fallback to local averages if gathering fails
                if accelerator.is_main_process:
                    print(f"Warning: Failed to gather metrics, using local averages: {e}")
                averages = self.running_metrics.copy()
        else:
            # single process - use local averages
            averages = self.running_metrics.copy()
        
        return averages
    
    def reset(self) -> None:
        """Reset tracker for next epoch."""
        self.running_metrics.clear()
        self.running_counts.clear()
        self.total_samples = 0
    
    def get_current_averages(self) -> Dict[str, float]:
        """Get current local averages without distributed operations."""
        return self.running_metrics.copy()


class DistributedEarlyStopping:
    """
    Early stopping that properly synchronizes across distributed processes.
    """
    def __init__(self, patience: int = 10, delta: float = 0.0, 
                 mode: str = 'min', verbose: bool = True) -> None:
        """
        Initialize distributed early stopping.

        Args:
            patience (int): Number of epochs to wait before stopping
            delta (float): Minimum change threshold to qualify as improvement
            mode (str): 'min' for loss (lower is better) or 'max' for accuracy (higher is better)
            verbose (bool): Whether to print early stopping messages
            
        Raises:
            ValueError: If mode is not 'min' or 'max'
        """
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
            
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.mode = mode
        self.delta = delta
        
    def __call__(self, score: float, accelerator) -> bool:
        """
        Check if early stopping should be triggered with distributed synchronization.
        
        Args:
            score (float): Current validation score 
            accelerator: Accelerator instance for distributed operations
            
        Returns:
            bool: True if early stopping should be triggered
        """
        # since validation score is already globally averaged, 
        # all processes can make the same decision without communication
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose and accelerator.is_main_process:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f} \n')
            else:
                self.counter += 1
                if self.verbose and accelerator.is_main_process:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f} \n')
            
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose and accelerator.is_main_process:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f} \n')
            else:
                self.counter += 1
                if self.verbose and accelerator.is_main_process:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f} \n')
            
        self.early_stop = self.counter >= self.patience
        
        # ensure all processes make the same early stopping decision
        if accelerator.use_distributed:
            early_stop_tensor = torch.tensor([1 if self.early_stop else 0], 
                                           device=accelerator.device, dtype=torch.int)
            gathered_decisions = accelerator.gather_for_metrics(early_stop_tensor)
            # if any process decides to stop, all should stop
            should_stop = gathered_decisions.sum().item() > 0
            self.early_stop = should_stop
        
        # print early stop triggered message
        if self.early_stop and self.verbose and accelerator.is_main_process:
            print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f} \n')
        
        return self.early_stop