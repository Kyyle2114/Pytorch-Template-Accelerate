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


class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0, 
                 mode: str = 'min', verbose: bool = True) -> None:
        """
        Initialize early stopping.

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
        
    def __call__(self, score: float) -> None:
        """
        Check if early stopping should be triggered.
        
        Args:
            score (float): Current validation score
        """
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f} \n')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f} \n')
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f} \n')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f} \n')
                
        if self.counter >= self.patience:
            self.early_stop = True