"""
Training engines for deep learning models.

This package provides training and evaluation utilities for PyTorch models
with support for distributed training via Accelerate.
"""

from .engine_train import train_one_epoch, evaluate

__all__ = [
    'train_one_epoch',
    'evaluate'
]