"""
Utility functions and classes for deep learning training.

This package provides common utilities for model training, including:
- Dataset handling
- Learning rate scheduling  
- Miscellaneous helper functions
- Logging and monitoring utilities
"""

from . import misc
from . import datasets
from . import lr_sched

__all__ = [
    'misc',
    'datasets', 
    'lr_sched'
]