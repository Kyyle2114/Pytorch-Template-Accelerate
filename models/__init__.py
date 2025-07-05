"""
Deep learning models for various tasks.

This package contains PyTorch model implementations with proper initialization
and modular design for easy extension.
"""

from .cnn import SimpleCNNforCIFAR10

__all__ = [
    'SimpleCNNforCIFAR10'
]
