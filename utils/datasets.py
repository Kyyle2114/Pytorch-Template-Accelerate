import os
import cv2
from typing import Optional, Tuple, Union, List
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets

class CustomDataset(Dataset):
    """
    Custom dataset for image classification.
    
    Note: This is a basic implementation. For supervised learning,
    you need to modify this to include labels.
    """
    def __init__(self, dataset_path: str, transform: Optional[torch.nn.Module] = None) -> None:
        """
        Initialize custom dataset.
        
        Args:
            dataset_path (str): Path to the image directory
            transform (Optional[torch.nn.Module]): Optional transform to apply to images
            
        Raises:
            FileNotFoundError: If dataset_path doesn't exist
            ValueError: If no images found in the directory
        """
        self.image_dir = Path(dataset_path)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
            
        self.transform = transform
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.data_images = [
            f for f in os.listdir(self.image_dir) 
            if Path(f).suffix.lower() in valid_extensions
        ]
        
        if len(self.data_images) == 0:
            raise ValueError(f"No valid image files found in {dataset_path}")
        
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, int]]: Image tensor or (image, label) tuple
            
        Note: Currently returns only image. For supervised learning, modify to return (image, label).
        """
        img_path = self.image_dir / self.data_images[idx]
        
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)
            
        # TODO: For supervised learning, return (image, label)
        # For now, returning only image for compatibility
        return image
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_images)
    

# --- CIFAR10 for testing ---

class CIFAR10Dataset(Dataset):
    def __init__(
        self, 
        root: str, 
        train: bool = True, 
        transform: Optional[torch.nn.Module] = None
    ) -> None:
        """
        Initialize CIFAR10 Dataset wrapper.
        
        Args:
            root (str): Root directory of dataset where CIFAR10 will be downloaded
            train (bool): If True, creates dataset from training set, otherwise from test set
            transform (Optional[torch.nn.Module]): Optional transform to be applied on a sample
            
        Raises:
            RuntimeError: If dataset download or loading fails
        """
        try:
            self.dataset = datasets.CIFAR10(
                root=root,
                train=train,
                download=True,
                transform=transform
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CIFAR10 dataset: {e}")
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Tuple[torch.Tensor, int]: (image, target) where target is the index of the target class
            
        Raises:
            IndexError: If idx is out of range
        """
        if idx >= len(self.dataset) or idx < 0:
            raise IndexError(f"Index {idx} is out of range for dataset of size {len(self.dataset)}")
        return self.dataset[idx]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)
    
    @property
    def classes(self) -> List[str]:
        """Get the list of class names."""
        return self.dataset.classes


def make_cifar10_dataset(
    dataset_path: str,
    train: bool = True,
    transform: Optional[torch.nn.Module] = None
) -> CIFAR10Dataset:
    """
    Factory function to create PyTorch Dataset for CIFAR10.
    
    Args:
        dataset_path (str): Root directory where CIFAR10 will be downloaded
        train (bool): If True, creates dataset from training set, otherwise from test set
        transform (Optional[torch.nn.Module]): Optional transform to be applied on a sample
        
    Returns:
        CIFAR10Dataset: PyTorch Dataset for CIFAR10
        
    Raises:
        ValueError: If dataset_path is invalid
        RuntimeError: If dataset creation fails
    """
    if not dataset_path:
        raise ValueError("dataset_path cannot be empty")
        
    try:
        dataset = CIFAR10Dataset(
            root=dataset_path,
            train=train,
            transform=transform
        )
        return dataset
    
    except Exception as e:
        raise RuntimeError(f"Failed to create CIFAR10 dataset: {e}")