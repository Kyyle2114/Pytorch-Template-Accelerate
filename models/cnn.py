import torch
import torch.nn as nn

class SimpleCNNforCIFAR10(nn.Module):
    """
    Simple CNN model for CIFAR10 classification.
    
    Architecture:
    - 3 Convolutional blocks with batch normalization, ReLU, and max pooling
    - 2 Fully connected layers with dropout for regularization
    - Proper weight initialization
    """
    
    def __init__(self, num_classes: int = 10, dropout_p: float = 0.5) -> None:
        """
        Initialize the CNN model.
        
        Args:
            num_classes (int): Number of output classes. Defaults to 10 (CIFAR10)
            dropout_p (float): Dropout probability. Defaults to 0.5
            
        Raises:
            ValueError: If num_classes <= 0 or dropout_p not in [0, 1]
        """
        super().__init__()
        
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not 0 <= dropout_p <= 1:
            raise ValueError(f"dropout_p must be in [0, 1], got {dropout_p}")
        
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        
        # --- Convolutional layers ---
        self.features = nn.Sequential(
            # first conv block: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            # second conv block: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            
            # third conv block: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )
        
        # --- Classifier ---
        # calculate the size after convolutions: 128 * 4 * 4 = 2048
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(512, self.num_classes)
        )
        
        # --- Initialize weights ---
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        """
        Initialize the weights of the model using best practices.
        
        - Conv2d: Kaiming normal initialization (He initialization)
        - BatchNorm2d: weight=1, bias=0
        - Linear: Xavier normal initialization with small bias
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
            
        Raises:
            RuntimeError: If input tensor has wrong shape
        """
        expected_shape = (x.size(0), 3, 32, 32)
        if x.shape != expected_shape:
            raise RuntimeError(f"Expected input shape {expected_shape}, got {x.shape}")
            
        x = self.features(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature maps before the classifier.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Feature maps of shape (batch_size, 128, 4, 4)
        """
        return self.features(x)
    
    @property
    def num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)