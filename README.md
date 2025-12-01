# 🚀 PyTorch Template with Accelerate

> **Notice**: This repository is a modified version of [@Kyyle2114/Pytorch-Template](https://github.com/Kyyle2114/Pytorch-Template), enhanced to support training with [Hugging Face's Accelerate](https://huggingface.co/docs/accelerate/index) and improved with comprehensive type hints, error handling, and best practices.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Accelerate](https://img.shields.io/badge/🤗-Accelerate-yellow.svg)](https://huggingface.co/docs/accelerate)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Structured and modular template for deep learning model training with hugging face's accelerate.

## ✨ Features

### 🔧 Core Functionality
- **Distributed Training**: Full support via Hugging Face Accelerate
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Error Handling**: Robust exception handling with specific error types
- **Modular Design**: Clean separation of concerns with well-defined interfaces
- **Configuration Management**: YAML-based configuration with Pydantic validation

### 📊 Monitoring & Logging
- **Weights & Biases Integration**: Automatic experiment tracking and visualization
- **Early Stopping**: Configurable early stopping to prevent overfitting
- **Model Checkpointing**: Automatic saving of best and last models
- **Progress Tracking**: Detailed logging of training progress and metrics

### 🏗️ Code Quality
- **PEP 8 Compliance**: Consistent code formatting with Ruff
- **Documentation**: Google-style docstrings for all functions and classes
- **Input Validation**: Pydantic-based config validation with informative error messages
- **Memory Management**: Efficient memory usage with proper cleanup

### 🔄 Training Pipeline
- **Learning Rate Scheduling**: Cosine annealing with warm-up restarts
- **Gradient Accumulation**: Support for effective batch size scaling
- **Mixed Precision**: FP16 training for improved performance
- **Gradient Clipping**: Configurable gradient norm clipping

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (optional but recommended)
- Git

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/Pytorch-Template-Accelerate.git
cd Pytorch-Template-Accelerate
```

2. **Create virtual environment**:
```bash
# Using conda
conda create -n pytorch-template python=3.10
conda activate pytorch-template
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure Accelerate** (first time only):
```bash
accelerate config
```

### 🏃‍♂️ Running Training

#### Using the training script
```bash
chmod +x train.sh
bash train.sh
```

#### Using Python directly
```bash
# Single GPU
python main.py --config config/default.yaml

# Multi-GPU with Accelerate
accelerate launch main.py --config config/default.yaml

# With config overrides
python main.py --config config/default.yaml --set training.lr=0.0001 --set data.batch_size=32
```

## 📁 Project Structure

```
Pytorch-Template-Accelerate/
├── config/
│   ├── __init__.py          # Config package initialization
│   ├── default.yaml         # Default configuration file
│   └── schemas.py           # Pydantic configuration schemas
├── engines/
│   ├── __init__.py          # Engine package initialization
│   └── engine_train.py      # Training and evaluation logic
├── models/
│   ├── __init__.py          # Models package initialization
│   └── cnn.py               # CNN model implementation
├── utils/
│   ├── __init__.py          # Utils package initialization
│   ├── datasets.py          # Dataset handling utilities
│   ├── lr_sched.py          # Learning rate schedulers
│   └── misc.py              # Miscellaneous utilities
├── dataset/                 # Dataset storage 
├── output_dir/              # Training outputs (auto-created)
├── main.py                  # Main training script
├── train.sh                 # Training execution script
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## ⚙️ Configuration

### YAML Configuration File

Configuration is managed through YAML files with Pydantic validation. The default configuration file is `config/default.yaml`:

```yaml
# --- General Configuration ---
general:
  seed: 42
  output_dir: ./output_dir

# --- Data Configuration ---
data:
  dataset_path: ./dataset
  batch_size: 16
  num_workers: 4

# --- Training Configuration ---
training:
  epoch: 100
  patience: 50
  lr: 1e-3
  weight_decay: 1e-4
  grad_accum_steps: 1
  warmup_epochs: 10
  clip_grad: null   # null for no gradient clipping

# --- WandB Configuration ---
wandb:
  project_name: Model-Training
  run_name: Model-Training
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-c, --config` | str | config/default.yaml | Path to YAML configuration file |
| `--help_config` | flag | - | Show detailed help for configuration parameters |
| `--set` | KEY=VALUE | - | Override any config value (can be used multiple times) |

### Configuration Override Examples

```bash
# Override single value
python main.py --config config/default.yaml --set training.lr=0.0001

# Override multiple values
python main.py --config config/default.yaml \
    --set general.seed=42 \
    --set data.batch_size=32 \
    --set training.epoch=50

# Show config help
python main.py --help_config
```

### Configuration Parameters

| Section | Parameter | Type | Default | Description |
|---------|-----------|------|---------|-------------|
| **general** | seed | int | 42 | Random seed for reproducibility |
| | output_dir | str | ./output_dir | Output directory for checkpoints and logs |
| **data** | dataset_path | str | ./dataset | Dataset root path |
| | batch_size | int | 16 | Batch size per GPU |
| | num_workers | int | 4 | Number of data loading workers |
| **training** | epoch | int | 100 | Total number of training epochs |
| | patience | int | 50 | Early stopping patience |
| | lr | float | 1e-3 | Base learning rate |
| | weight_decay | float | 1e-4 | Weight decay for optimizer |
| | grad_accum_steps | int | 1 | Gradient accumulation steps |
| | warmup_epochs | int | 10 | Number of warmup epochs |
| | clip_grad | float/null | null | Gradient clipping norm (null for no clipping) |
| **wandb** | project_name | str | Model-Training | WandB project name |
| | run_name | str | Model-Training | WandB run name |

## 🔧 Customization

### Adding a New Model

1. Create your model in `models/your_model.py`:
```python
import torch.nn as nn
from typing import Optional

class YourModel(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Your model implementation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass implementation
        return x
```

2. Update `models/__init__.py`:
```python
from .your_model import YourModel
__all__ = ['SimpleCNNforCIFAR10', 'YourModel']
```

3. Modify `main.py` to use your model:
```python
model = models.YourModel(num_classes=10)
```

### Adding a New Dataset

1. Implement your dataset in `utils/datasets.py`:
```python
class YourDataset(Dataset):
    def __init__(self, data_path: str, transform: Optional[torch.nn.Module] = None):
        # Your dataset implementation
        pass
```

2. Create a factory function:
```python
def make_your_dataset(dataset_path: str, train: bool = True, transform=None):
    return YourDataset(dataset_path, transform)
```

### Creating a Custom Configuration

1. Create a new YAML file (e.g., `config/my_config.yaml`):
```yaml
general:
  seed: 123
  output_dir: ./my_experiment

data:
  dataset_path: /path/to/your/data
  batch_size: 64
  num_workers: 8

training:
  epoch: 200
  patience: 30
  lr: 5e-4
  weight_decay: 1e-5
  grad_accum_steps: 2
  warmup_epochs: 5
  clip_grad: 1.0

wandb:
  project_name: My-Project
  run_name: experiment-1
```

2. Run training with your config:
```bash
python main.py --config config/my_config.yaml
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `data.batch_size` in your config
- Increase `training.grad_accum_steps` to maintain effective batch size

**2. WandB Login Issues**
- Set `WANDB_API_KEY` environment variable
- Use `wandb login` command
- Set `WANDB_MODE=offline` for offline logging

**3. Import Errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: requires 3.10+

**4. Multi-GPU Training Issues**
- Run `accelerate config` to set up distributed training
- Ensure `CUDA_VISIBLE_DEVICES` is set correctly in `train.sh`

**5. Config Validation Errors**
- Check YAML syntax in your config file
- Ensure all required fields are present
- Use `--help_config` to see available parameters

## 📊 Monitoring

### WandB Dashboard
Training metrics automatically logged to WandB include:
- Training/validation loss
- Learning rate schedule
- Model accuracy
- Best model metrics (epoch, loss, accuracy)
- Hardware utilization
- Hyperparameter configurations

### Local Logs
- JSON logs: `output_dir/log.txt`
- Best model: `output_dir/best_model/`
- Last model: `output_dir/last_model/`
- Configuration: `output_dir/config.yaml`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with proper type hints and documentation
4. Run tests and ensure code quality
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📚 References

- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Weights & Biases](https://wandb.ai/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Original Template by @Kyyle2114](https://github.com/Kyyle2114/Pytorch-Template)

---

⭐ **Star this repository if you find it helpful!**
