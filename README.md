# 🚀 PyTorch Template with Accelerate

> **Notice**: This repository is a modified version of [@Kyyle2114/Pytorch-Template](https://github.com/Kyyle2114/Pytorch-Template), enhanced to support training with [Hugging Face's Accelerate](https://huggingface.co/docs/accelerate/index) and improved with comprehensive type hints, error handling, and best practices.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Accelerate](https://img.shields.io/badge/🤗-Accelerate-yellow.svg)](https://huggingface.co/docs/accelerate)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Structured and modular template for deep learning model training with enterprise-grade code quality.

## ✨ Features

### 🔧 Core Functionality
- **Distributed Training**: Full support via Hugging Face Accelerate
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Error Handling**: Robust exception handling with specific error types
- **Modular Design**: Clean separation of concerns with well-defined interfaces
- **Configuration Management**: Flexible hyperparameter configuration via CLI arguments

### 📊 Monitoring & Logging
- **Weights & Biases Integration**: Automatic experiment tracking and visualization
- **Early Stopping**: Configurable early stopping to prevent overfitting
- **Model Checkpointing**: Automatic saving of best models and periodic checkpoints
- **Progress Tracking**: Detailed logging of training progress and metrics

### 🏗️ Code Quality
- **PEP 8 Compliance**: Consistent code formatting with Ruff
- **Documentation**: Google-style docstrings for all functions and classes
- **Input Validation**: Comprehensive argument validation with informative error messages
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

# Or using venv
python -m venv pytorch-template
source pytorch-template/bin/activate  # On Windows: pytorch-template\Scripts\activate
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

#### Method 1: Using the training script (Recommended)
```bash
chmod +x train.sh
./train.sh
```

#### Method 2: Direct execution
```bash
accelerate launch main.py \
    --output_dir ./output_dir \
    --dataset_path ./dataset \
    --batch_size 32 \
    --epoch 100 \
    --lr 1e-3 \
    --project_name "My-CIFAR10-Experiment" \
    --run_name "baseline-model"
```

## 📁 Project Structure

```
Pytorch-Template-Accelerate/
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
├── dataset/                 # Dataset storage (auto-created)
├── output_dir/              # Training outputs (auto-created)
├── main.py                  # Main training script
├── train.sh                 # Training execution script
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## ⚙️ Configuration

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | int | 21 | Random seed for reproducibility |
| `--output_dir` | str | ./output_dir | Output directory for checkpoints and logs |
| `--dataset_path` | str | ./dataset | Dataset root path |
| `--batch_size` | int | 16 | Batch size per GPU |
| `--epoch` | int | 10 | Total number of training epochs |
| `--lr` | float | 1e-3 | Base learning rate |
| `--weight_decay` | float | 1e-4 | Weight decay for optimizer |
| `--warmup_epochs` | int | 10 | Number of warmup epochs |
| `--patience` | int | 50 | Early stopping patience |
| `--clip_grad` | float | None | Gradient clipping norm |
| `--num_workers` | int | 4 | Number of data loading workers |
| `--accum_iter` | int | 1 | Gradient accumulation steps |
| `--project_name` | str | Model-Training | WandB project name |
| `--run_name` | str | Model-Training | WandB run name |

> **Note**: For a complete list of all available arguments and their descriptions, please refer to the `get_args_parser` function in `main.py`.

### Environment Variables

For seamless WandB integration, set your API key:
```bash
export WANDB_API_KEY="your-wandb-api-key"
```

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

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `batch_size` in `train.sh`
- Increase `accum_iter` to maintain effective batch size

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

## 📊 Monitoring

### WandB Dashboard
Training metrics automatically logged to WandB include:
- Training/validation loss
- Learning rate schedule
- Model accuracy
- Hardware utilization
- Hyperparameter configurations

### Local Logs
- JSON logs: `output_dir/log.txt`
- Model checkpoints: `output_dir/checkpoint-*/`
- Best model: `output_dir/best_model/`
- Configuration: `output_dir/args.json`

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
- [Original Template by @Kyyle2114](https://github.com/Kyyle2114/Pytorch-Template)

---

⭐ **Star this repository if you find it helpful!**