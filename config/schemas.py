from pydantic import BaseModel, ConfigDict, Field


class ConfigBase(BaseModel):
    """Base configuration class with common functionality.
    
    Provides dictionary-style access and strict validation settings.
    All config sections should inherit from this class.
    """
    
    model_config = ConfigDict(
        extra='forbid',  # raise error on unknown fields
        validate_default=True,  # validate default values
        str_strip_whitespace=True,  # strip whitespace from strings
    )
    
    def __getitem__(self, key: str):
        """Enable dictionary-style access (e.g., config['seed'])."""
        return getattr(self, key)
    
    def __setitem__(self, key: str, value) -> None:
        """Enable dictionary-style assignment (e.g., config['seed'] = 42)."""
        setattr(self, key, value)
    
    def get(self, key: str, default=None):
        """Get attribute with optional default value."""
        return getattr(self, key, default)


class GeneralConfig(ConfigBase):
    """General configuration settings."""
    
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
        ge=0
    )
    output_dir: str = Field(
        default="./output_dir",
        description="Path where to save checkpoints and logs"
    )


class DataConfig(ConfigBase):
    """Dataset configuration settings."""
    
    dataset_path: str = Field(
        default="./dataset",
        description="Path to the dataset root directory"
    )
    batch_size: int = Field(
        default=16,
        description="Batch size per GPU",
        ge=1
    )
    num_workers: int = Field(
        default=4,
        description="Number of subprocesses for data loading",
        ge=0
    )


class TrainingConfig(ConfigBase):
    """Training configuration settings."""
    
    epoch: int = Field(
        default=100,
        description="Total number of training epochs",
        ge=1
    )
    patience: int = Field(
        default=50,
        description="Patience for early stopping (epochs)",
        ge=1
    )
    lr: float = Field(
        default=1e-3,
        description="Base learning rate",
        gt=0
    )
    weight_decay: float = Field(
        default=1e-4,
        description="Weight decay for optimizer",
        ge=0
    )
    grad_accum_steps: int = Field(
        default=1,
        description="Gradient accumulation steps to increase effective batch size",
        ge=1
    )
    warmup_epochs: int = Field(
        default=10,
        description="Number of warmup epochs for learning rate scheduler",
        ge=0
    )
    clip_grad: float | None = Field(
        default=None,
        description="Gradient clipping norm (None for no clipping)",
        gt=0
    )


class WandBConfig(ConfigBase):
    """Weights & Biases configuration settings."""
    
    project_name: str = Field(
        default="Model-Training",
        description="WandB project name"
    )
    run_name: str = Field(
        default="Model-Training",
        description="WandB run name"
    )


class Config(ConfigBase):
    """
    Overall configuration structure for model training.
    
    This class aggregates all configuration sections into a single
    validated configuration object.
    """
    
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)
