"""
Brain Utilities

Comprehensive utilities for data handling, logging, and metrics.
"""

# Data utilities
from .data import (
    BrainDataLoader,
    get_mnist_loaders,
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_imagenet_loaders,
    get_huggingface_dataset,
    ImagePreprocessor,
    TextPreprocessor,
    AudioPreprocessor,
    ImageAugmentation,
    CutMix,
    MixUp,
)

# Logging utilities
from .logging import (
    get_logger,
    setup_logging,
    BrainLogger,
    WandBLogger,
    init_wandb,
    MLflowLogger,
    init_mlflow,
    TensorBoardLogger,
    get_tensorboard_writer,
)

# Metrics utilities
from .metrics import (
    MetricsTracker,
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    classification_report,
    compute_metrics,
)

__all__ = [
    # Data
    'BrainDataLoader',
    'get_mnist_loaders',
    'get_cifar10_loaders',
    'get_cifar100_loaders',
    'get_imagenet_loaders',
    'get_huggingface_dataset',
    'ImagePreprocessor',
    'TextPreprocessor',
    'AudioPreprocessor',
    'ImageAugmentation',
    'CutMix',
    'MixUp',

    # Logging
    'get_logger',
    'setup_logging',
    'BrainLogger',
    'WandBLogger',
    'init_wandb',
    'MLflowLogger',
    'init_mlflow',
    'TensorBoardLogger',
    'get_tensorboard_writer',

    # Metrics
    'MetricsTracker',
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'confusion_matrix',
    'classification_report',
    'compute_metrics',
]
