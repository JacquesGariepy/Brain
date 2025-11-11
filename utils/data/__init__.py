"""
Brain Data Utilities

Provides comprehensive data handling infrastructure for scientific research:
- Dataset loaders for common benchmarks
- Preprocessing and augmentation utilities
- HuggingFace Datasets integration
- Custom dataset builders
"""

from .loaders import (
    BrainDataLoader,
    get_mnist_loaders,
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_imagenet_loaders,
    get_huggingface_dataset,
    get_text_dataset,
    get_multimodal_dataset,
)

from .preprocessing import (
    ImagePreprocessor,
    TextPreprocessor,
    AudioPreprocessor,
    get_image_transforms,
    get_text_tokenizer,
    normalize_tensor,
)

from .augmentation import (
    ImageAugmentation,
    TextAugmentation,
    get_train_augmentation,
    get_val_augmentation,
)

__all__ = [
    # Loaders
    'BrainDataLoader',
    'get_mnist_loaders',
    'get_cifar10_loaders',
    'get_cifar100_loaders',
    'get_imagenet_loaders',
    'get_huggingface_dataset',
    'get_text_dataset',
    'get_multimodal_dataset',

    # Preprocessing
    'ImagePreprocessor',
    'TextPreprocessor',
    'AudioPreprocessor',
    'get_image_transforms',
    'get_text_tokenizer',
    'normalize_tensor',

    # Augmentation
    'ImageAugmentation',
    'TextAugmentation',
    'get_train_augmentation',
    'get_val_augmentation',
]
