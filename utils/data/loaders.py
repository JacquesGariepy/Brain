"""
Brain Data Loaders

Comprehensive data loading utilities for common datasets and benchmarks.
Supports vision, text, audio, and multimodal datasets.
"""

import os
from typing import Optional, Tuple, Dict, Any, Callable
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    import torchvision
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from datasets import load_dataset as hf_load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class BrainDataLoader:
    """
    Universal data loader factory for Brain framework.

    Provides unified interface for loading various datasets with
    consistent preprocessing and batching.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Args:
            dataset_name: Name of dataset (mnist, cifar10, cifar100, imagenet, etc.)
            data_dir: Directory to store/load datasets
            batch_size: Batch size for DataLoader
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_loaders(self, **kwargs) -> Tuple[DataLoader, DataLoader]:
        """
        Get train and validation/test DataLoaders for the specified dataset.

        Returns:
            Tuple of (train_loader, val_loader)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for data loading. Install with: pip install torch torchvision")

        loader_map = {
            'mnist': get_mnist_loaders,
            'cifar10': get_cifar10_loaders,
            'cifar100': get_cifar100_loaders,
            'imagenet': get_imagenet_loaders,
        }

        if self.dataset_name in loader_map:
            return loader_map[self.dataset_name](
                data_dir=str(self.data_dir),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                **kwargs
            )
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported. Available: {list(loader_map.keys())}")


def get_mnist_loaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST dataset loaders.

    Args:
        data_dir: Directory to store/load dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
        download: Whether to download dataset if not present

    Returns:
        (train_loader, test_loader)

    Example:
        >>> train_loader, test_loader = get_mnist_loaders(batch_size=64)
        >>> for images, labels in train_loader:
        ...     # images: [64, 1, 28, 28]
        ...     # labels: [64]
        ...     pass
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required: pip install torch torchvision")

    # MNIST transform: normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 dataset loaders.

    Args:
        data_dir: Directory to store/load dataset
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Pin memory for GPU
        download: Download if not present
        augment: Apply data augmentation to training set

    Returns:
        (train_loader, test_loader)

    Example:
        >>> train_loader, test_loader = get_cifar10_loaders(augment=True)
        >>> for images, labels in train_loader:
        ...     # images: [32, 3, 32, 32]
        ...     # labels: [32]
        ...     pass
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required: pip install torch torchvision")

    # Normalization constants for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # Training transform with augmentation
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader


def get_cifar100_loaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-100 dataset loaders.

    Args:
        data_dir: Directory to store/load dataset
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Pin memory for GPU
        download: Download if not present
        augment: Apply data augmentation

    Returns:
        (train_loader, test_loader)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required: pip install torch torchvision")

    # CIFAR-100 normalization (same as CIFAR-10)
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader


def get_imagenet_loaders(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    image_size: int = 224,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ImageNet dataset loaders.

    NOTE: ImageNet must be manually downloaded and organized in the standard format:
    data_dir/
        train/
            n01440764/
                n01440764_10026.JPEG
                ...
            n01443537/
                ...
        val/
            n01440764/
                ...

    Args:
        data_dir: Path to ImageNet root directory
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Pin memory for GPU
        image_size: Input image size (default: 224)
        augment: Apply data augmentation

    Returns:
        (train_loader, val_loader)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required: pip install torch torchvision")

    # ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"ImageNet train directory not found at {train_dir}. "
            f"Please download ImageNet manually."
        )

    train_dataset = torchvision.datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = torchvision.datasets.ImageFolder(
        val_dir,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def get_huggingface_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "train",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    **kwargs
):
    """
    Load a dataset from HuggingFace Datasets.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "glue", "squad", "wikitext")
        config_name: Dataset configuration (e.g., "sst2" for GLUE)
        split: Dataset split ("train", "validation", "test")
        cache_dir: Directory to cache dataset
        streaming: Whether to use streaming mode for large datasets
        **kwargs: Additional arguments for load_dataset

    Returns:
        HuggingFace Dataset object

    Example:
        >>> # Load GLUE SST-2
        >>> dataset = get_huggingface_dataset("glue", "sst2", split="train")
        >>>
        >>> # Load WikiText-103
        >>> dataset = get_huggingface_dataset("wikitext", "wikitext-103-v1")
        >>>
        >>> # Load SQuAD v2
        >>> dataset = get_huggingface_dataset("squad_v2", split="train")
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "HuggingFace Datasets required: pip install datasets"
        )

    return hf_load_dataset(
        dataset_name,
        config_name,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        **kwargs
    )


def get_text_dataset(
    dataset_name: str,
    tokenizer: Optional[Any] = None,
    max_length: int = 512,
    batch_size: int = 32,
    split: str = "train",
    **kwargs
):
    """
    Load a text dataset with optional tokenization.

    Args:
        dataset_name: Name of the text dataset
        tokenizer: Optional tokenizer (HuggingFace tokenizer)
        max_length: Maximum sequence length
        batch_size: Batch size
        split: Dataset split
        **kwargs: Additional arguments

    Returns:
        Dataset or DataLoader depending on tokenizer

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> dataset = get_text_dataset("wikitext", tokenizer=tokenizer)
    """
    # Load base dataset
    dataset = get_huggingface_dataset(dataset_name, split=split, **kwargs)

    if tokenizer is not None:
        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        dataset = dataset.map(tokenize_function, batched=True)

    return dataset


def get_multimodal_dataset(
    dataset_name: str,
    image_processor: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    split: str = "train",
    **kwargs
):
    """
    Load a multimodal (vision + text) dataset.

    Args:
        dataset_name: Name of multimodal dataset (e.g., "coco", "flickr30k")
        image_processor: Image preprocessing function/processor
        tokenizer: Text tokenizer
        split: Dataset split
        **kwargs: Additional arguments

    Returns:
        Multimodal dataset

    Example:
        >>> from transformers import CLIPProcessor
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> dataset = get_multimodal_dataset("coco", processor=processor)
    """
    dataset = get_huggingface_dataset(dataset_name, split=split, **kwargs)

    if image_processor is not None or tokenizer is not None:
        def process_function(examples):
            processed = {}

            if image_processor is not None and "image" in examples:
                processed["pixel_values"] = image_processor(
                    examples["image"],
                    return_tensors="pt"
                ).pixel_values

            if tokenizer is not None and "text" in examples:
                processed.update(tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                ))

            return processed

        dataset = dataset.map(process_function, batched=True)

    return dataset


# Convenience function for testing
def test_loaders():
    """Test all data loaders"""
    print("Testing Brain Data Loaders...")

    if TORCH_AVAILABLE:
        print("\n1. Testing MNIST...")
        try:
            train_loader, test_loader = get_mnist_loaders(batch_size=64, download=True)
            batch = next(iter(train_loader))
            print(f"   ✓ MNIST: images {batch[0].shape}, labels {batch[1].shape}")
        except Exception as e:
            print(f"   ✗ MNIST failed: {e}")

        print("\n2. Testing CIFAR-10...")
        try:
            train_loader, test_loader = get_cifar10_loaders(batch_size=64, download=True)
            batch = next(iter(train_loader))
            print(f"   ✓ CIFAR-10: images {batch[0].shape}, labels {batch[1].shape}")
        except Exception as e:
            print(f"   ✗ CIFAR-10 failed: {e}")
    else:
        print("PyTorch not available, skipping vision dataset tests")

    if HF_AVAILABLE:
        print("\n3. Testing HuggingFace Datasets...")
        try:
            dataset = get_huggingface_dataset("glue", "sst2", split="train[:10]")
            print(f"   ✓ HuggingFace: loaded {len(dataset)} examples")
        except Exception as e:
            print(f"   ✗ HuggingFace failed: {e}")
    else:
        print("HuggingFace Datasets not available")

    print("\n✓ Data loader tests complete!")


if __name__ == "__main__":
    test_loaders()
