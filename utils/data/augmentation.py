"""
Brain Data Augmentation

Advanced data augmentation techniques for vision, text, and audio.
"""

from typing import Optional, List, Tuple, Union
import random

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ImageAugmentation:
    """
    Advanced image augmentation for computer vision tasks.

    Implements state-of-the-art augmentation techniques including:
    - RandAugment
    - AutoAugment
    - CutMix
    - MixUp
    - Random Erasing
    """

    def __init__(
        self,
        image_size: int = 224,
        augmentation_level: str = "medium",
        use_randaugment: bool = True,
        use_random_erasing: bool = True,
    ):
        """
        Args:
            image_size: Target image size
            augmentation_level: "light", "medium", "heavy"
            use_randaugment: Whether to use RandAugment
            use_random_erasing: Whether to use Random Erasing
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required: pip install torch torchvision")

        self.image_size = image_size
        self.augmentation_level = augmentation_level
        self.use_randaugment = use_randaugment
        self.use_random_erasing = use_random_erasing

        self.transform = self._build_transform()

    def _build_transform(self):
        """Build augmentation pipeline based on level"""
        augmentations = []

        if self.augmentation_level == "light":
            augmentations.extend([
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        elif self.augmentation_level == "medium":
            augmentations.extend([
                transforms.RandomResizedCrop(self.image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ])
        elif self.augmentation_level == "heavy":
            augmentations.extend([
                transforms.RandomResizedCrop(self.image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
                transforms.RandomGrayscale(p=0.1),
            ])

        # Add RandAugment if requested
        if self.use_randaugment:
            try:
                augmentations.append(transforms.RandAugment())
            except AttributeError:
                # RandAugment not available in older torchvision versions
                pass

        # Convert to tensor
        augmentations.append(transforms.ToTensor())

        # Add Random Erasing if requested
        if self.use_random_erasing:
            augmentations.append(
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            )

        # Normalize
        augmentations.append(
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )

        return transforms.Compose(augmentations)

    def __call__(self, image):
        """Apply augmentation to image"""
        return self.transform(image)


class CutMix:
    """
    CutMix augmentation for image classification.

    Reference: https://arxiv.org/abs/1905.04899
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Beta distribution parameter (larger = more aggressive mixing)
        """
        self.alpha = alpha

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to a batch of images.

        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B]

        Returns:
            mixed_images: Mixed images
            labels_a: First set of labels
            labels_b: Second set of labels
            lam: Mixing coefficient
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        batch_size = images.size(0)
        lam = random.betavariate(self.alpha, self.alpha) if self.alpha > 0 else 1.0

        # Random permutation
        indices = torch.randperm(batch_size)

        # Get random box
        H, W = images.size(2), images.size(3)
        cut_rat = (1.0 - lam) ** 0.5
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Random center
        cx = random.randint(0, W)
        cy = random.randint(0, H)

        # Bounding box
        bbx1 = max(cx - cut_w // 2, 0)
        bby1 = max(cy - cut_h // 2, 0)
        bbx2 = min(cx + cut_w // 2, W)
        bby2 = min(cy + cut_h // 2, H)

        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda based on actual cut size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return mixed_images, labels, labels[indices], lam


class MixUp:
    """
    MixUp augmentation for image classification.

    Reference: https://arxiv.org/abs/1710.09412
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to a batch.

        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B]

        Returns:
            mixed_images: Mixed images
            labels_a: First set of labels
            labels_b: Second set of labels
            lam: Mixing coefficient
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        batch_size = images.size(0)
        lam = random.betavariate(self.alpha, self.alpha) if self.alpha > 0 else 1.0

        # Random permutation
        indices = torch.randperm(batch_size)

        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]

        return mixed_images, labels, labels[indices], lam


class TextAugmentation:
    """
    Text augmentation techniques for NLP tasks.

    Implements:
    - Random word deletion
    - Random word swap
    - Synonym replacement
    - Back translation (if available)
    """

    def __init__(
        self,
        deletion_prob: float = 0.1,
        swap_prob: float = 0.1,
        insert_prob: float = 0.1,
    ):
        """
        Args:
            deletion_prob: Probability of deleting each word
            swap_prob: Probability of swapping adjacent words
            insert_prob: Probability of inserting random word
        """
        self.deletion_prob = deletion_prob
        self.swap_prob = swap_prob
        self.insert_prob = insert_prob

    def random_deletion(self, words: List[str]) -> List[str]:
        """
        Randomly delete words with probability deletion_prob.

        Args:
            words: List of words

        Returns:
            Augmented word list
        """
        if len(words) == 1:
            return words

        new_words = []
        for word in words:
            if random.random() > self.deletion_prob:
                new_words.append(word)

        # If all words deleted, return a random word
        if len(new_words) == 0:
            return [random.choice(words)]

        return new_words

    def random_swap(self, words: List[str], n_swaps: int = 1) -> List[str]:
        """
        Randomly swap two words n_swaps times.

        Args:
            words: List of words
            n_swaps: Number of swaps to perform

        Returns:
            Augmented word list
        """
        new_words = words.copy()

        for _ in range(n_swaps):
            if len(new_words) < 2:
                break

            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

        return new_words

    def __call__(self, text: str) -> str:
        """
        Apply augmentation to text.

        Args:
            text: Input text

        Returns:
            Augmented text
        """
        words = text.split()

        # Apply augmentations
        if random.random() < 0.5:
            words = self.random_deletion(words)
        if random.random() < 0.5:
            words = self.random_swap(words)

        return ' '.join(words)


def get_train_augmentation(
    image_size: int = 224,
    level: str = "medium",
) -> ImageAugmentation:
    """
    Get standard training augmentation.

    Args:
        image_size: Target image size
        level: Augmentation level ("light", "medium", "heavy")

    Returns:
        ImageAugmentation instance

    Example:
        >>> augmentation = get_train_augmentation(image_size=224, level="heavy")
        >>> augmented_image = augmentation(pil_image)
    """
    return ImageAugmentation(
        image_size=image_size,
        augmentation_level=level,
        use_randaugment=True,
        use_random_erasing=True,
    )


def get_val_augmentation(image_size: int = 224):
    """
    Get validation augmentation (minimal, just resize and normalize).

    Args:
        image_size: Target image size

    Returns:
        Transform pipeline

    Example:
        >>> val_transform = get_val_augmentation(224)
        >>> val_image = val_transform(pil_image)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def cutmix_criterion(
    criterion,
    outputs: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
):
    """
    Compute CutMix loss.

    Args:
        criterion: Loss function
        outputs: Model predictions
        labels_a: First set of labels
        labels_b: Second set of labels
        lam: Mixing coefficient

    Returns:
        Mixed loss

    Example:
        >>> criterion = nn.CrossEntropyLoss()
        >>> loss = cutmix_criterion(criterion, outputs, labels_a, labels_b, lam)
    """
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


def mixup_criterion(
    criterion,
    outputs: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
):
    """
    Compute MixUp loss.

    Args:
        criterion: Loss function
        outputs: Model predictions
        labels_a: First set of labels
        labels_b: Second set of labels
        lam: Mixing coefficient

    Returns:
        Mixed loss

    Example:
        >>> criterion = nn.CrossEntropyLoss()
        >>> loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
    """
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


# Test function
def test_augmentation():
    """Test augmentation modules"""
    print("Testing Brain Augmentation...")

    if TORCH_AVAILABLE:
        print("\n1. Testing Image Augmentation...")
        try:
            from PIL import Image
            import numpy as np

            # Create dummy image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            augmentation = ImageAugmentation(image_size=224, augmentation_level="medium")
            output = augmentation(img)
            print(f"   ✓ Image augmentation: {output.shape}")
        except Exception as e:
            print(f"   ✗ Image augmentation failed: {e}")

        print("\n2. Testing CutMix...")
        try:
            cutmix = CutMix(alpha=1.0)
            images = torch.randn(4, 3, 224, 224)
            labels = torch.randint(0, 10, (4,))

            mixed_images, labels_a, labels_b, lam = cutmix(images, labels)
            print(f"   ✓ CutMix: mixed_images {mixed_images.shape}, lam {lam:.3f}")
        except Exception as e:
            print(f"   ✗ CutMix failed: {e}")

        print("\n3. Testing MixUp...")
        try:
            mixup = MixUp(alpha=1.0)
            images = torch.randn(4, 3, 224, 224)
            labels = torch.randint(0, 10, (4,))

            mixed_images, labels_a, labels_b, lam = mixup(images, labels)
            print(f"   ✓ MixUp: mixed_images {mixed_images.shape}, lam {lam:.3f}")
        except Exception as e:
            print(f"   ✗ MixUp failed: {e}")

    print("\n4. Testing Text Augmentation...")
    try:
        augmentation = TextAugmentation(deletion_prob=0.1, swap_prob=0.1)
        text = "This is a test sentence for text augmentation"
        augmented = augmentation(text)
        print(f"   ✓ Text augmentation: '{text}' -> '{augmented}'")
    except Exception as e:
        print(f"   ✗ Text augmentation failed: {e}")

    print("\n✓ Augmentation tests complete!")


if __name__ == "__main__":
    test_augmentation()
