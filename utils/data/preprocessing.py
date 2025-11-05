"""
Brain Data Preprocessing

Comprehensive preprocessing utilities for vision, text, and audio data.
"""

from typing import Optional, List, Union, Tuple, Any
import warnings

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ImagePreprocessor:
    """
    Comprehensive image preprocessing for Brain models.

    Handles resizing, normalization, augmentation, and format conversion.
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        resize_mode: str = "center_crop",
        interpolation: str = "bilinear",
    ):
        """
        Args:
            image_size: Target image size (int or (height, width))
            mean: Normalization mean (ImageNet default)
            std: Normalization std (ImageNet default)
            resize_mode: "center_crop", "resize", or "random_crop"
            interpolation: Interpolation mode for resizing
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required: pip install torch torchvision")

        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.mean = mean
        self.std = std
        self.resize_mode = resize_mode

        # Build transform pipeline
        self.transform = self._build_transform()

    def _build_transform(self):
        """Build torchvision transform pipeline"""
        transform_list = []

        # Resize
        if self.resize_mode == "center_crop":
            transform_list.extend([
                transforms.Resize(int(self.image_size[0] * 1.15)),
                transforms.CenterCrop(self.image_size),
            ])
        elif self.resize_mode == "resize":
            transform_list.append(transforms.Resize(self.image_size))
        elif self.resize_mode == "random_crop":
            transform_list.extend([
                transforms.RandomResizedCrop(self.image_size),
            ])

        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        return transforms.Compose(transform_list)

    def __call__(self, image):
        """
        Preprocess an image.

        Args:
            image: PIL Image, numpy array, or tensor

        Returns:
            Preprocessed tensor [C, H, W]
        """
        return self.transform(image)

    def batch_process(self, images: List):
        """
        Process a batch of images.

        Args:
            images: List of images

        Returns:
            Batched tensor [B, C, H, W]
        """
        processed = [self(img) for img in images]
        return torch.stack(processed)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor for visualization.

        Args:
            tensor: Normalized tensor [C, H, W] or [B, C, H, W]

        Returns:
            Denormalized tensor
        """
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)

        if tensor.ndim == 4:  # Batch
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)

        return tensor * std + mean


class TextPreprocessor:
    """
    Text preprocessing for NLP models.

    Handles tokenization, padding, truncation, and special token handling.
    """

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
    ):
        """
        Args:
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            padding: Padding strategy ("max_length", "longest", "do_not_pad")
            truncation: Whether to truncate sequences
            return_tensors: Return format ("pt" for PyTorch, "np" for numpy)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers required: pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        """
        Preprocess text input.

        Args:
            text: Input text or list of texts
            text_pair: Optional second text for pair classification
            **kwargs: Additional tokenizer arguments

        Returns:
            Tokenized output with input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            text,
            text_pair=text_pair,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
            **kwargs
        )

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode

        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def batch_decode(self, token_ids_batch: Union[List[List[int]], torch.Tensor]) -> List[str]:
        """
        Decode a batch of token IDs.

        Args:
            token_ids_batch: Batch of token IDs

        Returns:
            List of decoded strings
        """
        if isinstance(token_ids_batch, torch.Tensor):
            token_ids_batch = token_ids_batch.tolist()
        return self.tokenizer.batch_decode(token_ids_batch, skip_special_tokens=True)


class AudioPreprocessor:
    """
    Audio preprocessing for speech and audio models.

    Handles sampling rate conversion, normalization, and feature extraction.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        normalize: bool = True,
    ):
        """
        Args:
            sample_rate: Target sampling rate
            n_mels: Number of mel filterbanks
            n_fft: FFT size
            hop_length: Hop length for STFT
            normalize: Whether to normalize audio
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize = normalize

    def __call__(self, audio: Union[torch.Tensor, Any], sample_rate: Optional[int] = None):
        """
        Preprocess audio input.

        Args:
            audio: Audio waveform (1D tensor or array)
            sample_rate: Original sample rate (if resampling needed)

        Returns:
            Preprocessed audio features
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for audio preprocessing")

        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        # Resample if needed
        if sample_rate is not None and sample_rate != self.sample_rate:
            try:
                import torchaudio
                audio = torchaudio.functional.resample(
                    audio,
                    orig_freq=sample_rate,
                    new_freq=self.sample_rate
                )
            except ImportError:
                warnings.warn("torchaudio not available, skipping resampling")

        # Normalize
        if self.normalize:
            audio = audio / (audio.abs().max() + 1e-8)

        return audio

    def extract_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract mel spectrogram features.

        Args:
            audio: Audio waveform [T]

        Returns:
            Mel spectrogram [n_mels, T']
        """
        try:
            import torchaudio
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            mel_spec = mel_transform(audio)
            return mel_spec
        except ImportError:
            raise ImportError("torchaudio required: pip install torchaudio")


def get_image_transforms(
    train: bool = True,
    image_size: int = 224,
    augment: bool = True,
) -> transforms.Compose:
    """
    Get standard image transforms for training or validation.

    Args:
        train: Whether for training (applies augmentation)
        image_size: Target image size
        augment: Whether to apply data augmentation

    Returns:
        Composed transforms

    Example:
        >>> train_transform = get_image_transforms(train=True, augment=True)
        >>> val_transform = get_image_transforms(train=False)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required: pip install torch torchvision")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_text_tokenizer(
    model_name: str = "bert-base-uncased",
    max_length: int = 512,
):
    """
    Get a text tokenizer with standard settings.

    Args:
        model_name: HuggingFace model name
        max_length: Maximum sequence length

    Returns:
        TextPreprocessor instance

    Example:
        >>> tokenizer = get_text_tokenizer("bert-base-uncased")
        >>> output = tokenizer("Hello world!")
    """
    return TextPreprocessor(
        tokenizer_name=model_name,
        max_length=max_length,
    )


def normalize_tensor(
    tensor: torch.Tensor,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Normalize a tensor.

    Args:
        tensor: Input tensor
        mean: Mean values (if None, computed from tensor)
        std: Std values (if None, computed from tensor)
        dim: Dimension to normalize over (if None, normalize entire tensor)

    Returns:
        Normalized tensor

    Example:
        >>> # Normalize to zero mean, unit variance
        >>> normalized = normalize_tensor(tensor)
        >>>
        >>> # Normalize with specific mean/std
        >>> normalized = normalize_tensor(tensor, mean=(0.5,), std=(0.5,))
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    if mean is None:
        if dim is not None:
            mean = tensor.mean(dim=dim, keepdim=True)
        else:
            mean = tensor.mean()

    if std is None:
        if dim is not None:
            std = tensor.std(dim=dim, keepdim=True)
        else:
            std = tensor.std()

    return (tensor - mean) / (std + 1e-8)


def denormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Denormalize image tensor for visualization.

    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        mean: Normalization mean used
        std: Normalization std used

    Returns:
        Denormalized tensor (values in [0, 1])

    Example:
        >>> # Denormalize and convert to PIL
        >>> denorm = denormalize_image(tensor)
        >>> pil_image = transforms.ToPILImage()(denorm)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.ndim == 4:  # Batch
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)


# Test function
def test_preprocessing():
    """Test preprocessing modules"""
    print("Testing Brain Preprocessing...")

    if TORCH_AVAILABLE:
        print("\n1. Testing Image Preprocessing...")
        try:
            from PIL import Image
            import numpy as np

            # Create dummy image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            preprocessor = ImagePreprocessor(image_size=224)
            output = preprocessor(img)
            print(f"   ✓ Image preprocessing: input PIL -> output {output.shape}")

            # Test denormalization
            denorm = preprocessor.denormalize(output)
            print(f"   ✓ Denormalization: {denorm.shape}")
        except Exception as e:
            print(f"   ✗ Image preprocessing failed: {e}")

    if TRANSFORMERS_AVAILABLE:
        print("\n2. Testing Text Preprocessing...")
        try:
            preprocessor = TextPreprocessor("bert-base-uncased", max_length=128)
            output = preprocessor("Hello world! This is a test.")
            print(f"   ✓ Text preprocessing: input_ids shape {output['input_ids'].shape}")

            # Test decoding
            decoded = preprocessor.decode(output['input_ids'][0])
            print(f"   ✓ Decoding: '{decoded}'")
        except Exception as e:
            print(f"   ✗ Text preprocessing failed: {e}")

    print("\n3. Testing Audio Preprocessing...")
    try:
        if TORCH_AVAILABLE:
            preprocessor = AudioPreprocessor(sample_rate=16000)
            audio = torch.randn(16000)  # 1 second of audio
            output = preprocessor(audio)
            print(f"   ✓ Audio preprocessing: {output.shape}")
        else:
            print("   - PyTorch not available, skipping")
    except Exception as e:
        print(f"   ✗ Audio preprocessing failed: {e}")

    print("\n✓ Preprocessing tests complete!")


if __name__ == "__main__":
    test_preprocessing()
