"""
Brain Architecture Base Classes

Unified interface for all Brain architectures.
Provides standard methods for training, inference, and state management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """Standard output format for all models"""
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    embeddings: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrainingConfig:
    """Standard training configuration"""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False


class BrainArchitecture(nn.Module, ABC):
    """
    Base class for all Brain architectures.

    All models in the Brain framework should inherit from this class
    and implement the required abstract methods.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize architecture.

        Args:
            config: Architecture-specific configuration
        """
        super().__init__()
        self.config = config
        self._device = "cpu"

    @abstractmethod
    def forward(self, *args, **kwargs) -> ModelOutput:
        """
        Forward pass of the model.

        Returns:
            ModelOutput containing predictions and optional metadata
        """
        pass

    def predict(self, inputs: Any, **kwargs) -> torch.Tensor:
        """
        Run inference on inputs.

        Args:
            inputs: Model inputs (format depends on architecture)
            **kwargs: Additional arguments

        Returns:
            Predictions tensor
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(inputs, **kwargs)
            return output.predictions if output.predictions is not None else output.logits

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        **kwargs
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Training batch
            optimizer: Optimizer
            **kwargs: Additional arguments

        Returns:
            Dictionary of metrics (loss, accuracy, etc.)
        """
        self.train()

        # Forward pass
        output = self.forward(**batch)

        # Backward pass
        if output.loss is not None:
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            return {"loss": output.loss.item()}
        else:
            raise ValueError("Model must return loss during training")

    def eval_step(
        self,
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, float]:
        """
        Single evaluation step.

        Args:
            batch: Evaluation batch
            **kwargs: Additional arguments

        Returns:
            Dictionary of metrics
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(**batch)

            metrics = {}
            if output.loss is not None:
                metrics["loss"] = output.loss.item()

            return metrics

    def save_pretrained(self, save_path: str):
        """
        Save model weights and configuration.

        Args:
            save_path: Path to save directory
        """
        import os
        os.makedirs(save_path, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_path, "model.pt"))

        # Save config if available
        if self.config is not None:
            import json
            with open(os.path.join(save_path, "config.json"), "w") as f:
                if hasattr(self.config, '__dict__'):
                    json.dump(self.config.__dict__, f, indent=2)
                else:
                    json.dump(self.config, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs):
        """
        Load model from saved weights.

        Args:
            load_path: Path to saved model directory
            **kwargs: Additional arguments for model initialization

        Returns:
            Loaded model instance
        """
        import os
        import json

        # Load config
        config_path = os.path.join(load_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            # Create config object (implementation specific)
            config = kwargs.get('config_class', dict)(config_dict) if kwargs.get('config_class') else config_dict
        else:
            config = None

        # Initialize model
        model = cls(config=config, **kwargs)

        # Load weights
        weights_path = os.path.join(load_path, "model.pt")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)

        return model

    def to_device(self, device: str):
        """
        Move model to device.

        Args:
            device: Target device ("cpu", "cuda", "cuda:0", etc.)
        """
        self._device = device
        return self.to(device)

    def get_device(self) -> str:
        """Get current device"""
        return self._device

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.

        Returns:
            Dictionary with total, trainable, and non-trainable parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": non_trainable_params,
        }

    def freeze(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_layers(self, layer_names: List[str]):
        """
        Freeze specific layers.

        Args:
            layer_names: List of layer names to freeze
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        params = self.count_parameters()

        return {
            "name": self.__class__.__name__,
            "parameters": params,
            "device": self.get_device(),
            "config": self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
        }


class VisionArchitecture(BrainArchitecture):
    """Base class for vision models"""

    def preprocess_image(self, image: Any) -> torch.Tensor:
        """
        Preprocess image input.

        Args:
            image: Input image (PIL, numpy, or tensor)

        Returns:
            Preprocessed tensor
        """
        # Use utils.data.ImagePreprocessor if available
        try:
            from utils.data import ImagePreprocessor
            preprocessor = ImagePreprocessor(image_size=224)
            return preprocessor(image)
        except ImportError:
            # Fallback: basic tensor conversion
            if not isinstance(image, torch.Tensor):
                import torchvision.transforms as transforms
                transform = transforms.ToTensor()
                return transform(image)
            return image


class LanguageArchitecture(BrainArchitecture):
    """Base class for language models"""

    def tokenize(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Tokenize text input.

        Args:
            text: Input text
            **kwargs: Tokenizer arguments

        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        # Use utils.data.TextPreprocessor if available
        try:
            from utils.data import TextPreprocessor
            preprocessor = TextPreprocessor()
            return preprocessor(text, **kwargs)
        except ImportError:
            raise NotImplementedError("Text preprocessing requires utils.data module")


class MultimodalArchitecture(BrainArchitecture):
    """Base class for multimodal models"""

    def process_inputs(
        self,
        image: Optional[Any] = None,
        text: Optional[str] = None,
        audio: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Process multimodal inputs.

        Args:
            image: Image input
            text: Text input
            audio: Audio input
            **kwargs: Additional inputs

        Returns:
            Dictionary of processed inputs
        """
        inputs = {}

        if image is not None:
            inputs["image"] = self.preprocess_image(image)

        if text is not None:
            inputs["text"] = self.tokenize(text)

        if audio is not None:
            inputs["audio"] = self.preprocess_audio(audio)

        return inputs


# Export all
__all__ = [
    'BrainArchitecture',
    'VisionArchitecture',
    'LanguageArchitecture',
    'MultimodalArchitecture',
    'ModelOutput',
    'TrainingConfig',
]
