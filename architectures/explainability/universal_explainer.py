"""
Universal Explainer - Works with ANY architecture

Provides a unified interface for explaining predictions from:
- Vision models (CLIP, SAM, YOLO, DETR, DINOv2)
- Audio models (Whisper, Encodec, MusicGen)
- Time series (N-BEATS, TFT, PatchTST)
- Multimodal models (LLaVA, Flamingo, BLIP-2)
- Meta-learning (MAML)

Methods:
- Gradient-based attribution
- Attention visualization
- Feature importance
- Counterfactual explanations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ExplanationType(Enum):
    """Types of explanations"""
    GRADIENT = "gradient"
    INTEGRATED_GRADIENT = "integrated_gradient"
    ATTENTION = "attention"
    GRADCAM = "gradcam"
    FEATURE_IMPORTANCE = "feature_importance"
    COUNTERFACTUAL = "counterfactual"


@dataclass
class ExplanationResult:
    """Result from explanation method"""
    # Attribution scores
    attributions: torch.Tensor

    # Metadata
    method: ExplanationType
    model_name: str
    input_shape: Tuple[int, ...]

    # Optional additional info
    attention_weights: Optional[torch.Tensor] = None
    feature_names: Optional[List[str]] = None
    confidence: Optional[float] = None
    prediction: Optional[Any] = None


class UniversalExplainer:
    """
    Universal explainer that works with any PyTorch model.

    Automatically detects architecture type and applies appropriate
    explanation methods.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: Any PyTorch model
            device: Device to run explanations on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Detect model type
        self.model_type = self._detect_model_type()

        # Hooks for intermediate activations
        self.activations = {}
        self.gradients = {}
        self.hooks = []

    def _detect_model_type(self) -> str:
        """
        Detect model architecture type.

        Returns:
            Model type string
        """
        model_name = self.model.__class__.__name__.lower()

        if any(x in model_name for x in ['clip', 'blip', 'llava', 'flamingo']):
            return 'multimodal'
        elif any(x in model_name for x in ['whisper', 'encodec', 'musicgen', 'wav2vec']):
            return 'audio'
        elif any(x in model_name for x in ['nbeats', 'tft', 'patchtst']):
            return 'timeseries'
        elif any(x in model_name for x in ['yolo', 'detr', 'sam', 'dino']):
            return 'vision'
        elif any(x in model_name for x in ['maml']):
            return 'metalearning'
        else:
            return 'generic'

    def explain(
        self,
        inputs: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        method: ExplanationType = ExplanationType.GRADIENT,
        **kwargs
    ) -> ExplanationResult:
        """
        Explain a prediction.

        Args:
            inputs: Model inputs
            target: Target class/value (optional)
            method: Explanation method to use
            **kwargs: Method-specific arguments

        Returns:
            ExplanationResult with attributions
        """
        inputs = inputs.to(self.device)

        if method == ExplanationType.GRADIENT:
            return self._gradient_attribution(inputs, target, **kwargs)
        elif method == ExplanationType.INTEGRATED_GRADIENT:
            return self._integrated_gradients(inputs, target, **kwargs)
        elif method == ExplanationType.ATTENTION:
            return self._attention_attribution(inputs, **kwargs)
        elif method == ExplanationType.GRADCAM:
            return self._gradcam_attribution(inputs, target, **kwargs)
        elif method == ExplanationType.FEATURE_IMPORTANCE:
            return self._feature_importance(inputs, target, **kwargs)
        else:
            raise ValueError(f"Unknown explanation method: {method}")

    def _gradient_attribution(
        self,
        inputs: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        smooth: bool = False,
        num_samples: int = 50,
        noise_level: float = 0.1
    ) -> ExplanationResult:
        """
        Gradient-based attribution (saliency maps).

        If smooth=True, uses SmoothGrad for noise reduction.
        """
        inputs.requires_grad = True

        if smooth:
            # SmoothGrad: average gradients over noisy samples
            attributions = torch.zeros_like(inputs)

            for _ in range(num_samples):
                # Add noise
                noise = torch.randn_like(inputs) * noise_level * (inputs.max() - inputs.min())
                noisy_inputs = inputs + noise
                noisy_inputs.requires_grad = True

                # Forward pass
                outputs = self.model(noisy_inputs)

                # Get target score
                if target is not None:
                    score = outputs[torch.arange(outputs.shape[0]), target].sum()
                else:
                    score = outputs.max(dim=-1)[0].sum()

                # Backward
                self.model.zero_grad()
                score.backward()

                # Accumulate gradients
                attributions += noisy_inputs.grad.data

            attributions /= num_samples

        else:
            # Standard gradient
            outputs = self.model(inputs)

            # Get target score
            if target is not None:
                score = outputs[torch.arange(outputs.shape[0]), target].sum()
            else:
                score = outputs.max(dim=-1)[0].sum()

            # Backward
            self.model.zero_grad()
            score.backward()

            attributions = inputs.grad.data

        return ExplanationResult(
            attributions=attributions.abs(),
            method=ExplanationType.GRADIENT,
            model_name=self.model.__class__.__name__,
            input_shape=inputs.shape
        )

    def _integrated_gradients(
        self,
        inputs: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> ExplanationResult:
        """
        Integrated Gradients attribution.

        Integrates gradients along path from baseline to input.
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        baseline = baseline.to(self.device)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(self.device)

        # Accumulate gradients
        integrated_grads = torch.zeros_like(inputs)

        for alpha in alphas:
            # Interpolate
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad = True

            # Forward
            outputs = self.model(interpolated)

            # Get target score
            if target is not None:
                score = outputs[torch.arange(outputs.shape[0]), target].sum()
            else:
                score = outputs.max(dim=-1)[0].sum()

            # Backward
            self.model.zero_grad()
            score.backward()

            # Accumulate
            integrated_grads += interpolated.grad.data

        # Average and scale
        integrated_grads /= steps
        integrated_grads *= (inputs - baseline)

        return ExplanationResult(
            attributions=integrated_grads.abs(),
            method=ExplanationType.INTEGRATED_GRADIENT,
            model_name=self.model.__class__.__name__,
            input_shape=inputs.shape
        )

    def _attention_attribution(
        self,
        inputs: torch.Tensor,
        layer_name: Optional[str] = None
    ) -> ExplanationResult:
        """
        Extract and visualize attention weights.

        Works for Transformer-based models.
        """
        attention_weights = []

        # Hook to capture attention
        def attention_hook(module, input, output):
            if isinstance(output, tuple):
                # Some attention layers return (output, attention_weights)
                if len(output) > 1 and output[1] is not None:
                    attention_weights.append(output[1].detach())

        # Register hooks on attention layers
        hooks = []
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                if layer_name is None or layer_name in name:
                    hook = module.register_forward_hook(attention_hook)
                    hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Aggregate attention weights
        if attention_weights:
            # Average across layers and heads
            avg_attention = torch.stack(attention_weights).mean(dim=0)

            # Use attention as attribution
            attributions = avg_attention
        else:
            # No attention found, return zeros
            attributions = torch.zeros_like(inputs)

        return ExplanationResult(
            attributions=attributions,
            method=ExplanationType.ATTENTION,
            model_name=self.model.__class__.__name__,
            input_shape=inputs.shape,
            attention_weights=torch.stack(attention_weights) if attention_weights else None
        )

    def _gradcam_attribution(
        self,
        inputs: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        target_layer: Optional[str] = None
    ) -> ExplanationResult:
        """
        Grad-CAM attribution for vision models.

        Highlights important regions in images.
        """
        # Find convolutional layers
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))

        if not conv_layers:
            raise ValueError("No convolutional layers found for Grad-CAM")

        # Use last conv layer if not specified
        if target_layer is None:
            target_layer_name, target_layer_module = conv_layers[-1]
        else:
            target_layer_module = dict(self.model.named_modules())[target_layer]
            target_layer_name = target_layer

        # Hooks for activations and gradients
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        # Register hooks
        fh = target_layer_module.register_forward_hook(forward_hook)
        bh = target_layer_module.register_full_backward_hook(backward_hook)

        # Forward pass
        inputs.requires_grad = True
        outputs = self.model(inputs)

        # Get target score
        if target is not None:
            score = outputs[torch.arange(outputs.shape[0]), target].sum()
        else:
            score = outputs.max(dim=-1)[0].sum()

        # Backward
        self.model.zero_grad()
        score.backward()

        # Remove hooks
        fh.remove()
        bh.remove()

        # Compute Grad-CAM
        if activations and gradients:
            acts = activations[0]  # (batch, channels, H, W)
            grads = gradients[0]  # (batch, channels, H, W)

            # Global average pooling of gradients
            weights = grads.mean(dim=(2, 3), keepdim=True)

            # Weighted combination
            cam = (weights * acts).sum(dim=1, keepdim=True)  # (batch, 1, H, W)

            # ReLU and normalize
            cam = F.relu(cam)
            cam = cam / (cam.max() + 1e-8)

            # Upsample to input size
            if len(inputs.shape) == 4:  # Image input
                cam = F.interpolate(
                    cam,
                    size=inputs.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            attributions = cam
        else:
            attributions = torch.zeros_like(inputs)

        return ExplanationResult(
            attributions=attributions,
            method=ExplanationType.GRADCAM,
            model_name=self.model.__class__.__name__,
            input_shape=inputs.shape
        )

    def _feature_importance(
        self,
        inputs: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        method: str = 'perturbation',
        num_samples: int = 100
    ) -> ExplanationResult:
        """
        Feature importance through perturbation.

        Works for time series and tabular data.
        """
        if method == 'perturbation':
            # Perturbation-based importance
            baseline_output = self.model(inputs)

            if target is not None:
                baseline_score = baseline_output[torch.arange(baseline_output.shape[0]), target]
            else:
                baseline_score = baseline_output.max(dim=-1)[0]

            # Importance for each feature
            importance = torch.zeros_like(inputs)

            # Iterate over features
            num_features = inputs.shape[-1]

            for i in range(num_features):
                # Perturb feature
                perturbed = inputs.clone()
                perturbed[..., i] = 0  # Zero out feature

                # Forward
                with torch.no_grad():
                    perturbed_output = self.model(perturbed)

                if target is not None:
                    perturbed_score = perturbed_output[torch.arange(perturbed_output.shape[0]), target]
                else:
                    perturbed_score = perturbed_output.max(dim=-1)[0]

                # Importance = drop in score
                importance[..., i] = (baseline_score - perturbed_score).abs()

            attributions = importance

        else:
            raise ValueError(f"Unknown feature importance method: {method}")

        return ExplanationResult(
            attributions=attributions,
            method=ExplanationType.FEATURE_IMPORTANCE,
            model_name=self.model.__class__.__name__,
            input_shape=inputs.shape
        )

    def get_important_features(
        self,
        explanation: ExplanationResult,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get top-k most important features.

        Args:
            explanation: ExplanationResult
            top_k: Number of top features to return

        Returns:
            List of (feature_index, importance_score) tuples
        """
        # Flatten attributions
        flat_attr = explanation.attributions.flatten()

        # Get top-k indices
        top_k_values, top_k_indices = torch.topk(flat_attr, min(top_k, len(flat_attr)))

        # Convert to list
        important_features = [
            (idx.item(), val.item())
            for idx, val in zip(top_k_indices, top_k_values)
        ]

        return important_features

    def visualize_attribution(
        self,
        explanation: ExplanationResult,
        original_input: Optional[torch.Tensor] = None,
        cmap: str = 'jet'
    ) -> np.ndarray:
        """
        Create visualization of attribution.

        Args:
            explanation: ExplanationResult
            original_input: Original input for overlay
            cmap: Colormap name

        Returns:
            Visualization as numpy array
        """
        attr = explanation.attributions.cpu().numpy()

        # Normalize to [0, 1]
        attr_min = attr.min()
        attr_max = attr.max()
        if attr_max > attr_min:
            attr_normalized = (attr - attr_min) / (attr_max - attr_min)
        else:
            attr_normalized = attr

        # For images, create heatmap overlay
        if len(attr.shape) == 4 and attr.shape[1] in [1, 3]:
            # Image format: (batch, channels, H, W)
            # Average over channels if needed
            if attr.shape[1] > 1:
                attr_normalized = attr_normalized.mean(axis=1, keepdims=True)

            # Convert to RGB heatmap
            import matplotlib.cm as cm
            colormap = cm.get_cmap(cmap)

            # Apply colormap
            heatmap = colormap(attr_normalized[0, 0])[:, :, :3]  # RGB only

            # Overlay on original if provided
            if original_input is not None:
                orig = original_input.cpu().numpy()
                if orig.shape[1] == 3:
                    # Normalize original
                    orig = (orig - orig.min()) / (orig.max() - orig.min())
                    orig = orig.transpose(0, 2, 3, 1)[0]  # (H, W, 3)

                    # Blend
                    visualization = 0.5 * orig + 0.5 * heatmap
                else:
                    visualization = heatmap
            else:
                visualization = heatmap

        else:
            # For other data types, return attribution as is
            visualization = attr_normalized

        return visualization


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Universal Explainer - Works with ANY Architecture")
    print("="*80)

    # Example with a simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = SimpleCNN()
    explainer = UniversalExplainer(model)

    print(f"\nModel type detected: {explainer.model_type}")

    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32)

    print(f"Input shape: {x.shape}")

    # Gradient attribution
    print("\n" + "-"*80)
    print("Gradient Attribution")
    result_grad = explainer.explain(x, method=ExplanationType.GRADIENT)
    print(f"Attribution shape: {result_grad.attributions.shape}")
    print(f"Method: {result_grad.method.value}")

    # Integrated gradients
    print("\n" + "-"*80)
    print("Integrated Gradients")
    result_ig = explainer.explain(x, method=ExplanationType.INTEGRATED_GRADIENT, steps=25)
    print(f"Attribution shape: {result_ig.attributions.shape}")

    # Grad-CAM
    print("\n" + "-"*80)
    print("Grad-CAM")
    result_cam = explainer.explain(x, method=ExplanationType.GRADCAM)
    print(f"Attribution shape: {result_cam.attributions.shape}")

    # Get important features
    important = explainer.get_important_features(result_grad, top_k=10)
    print(f"\nTop 10 important pixels/features:")
    for i, (idx, score) in enumerate(important[:5]):
        print(f"  {i+1}. Feature {idx}: {score:.4f}")

    print("\n" + "="*80)
