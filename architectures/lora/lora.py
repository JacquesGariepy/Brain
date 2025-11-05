"""
LoRA - Low-Rank Adaptation for Parameter-Efficient Fine-Tuning

Enables fine-tuning large models with <1% trainable parameters.

Key Techniques:
- LoRA: Low-rank matrices added to frozen weights
- QLoRA: LoRA with 4-bit quantized base model
- AdaLoRA: Adaptive rank allocation
- DoRA: Weight-decomposed LoRA

References:
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- AdaLoRA: https://arxiv.org/abs/2303.10512
- DoRA: https://arxiv.org/abs/2402.09353
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import math


# ============================================================================
# Base LoRA Configuration
# ============================================================================

@dataclass
class LoRAConfig:
    """Configuration for LoRA"""
    # Rank
    r: int = 8  # LoRA rank (typically 8-64)

    # Alpha (scaling factor)
    lora_alpha: int = 16  # Scaling = alpha / r

    # Dropout
    lora_dropout: float = 0.0

    # Target modules
    target_modules: List[str] = None  # e.g., ["q_proj", "v_proj"]

    # Initialization
    init_lora_weights: str = "default"  # "default", "gaussian", "zeros"

    # Bias
    bias: str = "none"  # "none", "all", "lora_only"

    def __post_init__(self):
        if self.target_modules is None:
            # Default: query and value projections
            self.target_modules = ["q_proj", "v_proj"]


# ============================================================================
# LoRA Linear Layer
# ============================================================================

class LoRALinear(nn.Module):
    """
    LoRA-adapted Linear Layer

    Instead of fine-tuning W, learns low-rank decomposition:
    W' = W + (B @ A) * (alpha / r)

    Where:
    - W is frozen pretrained weight [out_features, in_features]
    - A is trainable [r, in_features]
    - B is trainable [out_features, r]
    - Only A and B are updated during fine-tuning!

    Example:
        >>> # Replace nn.Linear with LoRALinear
        >>> config = LoRAConfig(r=8, lora_alpha=16)
        >>> layer = LoRALinear(4096, 4096, config=config)
        >>>
        >>> # Only 0.2% of parameters are trainable!
        >>> trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        >>> total = sum(p.numel() for p in layer.parameters())
        >>> print(f"{trainable / total * 100:.2f}% trainable")  # 0.2%
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Frozen pretrained weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(config.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.r))

        # Scaling factor
        self.scaling = config.lora_alpha / config.r

        # Dropout
        if config.lora_dropout > 0:
            self.lora_dropout = nn.Dropout(config.lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            if config.bias == "none":
                self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)

        # Initialize
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """Initialize LoRA parameters"""
        if self.config.init_lora_weights == "default":
            # Kaiming uniform for A, zeros for B (like original paper)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        elif self.config.init_lora_weights == "gaussian":
            nn.init.normal_(self.lora_A, std=0.02)
            nn.init.zeros_(self.lora_B)
        elif self.config.init_lora_weights == "zeros":
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = (W + BA * scaling) @ x

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Base output from frozen weight
        result = F.linear(x, self.weight, self.bias)

        # LoRA adaptation
        lora_out = self.lora_dropout(x) @ self.lora_A.t()  # [..., r]
        lora_out = lora_out @ self.lora_B.t()  # [..., out_features]
        result = result + lora_out * self.scaling

        return result

    def merge_weights(self):
        """Merge LoRA weights into base weight (for inference)"""
        if not self.weight.requires_grad:  # Only if frozen
            # W' = W + BA * scaling
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data += delta_w

    def unmerge_weights(self):
        """Unmerge LoRA weights from base weight"""
        if not self.weight.requires_grad:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data -= delta_w


# ============================================================================
# QLoRA - Quantized LoRA
# ============================================================================

@dataclass
class QLoRAConfig(LoRAConfig):
    """Configuration for QLoRA"""
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"


class QLoRALinear(nn.Module):
    """
    QLoRA - LoRA with 4-bit Quantized Base Model

    Key innovation: Base model in 4-bit (NF4), LoRA adapters in FP16.
    Enables fine-tuning 65B models on a single 48GB GPU!

    Memory savings:
    - 7B model: 14GB → 5GB (65% reduction)
    - 13B model: 26GB → 9GB (65% reduction)
    - 65B model: 130GB → 48GB (63% reduction)

    Example:
        >>> config = QLoRAConfig(r=8, load_in_4bit=True)
        >>> layer = QLoRALinear(4096, 4096, config=config)
        >>> # Base model uses 1/4 the memory, adapters are tiny!
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QLoRAConfig,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Quantized base weight (4-bit NF4)
        self.register_buffer('weight_quant', torch.zeros(
            out_features * in_features,
            dtype=torch.uint8
        ))
        self.register_buffer('weight_scale', torch.zeros(1, dtype=torch.float16))

        # LoRA adapters (FP16)
        self.lora_A = nn.Parameter(torch.zeros(
            config.r, in_features,
            dtype=config.bnb_4bit_compute_dtype
        ))
        self.lora_B = nn.Parameter(torch.zeros(
            out_features, config.r,
            dtype=config.bnb_4bit_compute_dtype
        ))

        self.scaling = config.lora_alpha / config.r

        if config.lora_dropout > 0:
            self.lora_dropout = nn.Dropout(config.lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        if bias:
            self.bias = nn.Parameter(torch.zeros(
                out_features,
                dtype=config.bnb_4bit_compute_dtype
            ))
            if config.bias == "none":
                self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)

        # Initialize LoRA
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def quantize_weight(self, weight: torch.Tensor):
        """Quantize base weight to 4-bit NF4"""
        from architectures.quantization import NF4Quantizer

        # Quantize to NF4
        indices, scale = NF4Quantizer.quantize(weight)

        # Store
        self.weight_quant.copy_(indices.flatten())
        self.weight_scale.copy_(scale)

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize weight from NF4 to compute dtype"""
        from architectures.quantization import NF4Quantizer

        # Reshape indices
        indices = self.weight_quant.reshape(self.out_features, self.in_features)

        # Dequantize
        weight = NF4Quantizer.dequantize(indices, self.weight_scale)

        return weight.to(self.config.bnb_4bit_compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with 4-bit base + FP16 adapters"""
        # Dequantize base weight on the fly
        weight = self.dequantize_weight()

        # Base output
        result = F.linear(x, weight, self.bias)

        # LoRA adaptation
        lora_out = self.lora_dropout(x) @ self.lora_A.t()
        lora_out = lora_out @ self.lora_B.t()
        result = result + lora_out * self.scaling

        return result


# ============================================================================
# AdaLoRA - Adaptive LoRA
# ============================================================================

@dataclass
class AdaLoRAConfig(LoRAConfig):
    """Configuration for AdaLoRA"""
    # Adaptive rank
    target_r: int = 8  # Target average rank
    init_r: int = 12  # Initial rank (will be pruned to target_r)
    tinit: int = 0  # Warmup steps before pruning
    tfinal: int = 0  # Final step for pruning
    deltaT: int = 1  # Pruning frequency

    # Importance scoring
    orth_reg_weight: float = 0.5  # Orthogonality regularization


class AdaLoRALinear(nn.Module):
    """
    AdaLoRA - Adaptive Low-Rank Adaptation

    Dynamically allocates rank to important modules/singular values.
    Key insight: Not all parameters are equally important!

    Features:
    - Starts with high rank, prunes unimportant singular values
    - SVD-based importance scoring
    - Orthogonality regularization

    Example:
        >>> config = AdaLoRAConfig(init_r=12, target_r=8)
        >>> layer = AdaLoRALinear(4096, 4096, config=config)
        >>> # During training, rank automatically reduces from 12→8
        >>> # More rank allocated to important layers
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: AdaLoRAConfig,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Frozen base weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # SVD-based parameterization: W = U @ S @ V^T
        # Where S is diagonal matrix of singular values
        self.lora_E = nn.Parameter(torch.zeros(
            out_features, config.init_r
        ))  # U (left singular vectors)
        self.lora_A = nn.Parameter(torch.zeros(
            config.init_r, config.init_r
        ))  # S (singular values as diagonal)
        self.lora_B = nn.Parameter(torch.zeros(
            config.init_r, in_features
        ))  # V^T (right singular vectors)

        self.scaling = config.lora_alpha / config.target_r

        # Rank mask (which singular values are active)
        self.register_buffer('rank_mask', torch.ones(config.init_r, dtype=torch.bool))

        # Importance scores
        self.register_buffer('importance', torch.zeros(config.init_r))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            if config.bias == "none":
                self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_E, a=math.sqrt(5))
        nn.init.zeros_(self.lora_A)
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with adaptive rank"""
        # Base output
        result = F.linear(x, self.weight, self.bias)

        # AdaLoRA: y = E @ A @ B @ x
        # Apply rank mask
        E_masked = self.lora_E * self.rank_mask.unsqueeze(0).float()
        A_masked = self.lora_A * self.rank_mask.unsqueeze(0).float() * self.rank_mask.unsqueeze(1).float()
        B_masked = self.lora_B * self.rank_mask.unsqueeze(1).float()

        # Compute adaptation
        lora_out = x @ B_masked.t()  # [..., init_r]
        lora_out = lora_out @ A_masked.t()  # [..., init_r]
        lora_out = lora_out @ E_masked.t()  # [..., out_features]

        result = result + lora_out * self.scaling

        return result

    def compute_importance(self):
        """Compute importance scores for singular values"""
        # Importance = magnitude of singular values * gradient magnitude
        with torch.no_grad():
            # Diagonal of A represents singular values
            sv_magnitude = torch.diag(self.lora_A.data).abs()

            # Gradient magnitude
            if self.lora_A.grad is not None:
                sv_gradient = torch.diag(self.lora_A.grad).abs()
            else:
                sv_gradient = torch.zeros_like(sv_magnitude)

            # Combined importance
            self.importance = sv_magnitude * sv_gradient

    def update_rank_mask(self, target_r: int):
        """Update rank mask based on importance"""
        # Keep top target_r singular values
        _, indices = torch.topk(self.importance, target_r)

        # Update mask
        self.rank_mask.fill_(False)
        self.rank_mask[indices] = True

    def get_orthogonality_loss(self) -> torch.Tensor:
        """
        Orthogonality regularization.

        Encourages E and B to be orthonormal for better singular value interpretation.
        """
        # E should have orthonormal columns
        E_orth = self.lora_E.t() @ self.lora_E
        E_orth_loss = (E_orth - torch.eye(
            self.config.init_r,
            device=E_orth.device
        )).pow(2).mean()

        # B should have orthonormal rows
        B_orth = self.lora_B @ self.lora_B.t()
        B_orth_loss = (B_orth - torch.eye(
            self.config.init_r,
            device=B_orth.device
        )).pow(2).mean()

        return (E_orth_loss + B_orth_loss) * self.config.orth_reg_weight


# ============================================================================
# DoRA - Weight-Decomposed LoRA
# ============================================================================

@dataclass
class DoRAConfig(LoRAConfig):
    """Configuration for DoRA"""
    # DoRA decomposes weight into magnitude and direction
    # W' = m * (W + BA) / ||W + BA||
    use_dora: bool = True


class DoRALinear(nn.Module):
    """
    DoRA - Weight-Decomposed Low-Rank Adaptation

    Decomposes weight into:
    - Magnitude: ||W||
    - Direction: W / ||W||

    Key insight: Fine-tuning primarily changes direction, not magnitude.
    DoRA learns magnitude separately for better performance.

    W' = m * (W + BA) / ||W + BA||

    Where:
    - m is learnable magnitude
    - (W + BA) / ||W + BA|| is direction

    Performance: DoRA > LoRA on most tasks with same rank!

    Example:
        >>> config = DoRAConfig(r=8, use_dora=True)
        >>> layer = DoRALinear(4096, 4096, config=config)
        >>> # Outperforms standard LoRA with similar parameters
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: DoRAConfig,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Frozen base weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(config.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.r))

        # Magnitude (per output feature)
        self.magnitude = nn.Parameter(torch.ones(out_features, 1))

        self.scaling = config.lora_alpha / config.r

        if config.lora_dropout > 0:
            self.lora_dropout = nn.Dropout(config.lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            if config.bias == "none":
                self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Initialize magnitude as norm of base weight
        with torch.no_grad():
            self.magnitude.data = torch.norm(
                self.weight.data,
                p=2,
                dim=1,
                keepdim=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with weight decomposition"""
        if not self.config.use_dora:
            # Standard LoRA
            result = F.linear(x, self.weight, self.bias)
            lora_out = self.lora_dropout(x) @ self.lora_A.t()
            lora_out = lora_out @ self.lora_B.t()
            result = result + lora_out * self.scaling
            return result

        # DoRA: W' = m * (W + BA) / ||W + BA||
        # Compute directional weight
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        directional_weight = self.weight + delta_w

        # Normalize direction (per output feature)
        weight_norm = torch.norm(directional_weight, p=2, dim=1, keepdim=True)
        weight_norm = weight_norm.clamp(min=1e-8)  # Avoid division by zero

        directional_weight = directional_weight / weight_norm

        # Apply magnitude
        final_weight = self.magnitude * directional_weight

        # Forward pass
        result = F.linear(x, final_weight, self.bias)

        return result


# ============================================================================
# LoRA Model Wrapper
# ============================================================================

class LoRAModel(nn.Module):
    """
    Wrapper to apply LoRA to any model.

    Automatically replaces target modules with LoRA versions.

    Example:
        >>> # Apply LoRA to pretrained model
        >>> model = MyTransformer()
        >>> config = LoRAConfig(r=8, target_modules=["q_proj", "v_proj"])
        >>> lora_model = LoRAModel(model, config)
        >>>
        >>> # Only 0.1% of parameters are trainable!
        >>> trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        >>> total = sum(p.numel() for p in lora_model.parameters())
        >>> print(f"{trainable / total * 100:.2f}% trainable")
    """

    def __init__(
        self,
        model: nn.Module,
        config: LoRAConfig,
        lora_type: str = "standard"  # "standard", "qlora", "adalora", "dora"
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.lora_type = lora_type

        # Apply LoRA to target modules
        self._apply_lora()

    def _apply_lora(self):
        """Replace target modules with LoRA versions"""
        for name, module in self.model.named_modules():
            # Check if this module should be replaced
            if self._is_target_module(name, module):
                self._replace_module(name, module)

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Check if module should be replaced with LoRA"""
        # Check if module name matches target patterns
        for target in self.config.target_modules:
            if target in name:
                return isinstance(module, nn.Linear)
        return False

    def _replace_module(self, name: str, module: nn.Linear):
        """Replace module with LoRA version"""
        # Get module path
        path_parts = name.split('.')
        parent = self.model

        for part in path_parts[:-1]:
            parent = getattr(parent, part)

        # Create LoRA module
        if self.lora_type == "standard":
            lora_module = LoRALinear(
                module.in_features,
                module.out_features,
                self.config,
                bias=module.bias is not None
            )
        elif self.lora_type == "qlora":
            lora_module = QLoRALinear(
                module.in_features,
                module.out_features,
                self.config,
                bias=module.bias is not None
            )
            # Quantize base weight
            lora_module.quantize_weight(module.weight.data)
        elif self.lora_type == "adalora":
            lora_module = AdaLoRALinear(
                module.in_features,
                module.out_features,
                self.config,
                bias=module.bias is not None
            )
        elif self.lora_type == "dora":
            lora_module = DoRALinear(
                module.in_features,
                module.out_features,
                self.config,
                bias=module.bias is not None
            )
        else:
            raise ValueError(f"Unknown LoRA type: {self.lora_type}")

        # Copy weight and bias
        if hasattr(lora_module, 'weight'):
            lora_module.weight.data.copy_(module.weight.data)

        if module.bias is not None and hasattr(lora_module, 'bias'):
            lora_module.bias.data.copy_(module.bias.data)

        # Replace module
        setattr(parent, path_parts[-1], lora_module)

    def forward(self, *args, **kwargs):
        """Forward pass through model"""
        return self.model(*args, **kwargs)

    def merge_and_unload(self):
        """Merge LoRA weights and return base model"""
        for module in self.model.modules():
            if isinstance(module, (LoRALinear, DoRALinear)):
                module.merge_weights()

        return self.model

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get only LoRA parameters (for optimizer)"""
        lora_params = []
        for module in self.model.modules():
            if isinstance(module, (LoRALinear, QLoRALinear, AdaLoRALinear, DoRALinear)):
                lora_params.extend([
                    p for p in module.parameters() if p.requires_grad
                ])
        return lora_params


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LoRA - Low-Rank Adaptation for Parameter-Efficient Fine-Tuning")
    print("=" * 80)

    # Standard LoRA Example
    print("\n" + "=" * 80)
    print("Standard LoRA")
    print("=" * 80)

    config = LoRAConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    layer = LoRALinear(4096, 4096, config=config)

    x = torch.randn(2, 10, 4096)
    out = layer(x)

    # Count parameters
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in layer.parameters())

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable ratio: {trainable / total * 100:.2f}%")
    print(f"Memory savings: {(1 - trainable / total) * 100:.1f}%")
    print(f"Output shape: {out.shape}")

    # QLoRA Example
    print("\n" + "=" * 80)
    print("QLoRA - 4-bit Base + LoRA Adapters")
    print("=" * 80)

    qlora_config = QLoRAConfig(r=8, lora_alpha=16, load_in_4bit=True)
    qlora_layer = QLoRALinear(4096, 4096, config=qlora_config)

    # Simulate quantization
    weight = torch.randn(4096, 4096)
    qlora_layer.quantize_weight(weight)

    out = qlora_layer(x)

    base_size = 4096 * 4096 * 2  # FP16
    quant_size = 4096 * 4096 * 0.5  # 4-bit
    adapter_size = trainable * 2  # FP16 adapters

    print(f"Base model (FP16): {base_size / 1024 / 1024:.1f} MB")
    print(f"Quantized base (4-bit): {quant_size / 1024 / 1024:.1f} MB")
    print(f"LoRA adapters (FP16): {adapter_size / 1024 / 1024:.1f} MB")
    print(f"Total: {(quant_size + adapter_size) / 1024 / 1024:.1f} MB")
    print(f"Memory reduction: {(1 - (quant_size + adapter_size) / base_size) * 100:.1f}%")

    # AdaLoRA Example
    print("\n" + "=" * 80)
    print("AdaLoRA - Adaptive Rank Allocation")
    print("=" * 80)

    adalora_config = AdaLoRAConfig(init_r=12, target_r=8, lora_alpha=16)
    adalora_layer = AdaLoRALinear(4096, 4096, config=adalora_config)

    out = adalora_layer(x)

    print(f"Initial rank: {adalora_config.init_r}")
    print(f"Target rank: {adalora_config.target_r}")
    print(f"Active rank: {adalora_layer.rank_mask.sum().item()}")
    print("Rank will be pruned during training based on importance")

    # Simulate importance update
    adalora_layer.compute_importance()
    print(f"Importance scores: {adalora_layer.importance[:5].tolist()}")

    # DoRA Example
    print("\n" + "=" * 80)
    print("DoRA - Weight-Decomposed LoRA")
    print("=" * 80)

    dora_config = DoRAConfig(r=8, lora_alpha=16, use_dora=True)
    dora_layer = DoRALinear(4096, 4096, config=dora_config)

    out = dora_layer(x)

    print(f"Trainable parameters: {sum(p.numel() for p in dora_layer.parameters() if p.requires_grad):,}")
    print("DoRA decomposes weight into magnitude and direction")
    print("Typically outperforms standard LoRA with similar parameter count")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
LoRA Variants Comparison:

1. Standard LoRA:
   - Memory: ~0.1-0.5% trainable parameters
   - Quality: Good, minimal degradation vs full fine-tuning
   - Speed: Fast training (only update small matrices)
   - Use: General purpose, most tasks

2. QLoRA:
   - Memory: 4x reduction (4-bit base) + tiny adapters
   - Quality: Matches full fine-tuning
   - Speed: Slightly slower (quantization overhead)
   - Use: When GPU memory is limited (fine-tune 65B on 48GB!)

3. AdaLoRA:
   - Memory: Similar to LoRA but adapts rank
   - Quality: Better than LoRA (allocates rank optimally)
   - Speed: Slightly slower (importance computation)
   - Use: When you want best quality with limited budget

4. DoRA:
   - Memory: Slightly more than LoRA (magnitude parameter)
   - Quality: Best among LoRA variants
   - Speed: Similar to LoRA
   - Use: When you want maximum quality with LoRA efficiency

Memory Savings Examples:
- 7B model with LoRA (r=8): 14GB → 0.1GB trainable (99.3% reduction)
- 7B model with QLoRA: 14GB → 5GB total (65% reduction)
- 65B model with QLoRA: 130GB → 48GB (63% reduction)

Recommendation:
- Default: QLoRA (best memory/quality trade-off)
- Maximum quality: DoRA
- Adaptive budget: AdaLoRA
- Simple & fast: Standard LoRA
""")

    print("=" * 80)
