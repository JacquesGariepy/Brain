"""
Quantization - Model Compression for Efficient Inference

Reduces model size and memory by 4-8x with minimal quality loss.

Key Techniques:
- GPTQ: Post-training quantization with optimal brain surgeon
- AWQ: Activation-aware weight quantization
- bitsandbytes: 8-bit and 4-bit quantization
- GGML/GGUF: llama.cpp compatible format

References:
- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
- LLM.int8(): https://arxiv.org/abs/2208.07339
- GGML: https://github.com/ggerganov/ggml
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod


# ============================================================================
# Quantization Base Classes
# ============================================================================

class QuantizationType(Enum):
    """Types of quantization"""
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"  # 4-bit NormalFloat (QLoRA)
    FP8 = "fp8"


@dataclass
class QuantizationConfig:
    """Base configuration for quantization"""
    quant_type: QuantizationType = QuantizationType.INT8
    symmetric: bool = True  # Symmetric vs asymmetric
    per_channel: bool = True  # Per-channel vs per-tensor
    calibration_samples: int = 128

    # Outlier handling
    outlier_threshold: Optional[float] = None


class QuantizedLinear(nn.Module, ABC):
    """Base class for quantized linear layers"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.quant_config = quant_config or QuantizationConfig()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        pass

    @abstractmethod
    def quantize_weights(self, weight: torch.Tensor):
        """Quantize weights"""
        pass


# ============================================================================
# GPTQ - Optimal Brain Surgeon Quantization
# ============================================================================

@dataclass
class GPTQConfig:
    """Configuration for GPTQ quantization"""
    bits: int = 4  # 2, 3, 4, or 8 bits
    group_size: int = 128  # Quantization group size
    desc_act: bool = False  # Act order (reduces error but slower)
    sym: bool = True  # Symmetric quantization
    damp_percent: float = 0.01  # Dampening for Hessian

    # Calibration
    calibration_samples: int = 128
    batch_size: int = 1


class GPTQLinear(QuantizedLinear):
    """
    GPTQ-quantized Linear Layer

    Uses optimal brain surgeon to find optimal quantization.
    Minimizes (WX - QX)^2 where Q is quantized weight.

    Example:
        >>> config = GPTQConfig(bits=4, group_size=128)
        >>> layer = GPTQLinear(4096, 4096, quant_config=config)
        >>> layer.quantize_weights(original_weight)
        >>> output = layer(input)  # 4x smaller, 4x faster
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_config: Optional[GPTQConfig] = None
    ):
        super().__init__(in_features, out_features, bias)
        self.quant_config = quant_config or GPTQConfig()

        # Quantized weights stored as int
        self.register_buffer('qweight', torch.zeros(
            (out_features, in_features // (32 // self.quant_config.bits)),
            dtype=torch.int32
        ))

        # Scales for dequantization
        num_groups = in_features // self.quant_config.group_size
        self.register_buffer('scales', torch.zeros(
            (out_features, num_groups),
            dtype=torch.float16
        ))

        # Zero points (if asymmetric)
        if not self.quant_config.sym:
            self.register_buffer('qzeros', torch.zeros(
                (out_features, num_groups),
                dtype=torch.int32
            ))

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))

    def quantize_weights(
        self,
        weight: torch.Tensor,
        H: Optional[torch.Tensor] = None
    ):
        """
        Quantize weights using GPTQ algorithm.

        Args:
            weight: Original weight matrix [out_features, in_features]
            H: Hessian matrix (X^T X) for optimal quantization
        """
        bits = self.quant_config.bits
        group_size = self.quant_config.group_size
        maxq = 2 ** bits - 1

        weight = weight.clone()
        out_features, in_features = weight.shape

        # Compute or use provided Hessian
        if H is None:
            H = torch.eye(in_features, device=weight.device, dtype=torch.float32)

        # Add dampening
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        damp = self.quant_config.damp_percent * torch.mean(torch.diag(H))
        diag = torch.arange(in_features, device=weight.device)
        H[diag, diag] += damp

        # Invert Hessian
        try:
            Hinv = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(Hinv)
        except:
            # Fallback to regular inverse if Cholesky fails
            Hinv = torch.linalg.inv(H)

        # Quantize column by column (or group by group)
        Q = torch.zeros_like(weight)
        Losses = torch.zeros_like(weight)
        Err = torch.zeros_like(weight)

        for i1 in range(0, in_features, group_size):
            i2 = min(i1 + group_size, in_features)
            count = i2 - i1

            W = weight[:, i1:i2].clone()
            Q1 = torch.zeros_like(W)

            # Find scale for this group
            if self.quant_config.sym:
                # Symmetric: scale based on max absolute value
                scale = W.abs().max(dim=1, keepdim=True)[0] / (maxq / 2)
                scale = scale.clamp(min=1e-5)
            else:
                # Asymmetric: scale and zero point
                wmin = W.min(dim=1, keepdim=True)[0]
                wmax = W.max(dim=1, keepdim=True)[0]
                scale = (wmax - wmin) / maxq
                scale = scale.clamp(min=1e-5)

            # Quantize
            if self.quant_config.sym:
                q = torch.clamp(torch.round(W / scale), -(maxq // 2), maxq // 2)
            else:
                zero = torch.round(-wmin / scale)
                q = torch.clamp(torch.round(W / scale + zero), 0, maxq)

            # Dequantize
            if self.quant_config.sym:
                Q1 = q * scale
            else:
                Q1 = (q - zero) * scale

            # Apply optimal brain surgeon correction
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W[:, i]
                q = Q1[:, i]

                d = Hinv1[i, i]
                err = (w - q) / d

                Q1[:, i] = q
                Losses[:, i1:i2] += err.unsqueeze(1) ** 2

                # Update remaining weights
                if i < count - 1:
                    W[:, i + 1:] -= err.unsqueeze(1) * Hinv1[i, i + 1:].unsqueeze(0)

            # Store quantized weights and scale
            Q[:, i1:i2] = Q1

            # Store scale for this group
            group_idx = i1 // group_size
            self.scales[:, group_idx] = scale.squeeze()

        # Pack quantized weights into int32
        self._pack_weights(Q)

    def _pack_weights(self, Q: torch.Tensor):
        """Pack quantized weights into int32"""
        bits = self.quant_config.bits
        maxq = 2 ** bits - 1

        # Quantize to int
        if self.quant_config.sym:
            Q_int = torch.clamp(Q, -(maxq // 2), maxq // 2).to(torch.int32)
        else:
            Q_int = torch.clamp(Q, 0, maxq).to(torch.int32)

        # Pack multiple values into int32
        values_per_int = 32 // bits
        packed = torch.zeros(
            (Q.shape[0], Q.shape[1] // values_per_int),
            dtype=torch.int32,
            device=Q.device
        )

        for i in range(values_per_int):
            packed |= (Q_int[:, i::values_per_int] & maxq) << (i * bits)

        self.qweight.copy_(packed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization"""
        # Dequantize weights on the fly
        weight = self._dequantize_weights()

        # Standard linear operation
        out = F.linear(x, weight, self.bias if self.has_bias else None)

        return out

    def _dequantize_weights(self) -> torch.Tensor:
        """Dequantize weights from packed int32"""
        bits = self.quant_config.bits
        group_size = self.quant_config.group_size
        maxq = 2 ** bits - 1

        # Unpack
        values_per_int = 32 // bits
        out_features, _ = self.qweight.shape
        in_features = self.qweight.shape[1] * values_per_int

        Q_int = torch.zeros(
            (out_features, in_features),
            dtype=torch.int32,
            device=self.qweight.device
        )

        for i in range(values_per_int):
            Q_int[:, i::values_per_int] = (self.qweight >> (i * bits)) & maxq

        # Convert to float and apply scales
        Q_float = Q_int.to(torch.float16)

        for i in range(self.scales.shape[1]):
            i1 = i * group_size
            i2 = min(i1 + group_size, in_features)
            scale = self.scales[:, i:i + 1]

            if self.quant_config.sym:
                Q_float[:, i1:i2] *= scale
            else:
                zero = self.qzeros[:, i:i + 1]
                Q_float[:, i1:i2] = (Q_float[:, i1:i2] - zero) * scale

        return Q_float


# ============================================================================
# AWQ - Activation-Aware Weight Quantization
# ============================================================================

@dataclass
class AWQConfig:
    """Configuration for AWQ quantization"""
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True

    # Activation-aware scaling
    alpha: float = 0.5  # Scaling factor for important channels
    search_alpha: bool = True  # Search for optimal alpha


class AWQLinear(QuantizedLinear):
    """
    AWQ-quantized Linear Layer

    Protects important weight channels based on activation magnitudes.
    Key insight: not all weights are equally important!

    Example:
        >>> config = AWQConfig(bits=4, group_size=128)
        >>> layer = AWQLinear(4096, 4096, quant_config=config)
        >>> layer.quantize_weights(weight, activations)
        >>> output = layer(input)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_config: Optional[AWQConfig] = None
    ):
        super().__init__(in_features, out_features, bias)
        self.quant_config = quant_config or AWQConfig()

        # Quantized weights
        num_groups = in_features // self.quant_config.group_size
        self.register_buffer('qweight', torch.zeros(
            (out_features, in_features),
            dtype=torch.int8
        ))
        self.register_buffer('scales', torch.zeros(
            (out_features, num_groups),
            dtype=torch.float16
        ))
        self.register_buffer('zeros', torch.zeros(
            (out_features, num_groups),
            dtype=torch.float16
        ))

        # Activation-aware channel scaling
        self.register_buffer('channel_scales', torch.ones(
            in_features,
            dtype=torch.float16
        ))

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))

    def quantize_weights(
        self,
        weight: torch.Tensor,
        activations: Optional[torch.Tensor] = None
    ):
        """
        Quantize weights with activation-aware scaling.

        Args:
            weight: Original weight matrix [out_features, in_features]
            activations: Sample activations [batch, in_features] for importance
        """
        # Compute channel importance from activations
        if activations is not None:
            # Average magnitude per channel
            importance = activations.abs().mean(dim=0)

            # Scale factor: s_x^alpha
            alpha = self.quant_config.alpha
            if self.quant_config.search_alpha:
                # Grid search for best alpha (simplified)
                alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
                best_alpha = alpha
                best_error = float('inf')

                for a in alphas:
                    s = importance ** a
                    scaled_weight = weight / s.unsqueeze(0)
                    error = self._quantization_error(scaled_weight)
                    if error < best_error:
                        best_error = error
                        best_alpha = a

                alpha = best_alpha

            # Compute channel scales
            self.channel_scales.copy_(importance ** alpha)

            # Scale weights by channel importance
            weight = weight / self.channel_scales.unsqueeze(0)

        # Quantize scaled weights
        bits = self.quant_config.bits
        group_size = self.quant_config.group_size
        maxq = 2 ** bits - 1

        for i in range(0, weight.shape[1], group_size):
            i1 = i
            i2 = min(i + group_size, weight.shape[1])
            group_idx = i // group_size

            W = weight[:, i1:i2]

            # Compute scale and zero point
            wmin = W.min(dim=1, keepdim=True)[0]
            wmax = W.max(dim=1, keepdim=True)[0]

            scale = (wmax - wmin) / maxq
            scale = scale.clamp(min=1e-5)

            if self.quant_config.zero_point:
                zero = torch.round(-wmin / scale)
            else:
                zero = torch.zeros_like(wmin)

            # Quantize
            q = torch.clamp(torch.round(W / scale + zero), 0, maxq)

            # Store
            self.qweight[:, i1:i2] = q.to(torch.int8)
            self.scales[:, group_idx] = scale.squeeze()
            self.zeros[:, group_idx] = zero.squeeze()

    def _quantization_error(self, weight: torch.Tensor) -> float:
        """Estimate quantization error"""
        bits = self.quant_config.bits
        maxq = 2 ** bits - 1

        # Simple quantization
        wmin = weight.min()
        wmax = weight.max()
        scale = (wmax - wmin) / maxq

        q = torch.clamp(torch.round((weight - wmin) / scale), 0, maxq)
        dequant = q * scale + wmin

        error = (weight - dequant).pow(2).mean().item()
        return error

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Dequantize weights
        weight = self._dequantize_weights()

        # Linear operation
        out = F.linear(x, weight, self.bias if self.has_bias else None)

        return out

    def _dequantize_weights(self) -> torch.Tensor:
        """Dequantize weights"""
        group_size = self.quant_config.group_size
        weight = self.qweight.to(torch.float16)

        for i in range(self.scales.shape[1]):
            i1 = i * group_size
            i2 = min(i1 + group_size, weight.shape[1])

            scale = self.scales[:, i:i + 1]
            zero = self.zeros[:, i:i + 1]

            weight[:, i1:i2] = (weight[:, i1:i2] - zero) * scale

        # Unscale by channel scales
        weight = weight * self.channel_scales.unsqueeze(0)

        return weight


# ============================================================================
# bitsandbytes - 8-bit and 4-bit Quantization
# ============================================================================

@dataclass
class BitsAndBytesConfig:
    """Configuration for bitsandbytes quantization"""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # 4-bit config (QLoRA)
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_use_double_quant: bool = True  # Double quantization
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"

    # Outlier handling (LLM.int8())
    llm_int8_threshold: float = 6.0  # Outlier threshold
    llm_int8_has_fp16_weight: bool = False


class Int8Linear(QuantizedLinear):
    """
    8-bit Linear with Outlier Handling (LLM.int8())

    Key insight: Most activations are normal, but a few outliers
    cause large errors. Handle outliers in FP16, rest in INT8.

    Example:
        >>> layer = Int8Linear(4096, 4096, threshold=6.0)
        >>> layer.quantize_weights(weight)
        >>> output = layer(input)  # 2x memory reduction
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 6.0
    ):
        super().__init__(in_features, out_features, bias)
        self.threshold = threshold

        # INT8 weights
        self.register_buffer('weight_int8', torch.zeros(
            (out_features, in_features),
            dtype=torch.int8
        ))
        self.register_buffer('weight_scale', torch.zeros(
            out_features,
            dtype=torch.float32
        ))

        # Outlier columns (keep in FP16)
        self.register_buffer('outlier_cols', torch.tensor([], dtype=torch.long))
        self.register_buffer('outlier_weights', torch.tensor([], dtype=torch.float16))

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))

    def quantize_weights(self, weight: torch.Tensor):
        """Quantize weights to INT8 with outlier detection"""
        # Compute column-wise magnitude
        col_magnitude = weight.abs().max(dim=0)[0]

        # Find outlier columns
        outlier_mask = col_magnitude > self.threshold
        self.outlier_cols = torch.where(outlier_mask)[0]

        # Separate outliers and normal weights
        normal_mask = ~outlier_mask
        normal_weight = weight[:, normal_mask]

        if self.outlier_cols.numel() > 0:
            self.outlier_weights = weight[:, outlier_mask].to(torch.float16)

        # Quantize normal weights per row
        scale = normal_weight.abs().max(dim=1, keepdim=True)[0] / 127.0
        scale = scale.clamp(min=1e-5)

        weight_int8 = torch.clamp(
            torch.round(normal_weight / scale),
            -127, 127
        ).to(torch.int8)

        # Store
        full_weight_int8 = torch.zeros_like(weight, dtype=torch.int8)
        full_weight_int8[:, normal_mask] = weight_int8

        self.weight_int8.copy_(full_weight_int8)
        self.weight_scale.copy_(scale.squeeze())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with mixed precision"""
        # Detect outliers in input
        x_magnitude = x.abs().max()

        if x_magnitude > self.threshold or self.outlier_cols.numel() > 0:
            # Mixed precision path
            return self._mixed_precision_forward(x)
        else:
            # Pure INT8 path
            return self._int8_forward(x)

    def _int8_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pure INT8 matrix multiplication"""
        # Quantize input
        x_scale = x.abs().max() / 127.0
        x_int8 = torch.clamp(torch.round(x / x_scale), -127, 127).to(torch.int8)

        # INT8 matmul (simulated - would use INT8 GEMM in practice)
        out_int8 = torch.matmul(x_int8.float(), self.weight_int8.t().float())

        # Dequantize
        out = out_int8 * x_scale * self.weight_scale.unsqueeze(0)

        if self.has_bias:
            out += self.bias

        return out

    def _mixed_precision_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mixed precision with outlier handling"""
        # Build normal mask
        all_cols = torch.arange(self.in_features, device=x.device)
        normal_mask = ~torch.isin(all_cols, self.outlier_cols)

        # Normal path (INT8)
        x_normal = x[..., normal_mask]
        weight_normal = self.weight_int8[:, normal_mask].float()
        weight_normal = weight_normal * self.weight_scale.unsqueeze(1)

        out = torch.matmul(x_normal, weight_normal.t())

        # Outlier path (FP16)
        if self.outlier_cols.numel() > 0:
            x_outlier = x[..., self.outlier_cols]
            out += torch.matmul(x_outlier, self.outlier_weights.t())

        if self.has_bias:
            out += self.bias

        return out


@dataclass
class NF4Quantizer:
    """
    4-bit NormalFloat Quantizer

    Optimal for normally-distributed weights (like in neural nets).
    Used in QLoRA.
    """

    # NF4 quantization levels (asymmetric)
    NF4_LEVELS = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367,
        -0.39491748809814453, -0.28444138169288635,
        -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725,
        0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0
    ])

    @staticmethod
    def quantize(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize to NF4.

        Returns:
            (quantized_indices, scale)
        """
        # Normalize to [-1, 1]
        scale = weight.abs().max()
        normalized = weight / scale

        # Find closest NF4 level
        levels = NF4Quantizer.NF4_LEVELS.to(weight.device)
        distances = (normalized.unsqueeze(-1) - levels).abs()
        indices = distances.argmin(dim=-1).to(torch.uint8)

        return indices, scale

    @staticmethod
    def dequantize(indices: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize from NF4"""
        levels = NF4Quantizer.NF4_LEVELS.to(indices.device)
        values = levels[indices]
        return values * scale


# ============================================================================
# GGML/GGUF Format Support
# ============================================================================

@dataclass
class GGMLConfig:
    """Configuration for GGML/GGUF format"""
    quant_type: str = "Q4_K_M"  # Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K_S, Q4_K_M, etc.


class GGMLQuantizer:
    """
    GGML/GGUF Format Quantizer

    Compatible with llama.cpp for CPU inference.

    Quantization types:
    - Q4_0: 4-bit, 32 values per block
    - Q4_1: 4-bit with min value per block
    - Q5_0/Q5_1: 5-bit variants
    - Q8_0: 8-bit
    - Q4_K_S/M/L: K-quantization (smaller, medium, large)

    Example:
        >>> quantizer = GGMLQuantizer("Q4_K_M")
        >>> quantized = quantizer.quantize(weight)
        >>> # Export to GGUF file for llama.cpp
    """

    def __init__(self, quant_type: str = "Q4_K_M"):
        self.quant_type = quant_type

    def quantize(self, weight: torch.Tensor) -> bytes:
        """
        Quantize weight to GGML format.

        Returns:
            Serialized bytes in GGML format
        """
        if self.quant_type == "Q4_0":
            return self._quantize_q4_0(weight)
        elif self.quant_type == "Q4_1":
            return self._quantize_q4_1(weight)
        elif self.quant_type == "Q8_0":
            return self._quantize_q8_0(weight)
        else:
            raise ValueError(f"Unsupported quant type: {self.quant_type}")

    def _quantize_q4_0(self, weight: torch.Tensor) -> bytes:
        """Q4_0: 4-bit, scale per 32 values"""
        block_size = 32
        weight = weight.flatten()

        # Pad to multiple of block_size
        n = weight.numel()
        n_blocks = (n + block_size - 1) // block_size
        if n % block_size != 0:
            weight = F.pad(weight, (0, n_blocks * block_size - n))

        weight = weight.reshape(n_blocks, block_size)

        # Quantize each block
        blocks = []
        for block in weight:
            # Compute scale
            amax = block.abs().max()
            scale = amax / 7.0  # 4-bit signed: -7 to 7

            # Quantize
            q = torch.clamp(torch.round(block / scale), -7, 7).to(torch.int8)

            # Pack scale (fp16) + quantized values
            scale_bytes = scale.to(torch.float16).numpy().tobytes()
            q_bytes = q.numpy().tobytes()

            blocks.append(scale_bytes + q_bytes)

        return b''.join(blocks)

    def _quantize_q4_1(self, weight: torch.Tensor) -> bytes:
        """Q4_1: 4-bit with min per 32 values"""
        # Similar to Q4_0 but stores min value too
        block_size = 32
        weight = weight.flatten()

        n = weight.numel()
        n_blocks = (n + block_size - 1) // block_size
        if n % block_size != 0:
            weight = F.pad(weight, (0, n_blocks * block_size - n))

        weight = weight.reshape(n_blocks, block_size)

        blocks = []
        for block in weight:
            # Compute scale and min
            bmin = block.min()
            bmax = block.max()
            scale = (bmax - bmin) / 15.0  # 4-bit unsigned: 0 to 15

            # Quantize
            q = torch.clamp(torch.round((block - bmin) / scale), 0, 15).to(torch.uint8)

            # Pack scale (fp16) + min (fp16) + quantized values
            scale_bytes = scale.to(torch.float16).numpy().tobytes()
            min_bytes = bmin.to(torch.float16).numpy().tobytes()
            q_bytes = q.numpy().tobytes()

            blocks.append(scale_bytes + min_bytes + q_bytes)

        return b''.join(blocks)

    def _quantize_q8_0(self, weight: torch.Tensor) -> bytes:
        """Q8_0: 8-bit per 32 values"""
        block_size = 32
        weight = weight.flatten()

        n = weight.numel()
        n_blocks = (n + block_size - 1) // block_size
        if n % block_size != 0:
            weight = F.pad(weight, (0, n_blocks * block_size - n))

        weight = weight.reshape(n_blocks, block_size)

        blocks = []
        for block in weight:
            # Compute scale
            amax = block.abs().max()
            scale = amax / 127.0

            # Quantize
            q = torch.clamp(torch.round(block / scale), -127, 127).to(torch.int8)

            # Pack
            scale_bytes = scale.to(torch.float32).numpy().tobytes()
            q_bytes = q.numpy().tobytes()

            blocks.append(scale_bytes + q_bytes)

        return b''.join(blocks)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Quantization - Model Compression Techniques")
    print("=" * 80)

    # Example weight matrix
    weight = torch.randn(1024, 1024) * 0.1

    # GPTQ Example
    print("\n" + "=" * 80)
    print("GPTQ - Optimal Brain Surgeon Quantization")
    print("=" * 80)

    gptq_config = GPTQConfig(bits=4, group_size=128)
    gptq_layer = GPTQLinear(1024, 1024, quant_config=gptq_config)
    gptq_layer.quantize_weights(weight)

    x = torch.randn(1, 1024)
    out = gptq_layer(x)

    print(f"Original weight size: {weight.numel() * 2 / 1024:.2f} KB (FP16)")
    print(f"Quantized weight size: {gptq_layer.qweight.numel() * 4 / 1024:.2f} KB (INT32 packed)")
    print(f"Compression ratio: {weight.numel() * 2 / (gptq_layer.qweight.numel() * 4):.1f}x")
    print(f"Output shape: {out.shape}")

    # AWQ Example
    print("\n" + "=" * 80)
    print("AWQ - Activation-Aware Weight Quantization")
    print("=" * 80)

    awq_config = AWQConfig(bits=4, group_size=128)
    awq_layer = AWQLinear(1024, 1024, quant_config=awq_config)

    # Generate sample activations
    activations = torch.randn(128, 1024).abs()
    awq_layer.quantize_weights(weight, activations)

    out = awq_layer(x)
    print(f"Quantized with activation awareness")
    print(f"Output shape: {out.shape}")

    # LLM.int8() Example
    print("\n" + "=" * 80)
    print("LLM.int8() - 8-bit with Outlier Handling")
    print("=" * 80)

    int8_layer = Int8Linear(1024, 1024, threshold=6.0)
    int8_layer.quantize_weights(weight)

    out = int8_layer(x)
    print(f"Original weight size: {weight.numel() * 2 / 1024:.2f} KB (FP16)")
    print(f"INT8 weight size: {int8_layer.weight_int8.numel() / 1024:.2f} KB")
    print(f"Compression ratio: ~2x")
    print(f"Output shape: {out.shape}")

    # NF4 Example
    print("\n" + "=" * 80)
    print("NF4 - 4-bit NormalFloat (QLoRA)")
    print("=" * 80)

    indices, scale = NF4Quantizer.quantize(weight)
    dequantized = NF4Quantizer.dequantize(indices, scale)

    error = (weight - dequantized).pow(2).mean().sqrt()
    print(f"Original weight size: {weight.numel() * 2 / 1024:.2f} KB (FP16)")
    print(f"NF4 weight size: {indices.numel() * 0.5 / 1024:.2f} KB (4-bit)")
    print(f"Compression ratio: 4x")
    print(f"Reconstruction RMSE: {error:.6f}")

    # GGML Example
    print("\n" + "=" * 80)
    print("GGML/GGUF - llama.cpp Compatible Format")
    print("=" * 80)

    ggml_quantizer = GGMLQuantizer("Q4_0")
    quantized_bytes = ggml_quantizer.quantize(weight[:256, :256])  # Small example

    print(f"Quantized to GGML Q4_0 format")
    print(f"Serialized size: {len(quantized_bytes)} bytes")
    print(f"Can be loaded by llama.cpp for CPU inference")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
Quantization Techniques Comparison:

1. GPTQ (4-bit):
   - Compression: 4x
   - Quality: Excellent (optimal quantization)
   - Speed: Fast inference
   - Use: General purpose, best quality/size trade-off

2. AWQ (4-bit):
   - Compression: 4x
   - Quality: Excellent (preserves important weights)
   - Speed: Fast inference
   - Use: When you have calibration data

3. LLM.int8() (8-bit):
   - Compression: 2x
   - Quality: Excellent (outlier handling)
   - Speed: Good inference
   - Use: When you need high quality, moderate compression

4. NF4 (4-bit):
   - Compression: 4x
   - Quality: Very good
   - Speed: Fast inference
   - Use: QLoRA fine-tuning

5. GGML/GGUF:
   - Compression: 2-4x (various formats)
   - Quality: Good to excellent
   - Speed: Optimized for CPU
   - Use: llama.cpp, CPU inference

Recommendation:
- Training: FP16/BF16
- Fine-tuning: QLoRA (NF4 + LoRA)
- Inference (GPU): GPTQ or AWQ
- Inference (CPU): GGML/GGUF
- Maximum quality: LLM.int8()
""")

    print("=" * 80)
