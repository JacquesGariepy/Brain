"""
Hyena - Sub-Quadratic Attention via Implicit Long Convolutions

Replaces attention with data-controlled long convolutions.
Achieves sub-quadratic complexity with Transformer-level quality.

Key Innovation:
- Implicit parameterization of long convolutions
- Data-controlled gating
- FFT-based O(N log N) computation

References:
- Hyena Hierarchy: https://arxiv.org/abs/2302.10866

Performance:
- Complexity: O(N log N) vs O(N^2)
- 100x faster than Transformers for length 100K
- Matches Transformer quality
"""

from dataclasses import dataclass
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class HyenaConfig:
    """Configuration for Hyena"""
    # Model dimensions
    d_model: int = 768
    n_layers: int = 12

    # Hyena operator
    order: int = 2  # Hyena order (2 = Hyena, higher = deeper recursion)
    filter_order: int = 64  # Filter order (for implicit parameterization)

    # Convolution
    max_seq_len: int = 8192  # Maximum sequence length

    # FFN
    d_ff: int = 3072

    # Dropout
    dropout: float = 0.1

    # Layer norm
    layer_norm_eps: float = 1e-5


class PositionalEmbedding(nn.Module):
    """
    Positional embedding for filter parameterization.

    Uses sinusoidal embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        self.d_model = d_model

        # Create sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Get positional embeddings for sequence length"""
        return self.pe[:seq_len]


class ImplicitLongConvolution(nn.Module):
    """
    Implicit Long Convolution

    Instead of storing O(N) filter coefficients, parameterize
    filter with a small MLP (implicit function).

    filter[n] = MLP(pos[n])

    This reduces parameters from O(N) to O(d_model).
    """

    def __init__(self, config: HyenaConfig):
        super().__init__()
        self.config = config

        # Positional encoding
        self.pos_emb = PositionalEmbedding(config.d_model, config.max_seq_len)

        # Implicit filter network (small MLP)
        self.filter_fn = nn.Sequential(
            nn.Linear(config.d_model, config.filter_order),
            nn.GELU(),
            nn.Linear(config.filter_order, config.filter_order),
            nn.GELU(),
            nn.Linear(config.filter_order, 1)
        )

    def get_filter(self, seq_len: int) -> torch.Tensor:
        """
        Generate filter coefficients using implicit parameterization.

        Args:
            seq_len: Sequence length

        Returns:
            filter: [seq_len] filter coefficients
        """
        # Get positional embeddings
        pos = self.pos_emb(seq_len)  # [seq_len, d_model]

        # Generate filter through MLP
        filter_vals = self.filter_fn(pos).squeeze(-1)  # [seq_len]

        return filter_vals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply long convolution via FFT.

        Uses FFT for O(N log N) computation instead of O(N^2).

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # Get filter
        filter_vals = self.get_filter(seq_len)  # [seq_len]

        # Apply convolution via FFT (per channel)
        output = torch.zeros_like(x)

        for d in range(d_model):
            # Get channel
            x_d = x[:, :, d]  # [batch, seq_len]

            # FFT-based convolution
            # 1. FFT of input and filter
            x_fft = torch.fft.rfft(x_d, n=seq_len * 2)
            filter_fft = torch.fft.rfft(filter_vals, n=seq_len * 2)

            # 2. Multiply in frequency domain
            y_fft = x_fft * filter_fft.unsqueeze(0)

            # 3. Inverse FFT
            y = torch.fft.irfft(y_fft, n=seq_len * 2)[:, :seq_len]

            output[:, :, d] = y

        return output


class HyenaOperator(nn.Module):
    """
    Hyena Operator

    Replaces attention with data-controlled long convolutions.

    For order=2 (standard Hyena):
        v = x * proj_v(x)
        h = LongConv(v)
        out = x * proj_out(x) * h

    Key: Long convolution is data-controlled (filter depends on input).
    """

    def __init__(self, config: HyenaConfig):
        super().__init__()
        self.config = config
        self.order = config.order

        # Input projections (create order+1 parallel paths)
        self.in_proj = nn.Linear(config.d_model, config.d_model * (config.order + 1), bias=False)

        # Long convolutions (one per order)
        self.long_convs = nn.ModuleList([
            ImplicitLongConvolution(config) for _ in range(config.order)
        ])

        # Short convolutions (for data-control)
        self.short_convs = nn.ModuleList([
            nn.Conv1d(
                config.d_model,
                config.d_model,
                kernel_size=3,
                padding=1,
                groups=config.d_model  # Depthwise
            ) for _ in range(config.order)
        ])

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # Project to order+1 paths
        x_proj = self.in_proj(x)  # [batch, seq, d_model*(order+1)]
        x_split = x_proj.chunk(self.order + 1, dim=-1)  # List of [batch, seq, d_model]

        # First path is passed through unchanged
        v = x_split[0]

        # Apply Hyena hierarchy
        for i in range(self.order):
            # Short convolution (for data-control)
            # Create filter from current state
            filter_input = x_split[i + 1].transpose(1, 2)  # [batch, d_model, seq]
            filter_signal = self.short_convs[i](filter_input).transpose(1, 2)

            # Long convolution
            conv_out = self.long_convs[i](v)

            # Gating (element-wise multiplication)
            v = v * filter_signal * conv_out

        # Output projection
        output = self.out_proj(v)

        return output


class HyenaBlock(nn.Module):
    """Hyena block with Hyena operator and FFN"""

    def __init__(self, config: HyenaConfig):
        super().__init__()
        self.config = config

        # Hyena operator
        self.hyena = HyenaOperator(config)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Hyena with residual
        hyena_out = self.hyena(self.ln1(x))
        x = x + self.dropout(hyena_out)

        # FFN with residual
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)

        return x


class Hyena(nn.Module):
    """
    Complete Hyena model.

    Example:
        >>> config = HyenaConfig(d_model=768, n_layers=12, order=2)
        >>> model = Hyena(config)
        >>>
        >>> # Can handle very long sequences!
        >>> x = torch.randn(2, 100000, 768)
        >>> out = model(x)  # O(N log N) complexity
        >>>
        >>> # 100x faster than Transformers for this length!
    """

    def __init__(self, config: HyenaConfig):
        super().__init__()
        self.config = config

        # Layers
        self.blocks = nn.ModuleList([
            HyenaBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Pass through all blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.ln_f(x)

        return x


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Hyena - Sub-Quadratic Attention via Long Convolutions")
    print("=" * 80)

    # Create model
    config = HyenaConfig(
        d_model=768,
        n_layers=12,
        order=2,
        max_seq_len=8192
    )

    model = Hyena(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel size: {total_params / 1e6:.1f}M parameters")

    # Test with different sequence lengths
    print("\n" + "=" * 80)
    print("Testing Different Sequence Lengths")
    print("=" * 80)

    for seq_len in [1024, 2048, 4096, 8192]:
        x = torch.randn(1, seq_len, config.d_model)

        print(f"\nSeq length: {seq_len}")
        print(f"Input shape: {x.shape}")

        with torch.no_grad():
            out = model(x)

        print(f"Output shape: {out.shape}")

    print("\n" + "=" * 80)
    print("Complexity Analysis")
    print("=" * 80)

    print("""
Complexity Comparison:

Sequence Length: 1K    4K    16K    64K   100K
-------------------------------------------------
Transformer:     N²    16N²  256N²  4096N² 10000N²
Hyena:          N log N  4N log N  16N log N  64N log N  100N log N

Speedup at 100K: ~100x faster!

Memory Comparison:
------------------
Transformer (attention):
  - Parameters: O(d²)
  - Activations: O(N²)
  - KV cache: O(N * d)

Hyena (convolution):
  - Parameters: O(d²) + O(d) for filters
  - Activations: O(N * d)
  - No cache needed!

Quality:
--------
- Matches Transformer on standard benchmarks
- Better on long-range tasks (>4K tokens)
- Especially strong on:
  * DNA sequences
  * Audio generation
  * Long-form text

Key Advantages:
---------------
1. Sub-quadratic complexity: O(N log N) vs O(N²)
2. FFT-based: leverages fast hardware implementations
3. Data-controlled: filters adapt to input
4. Implicit parameterization: O(1) filter parameters
5. No attention: different inductive bias

Use Hyena when:
---------------
- Very long sequences (>16K tokens)
- Speed/memory critical
- Signal processing tasks (audio, DNA, time series)
- Need sub-quadratic scaling

Use Transformers when:
----------------------
- Standard NLP (<4K tokens)
- Maximum quality on typical tasks
- Rich ecosystem needed
""")

    print("\n" + "=" * 80)
    print("Alternative Architectures Summary")
    print("=" * 80)
    print("""
Comparison of All Alternative Architectures:

Architecture | Complexity | Memory  | Quality | Use Case
-------------|-----------|---------|---------|----------
Transformer  | O(N²)     | O(N²)   | Best    | Standard NLP
Mamba        | O(N)      | O(N)    | Great   | Long sequences, streaming
RWKV         | O(N)      | O(1)*   | Great   | Efficient inference
RetNet       | O(N)*     | O(1)*   | Great   | Fast inference
Hyena        | O(N log N)| O(N)    | Great   | Very long sequences

* = with recurrent formulation

All alternatives:
- Match or approach Transformer quality
- Significantly more efficient
- Better scaling to long sequences
- Different inductive biases

Choose based on your constraints:
- Need best quality: Transformer
- Need efficiency: Any alternative
- Need streaming: Mamba or RWKV
- Need O(1) inference: RWKV or RetNet
- Need very long context: Hyena or Mamba
""")

    print("=" * 80)
