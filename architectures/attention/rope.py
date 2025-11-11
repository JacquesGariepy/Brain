"""
RoPE - Rotary Position Embeddings

SOTA positional encoding technique for transformers.

Key advantages over absolute/learned positional embeddings:
- Relative position encoding (distance-aware)
- Extrapolates to longer sequences than trained on
- Better long-range dependencies
- No additional parameters needed

Used in: GPT-NeoX, PaLM, LLaMA, Mistral, GPT-J, etc.

References:
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RoPE(nn.Module):
    """
    Rotary Position Embeddings (RoPE).

    Applies rotation matrices to query and key embeddings to encode
    relative positions.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            dim: Dimension of embeddings (must be even)
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
            device: Device to place tensors on
        """
        super().__init__()
        assert dim % 2 == 0, "dim must be even for RoPE"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency tensor
        # freq_i = 1 / (base^(2i/dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin for all positions
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Precompute rotation matrices for all positions."""
        self.max_seq_len = seq_len

        # Position indices
        t = torch.arange(seq_len, dtype=torch.float32, device=device)

        # Compute frequencies for all positions
        # shape: (seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)

        # Compute cos and sin
        # shape: (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)

        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key.

        Args:
            q: Query tensor (batch, ..., seq_len, dim)
            k: Key tensor (batch, ..., seq_len, dim)
            seq_len: Sequence length (if None, use q.shape[-2])

        Returns:
            q_rotated, k_rotated: Rotated query and key tensors
        """
        if seq_len is None:
            seq_len = q.shape[-2]

        # Extend cache if needed
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len, q.device)

        # Get cached cos/sin for this sequence length
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]

        # Apply rotation
        q_rotated = apply_rotary_pos_emb(q, cos, sin)
        k_rotated = apply_rotary_pos_emb(k, cos, sin)

        return q_rotated, k_rotated


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims of the input.

    Splits x into two halves along the last dimension and swaps them
    with negation.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0
) -> torch.Tensor:
    """
    Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor (..., seq_len, dim)
        cos: Cosine values (seq_len, dim)
        sin: Sine values (seq_len, dim)
        offset: Position offset for the sequence

    Returns:
        Tensor with rotary embeddings applied
    """
    # Handle offset (for KV cache)
    if offset > 0:
        cos = cos[offset:offset + x.shape[-2]]
        sin = sin[offset:offset + x.shape[-2]]

    # Reshape cos/sin to match x dimensions
    # x: (..., seq_len, dim)
    # cos/sin: (seq_len, dim)
    # Need to add dimensions to match
    ndim = x.ndim
    cos = cos.view(*([1] * (ndim - 2)), *cos.shape)
    sin = sin.view(*([1] * (ndim - 2)), *sin.shape)

    # Apply rotation: x * cos + rotate_half(x) * sin
    return (x * cos) + (rotate_half(x) * sin)


class RoPEScaled(RoPE):
    """
    RoPE with scaling for longer contexts.

    Applies scaling to the base frequency to extend context length.
    Used in Code Llama, Llama 2 Long, etc.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            dim: Dimension of embeddings
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
            scaling_factor: Scaling factor for extending context
            device: Device
        """
        self.scaling_factor = scaling_factor

        # Adjust base for scaling
        scaled_base = base * (scaling_factor ** (dim / (dim - 2)))

        super().__init__(dim, max_seq_len, int(scaled_base), device)


class RoPENTK(RoPE):
    """
    RoPE with NTK (Neural Tangent Kernel) aware scaling.

    Better interpolation for long contexts by scaling base frequency
    non-linearly.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.original_max_seq_len = max_seq_len

        # NTK-aware base scaling
        base = base * scaling_factor

        super().__init__(dim, max_seq_len, base, device)


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("RoPE - Rotary Position Embeddings")
    print("="*80)

    # Configuration
    batch_size = 2
    num_heads = 8
    seq_len = 1024
    head_dim = 64

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {head_dim}")

    # Create RoPE module
    rope = RoPE(dim=head_dim, max_seq_len=seq_len)

    # Create query and key tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    print(f"\nInput shapes:")
    print(f"  Q: {q.shape}")
    print(f"  K: {k.shape}")

    # Apply RoPE
    q_rotated, k_rotated = rope(q, k)

    print(f"\nOutput shapes:")
    print(f"  Q rotated: {q_rotated.shape}")
    print(f"  K rotated: {k_rotated.shape}")

    # Verify rotation preserves norm (approximately)
    print(f"\nNorm preservation:")
    print(f"  Q original norm: {q.norm():.4f}")
    print(f"  Q rotated norm:  {q_rotated.norm():.4f}")

    # Test with different sequence lengths (extrapolation)
    print("\n" + "-"*80)
    print("Testing extrapolation to longer sequences")
    print("-"*80)

    longer_seq_len = 2048
    q_long = torch.randn(1, num_heads, longer_seq_len, head_dim)
    k_long = torch.randn(1, num_heads, longer_seq_len, head_dim)

    q_rotated_long, k_rotated_long = rope(q_long, k_long, seq_len=longer_seq_len)

    print(f"Original max length: {seq_len}")
    print(f"Extended to: {longer_seq_len}")
    print(f"Output shape: {q_rotated_long.shape}")

    # Compare different RoPE variants
    print("\n" + "-"*80)
    print("RoPE Variants Comparison")
    print("-"*80)

    rope_standard = RoPE(head_dim, max_seq_len=4096)
    rope_scaled = RoPEScaled(head_dim, max_seq_len=4096, scaling_factor=2.0)
    rope_ntk = RoPENTK(head_dim, max_seq_len=4096, scaling_factor=2.0)

    print("\nStandard RoPE:")
    print(f"  Max seq len: 4096")
    print(f"  Base: 10000")

    print("\nScaled RoPE:")
    print(f"  Max seq len: 4096 (2x context)")
    print(f"  Scaling factor: 2.0")

    print("\nNTK-aware RoPE:")
    print(f"  Max seq len: 4096 (2x context)")
    print(f"  NTK scaling: 2.0")

    print("\n" + "="*80)
    print("\nUsed in:")
    print("  RoPE: GPT-NeoX, PaLM, LLaMA, Mistral, GPT-J")
    print("  Scaled RoPE: Code Llama, Llama 2 Long")
    print("  NTK RoPE: Various long-context models")
    print("="*80)
