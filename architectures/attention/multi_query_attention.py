"""
Multi-Query Attention (MQA) & Grouped-Query Attention (GQA)

SOTA attention variants for faster inference in LLMs.

Multi-Query Attention (MQA):
- Uses single K/V head shared across all Q heads
- 10-20x faster inference (KV cache reduction)
- Used in: PaLM, Falcon, StarCoder

Grouped-Query Attention (GQA):
- Balance between MHA and MQA
- Groups of Q heads share K/V heads
- Better quality than MQA, faster than MHA
- Used in: Llama 2, Mistral, Gemma

References:
- "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019)
- "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA).

    Uses a single K/V head shared across all Q heads for faster inference.
    Dramatically reduces KV cache size during autoregressive generation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5

        # Separate projections for Q, K, V
        # Q: num_heads separate heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # K, V: Single shared head (key difference from MHA!)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout_module = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with Multi-Query Attention.

        Args:
            x: Input (batch, seq_len, embed_dim)
            key_padding_mask: Padding mask
            attn_mask: Attention mask
            kv_cache: Cached (K, V) from previous steps
            use_cache: Whether to return KV cache

        Returns:
            output: (batch, seq_len, embed_dim)
            new_kv_cache: Updated cache if use_cache=True
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project Q to multiple heads
        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

        # Project K, V to single head
        k = self.k_proj(x)  # (batch, seq_len, head_dim)
        v = self.v_proj(x)  # (batch, seq_len, head_dim)

        # Handle KV cache for autoregressive generation
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        # Expand K, V to match Q heads
        # (batch, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
        k = k.unsqueeze(1).expand(batch_size, self.num_heads, -1, self.head_dim)
        v = v.unsqueeze(1).expand(batch_size, self.num_heads, -1, self.head_dim)

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply masks
        if attn_mask is not None:
            scores = scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_module(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)

        # Return cache if requested
        new_cache = (k[:, 0, :, :], v[:, 0, :, :]) if use_cache else None

        return output, new_cache


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).

    Interpolates between MHA (full heads) and MQA (single head).
    Groups of Q heads share K/V heads.

    Example: 8 Q heads, 2 KV heads -> 4 Q heads per KV head
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,  # If None, defaults to num_heads (MHA)
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads  # Default to MHA
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_heads_per_kv = num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout_module = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with Grouped-Query Attention.

        Args:
            x: Input (batch, seq_len, embed_dim)
            key_padding_mask: Padding mask
            attn_mask: Attention mask
            kv_cache: Cached (K, V) from previous steps
            use_cache: Whether to return KV cache

        Returns:
            output: (batch, seq_len, embed_dim)
            new_kv_cache: Updated cache if use_cache=True
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project Q to all heads
        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

        # Project K, V to num_kv_heads
        k = self.k_proj(x)
        v = self.v_proj(x)

        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        k = k.transpose(1, 2)  # (batch, num_kv_heads, seq_len, head_dim)
        v = v.transpose(1, 2)

        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Repeat K, V for each group of Q heads
        # (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
        k = k.repeat_interleave(self.num_heads_per_kv, dim=1)
        v = v.repeat_interleave(self.num_heads_per_kv, dim=1)

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply masks
        if attn_mask is not None:
            scores = scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_module(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)

        # Return cache if requested
        new_cache = None
        if use_cache:
            # Cache only the KV heads (not the repeated version)
            k_cache = k[:, ::self.num_heads_per_kv, :, :]
            v_cache = v[:, ::self.num_heads_per_kv, :, :]
            new_cache = (k_cache, v_cache)

        return output, new_cache


# Example usage and comparison
if __name__ == "__main__":
    print("="*80)
    print("Multi-Query Attention (MQA) & Grouped-Query Attention (GQA)")
    print("="*80)

    embed_dim = 512
    num_heads = 8
    seq_len = 1024
    batch_size = 2

    print(f"\nConfiguration:")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Num Q heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")

    # Create attention variants
    mqa = MultiQueryAttention(embed_dim, num_heads)
    gqa_2 = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads=2)
    gqa_4 = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads=4)
    mha = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads=num_heads)  # MHA

    # Test input
    x = torch.randn(batch_size, seq_len, embed_dim)

    print("\n" + "-"*80)
    print("Multi-Query Attention (MQA)")
    output_mqa, cache_mqa = mqa(x, use_cache=True)
    print(f"  Output shape: {output_mqa.shape}")
    print(f"  KV heads: 1")
    k_cache, v_cache = cache_mqa
    print(f"  KV cache size: {k_cache.numel() + v_cache.numel():,} elements")
    print(f"  Parameters: {sum(p.numel() for p in mqa.parameters()):,}")

    print("\n" + "-"*80)
    print("Grouped-Query Attention (2 KV heads)")
    output_gqa2, cache_gqa2 = gqa_2(x, use_cache=True)
    print(f"  Output shape: {output_gqa2.shape}")
    print(f"  KV heads: 2 (4 Q heads per KV head)")
    k_cache, v_cache = cache_gqa2
    print(f"  KV cache size: {k_cache.numel() + v_cache.numel():,} elements")
    print(f"  Parameters: {sum(p.numel() for p in gqa_2.parameters()):,}")

    print("\n" + "-"*80)
    print("Grouped-Query Attention (4 KV heads)")
    output_gqa4, cache_gqa4 = gqa_4(x, use_cache=True)
    print(f"  Output shape: {output_gqa4.shape}")
    print(f"  KV heads: 4 (2 Q heads per KV head)")
    k_cache, v_cache = cache_gqa4
    print(f"  KV cache size: {k_cache.numel() + v_cache.numel():,} elements")
    print(f"  Parameters: {sum(p.numel() for p in gqa_4.parameters()):,}")

    print("\n" + "-"*80)
    print("Multi-Head Attention (MHA - baseline)")
    output_mha, cache_mha = mha(x, use_cache=True)
    print(f"  Output shape: {output_mha.shape}")
    print(f"  KV heads: 8 (1 Q head per KV head)")
    k_cache, v_cache = cache_mha
    print(f"  KV cache size: {k_cache.numel() + v_cache.numel():,} elements")
    print(f"  Parameters: {sum(p.numel() for p in mha.parameters()):,}")

    # Compare KV cache sizes
    k_mha, v_mha = cache_mha
    mha_cache_size = k_mha.numel() + v_mha.numel()

    k_mqa, v_mqa = cache_mqa
    mqa_cache_size = k_mqa.numel() + v_mqa.numel()

    k_gqa2, v_gqa2 = cache_gqa2
    gqa2_cache_size = k_gqa2.numel() + v_gqa2.numel()

    print("\n" + "-"*80)
    print("KV Cache Size Comparison:")
    print(f"  MHA:  {mha_cache_size:,} elements (baseline)")
    print(f"  GQA-4: {gqa2_cache_size:,} elements ({mha_cache_size / gqa2_cache_size:.1f}x smaller)")
    print(f"  GQA-2: {gqa2_cache_size:,} elements ({mha_cache_size / gqa2_cache_size:.1f}x smaller)")
    print(f"  MQA:  {mqa_cache_size:,} elements ({mha_cache_size / mqa_cache_size:.1f}x smaller)")

    print("\n" + "="*80)
    print("\nUsed in:")
    print("  MQA: PaLM, Falcon, StarCoder")
    print("  GQA: Llama 2, Mistral, Gemma, Command R")
    print("="*80)
