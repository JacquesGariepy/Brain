"""
Multi-Head Attention Mechanism - SOTA Implementation

This module implements state-of-the-art multi-head attention with various optimizations:
- Scaled dot-product attention
- Multi-head parallelism
- Causal masking support
- Attention dropout
- Key-value caching for inference
- Flash Attention compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """Configuration for Multi-Head Attention"""
    d_model: int = 512  # Model dimension
    num_heads: int = 8  # Number of attention heads
    dropout: float = 0.1  # Dropout probability
    bias: bool = True  # Use bias in linear projections
    causal: bool = False  # Use causal (autoregressive) masking
    use_flash: bool = False  # Use Flash Attention if available
    qkv_bias: bool = True  # Bias for QKV projections
    out_bias: bool = True  # Bias for output projection


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism with SOTA optimizations.

    Features:
    - Efficient multi-head parallelization
    - Scaled dot-product attention
    - Optional causal masking
    - Dropout regularization
    - KV caching for autoregressive generation
    - Flash Attention support (when available)

    Args:
        config: AttentionConfig with all hyperparameters
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config

        assert config.d_model % config.num_heads == 0, \
            f"d_model ({config.d_model}) must be divisible by num_heads ({config.num_heads})"

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection - can be fused for efficiency
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.out_bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Flash Attention support
        self.use_flash = config.use_flash and hasattr(F, 'scaled_dot_product_attention')

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            kv_cache: Optional cached (key, value) for autoregressive generation
            use_cache: Whether to return updated cache

        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            new_cache: Updated (key, value) cache if use_cache=True
        """
        batch_size, seq_len, d_model = x.shape

        # Compute QKV
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Handle KV cache for autoregressive generation
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Compute attention
        if self.use_flash:
            # Use Flash Attention (PyTorch 2.0+)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=self.config.causal and mask is None
            )
        else:
            # Standard scaled dot-product attention
            attn_output = self._standard_attention(q, k, v, mask)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)

        output = self.resid_dropout(self.out_proj(attn_output))

        return output, new_cache

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention.

        Args:
            q: Query tensor (batch, heads, seq_q, head_dim)
            k: Key tensor (batch, heads, seq_k, head_dim)
            v: Value tensor (batch, heads, seq_v, head_dim)
            mask: Optional attention mask

        Returns:
            Attention output (batch, heads, seq_q, head_dim)
        """
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if self.config.causal:
            seq_len = q.size(2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # Apply additional mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)

        return attn_output

    def get_attention_maps(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get attention maps for visualization.

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Attention weights of shape (batch, heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape

        # Compute QKV
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply masks
        if self.config.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Return attention weights
        return F.softmax(attn_scores, dim=-1)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) - More efficient variant of MHA.

    Uses fewer key-value heads than query heads for better efficiency.
    This is used in modern LLMs like Llama 2.

    Args:
        d_model: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (typically num_heads // 4 or // 8)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
        bias: bool = False
    ):
        super().__init__()

        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate Q and KV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_proj = nn.Linear(d_model, 2 * num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of Grouped-Query Attention"""
        batch_size, seq_len, d_model = x.shape

        # Project queries
        q = self.q_proj(x)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

        # Project keys and values
        kv = self.kv_proj(x)
        kv = kv.reshape(batch_size, seq_len, 2, self.num_kv_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, batch, num_kv_heads, seq_len, head_dim)
        k, v = kv[0], kv[1]

        # Expand KV to match number of query heads
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)

        output = self.resid_dropout(self.out_proj(attn_output))

        return output


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) - Even more efficient variant.

    Uses single key-value head shared across all query heads.
    Used for very fast inference in models like PaLM.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Q projection for all heads, single KV
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_proj = nn.Linear(d_model, 2 * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of Multi-Query Attention"""
        batch_size, seq_len, d_model = x.shape

        # Project queries (one per head)
        q = self.q_proj(x)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)

        # Project single shared KV
        kv = self.kv_proj(x)
        kv = kv.reshape(batch_size, seq_len, 2, self.head_dim)
        kv = kv.permute(2, 0, 1, 3)  # (2, batch, seq_len, head_dim)
        k, v = kv[0], kv[1]

        # Expand to all heads
        k = k.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        v = v.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)

        output = self.resid_dropout(self.out_proj(attn_output))

        return output
