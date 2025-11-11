"""
Advanced Attention Mechanisms for LLMs

Implements all SOTA attention variants:
- Flash Attention (v1 & v2): Memory-efficient exact attention
- Multi-Query Attention (MQA): Faster inference with shared K/V
- Grouped-Query Attention (GQA): Balance between MHA and MQA
- Sliding Window Attention: Local attention for long sequences
- Sparse Attention: Reduced complexity patterns
- Linear Attention: O(N) complexity alternatives
"""

from .flash_attention import FlashAttention, FlashAttentionV2
from .multi_query_attention import MultiQueryAttention, GroupedQueryAttention
from .sliding_window import SlidingWindowAttention
from .sparse_attention import SparseAttention, BigBirdAttention, LongformerAttention
from .linear_attention import LinearAttention, PerformerAttention
from .rope import RoPE, apply_rotary_pos_emb
from .alibi import ALiBi

__all__ = [
    'FlashAttention', 'FlashAttentionV2',
    'MultiQueryAttention', 'GroupedQueryAttention',
    'SlidingWindowAttention',
    'SparseAttention', 'BigBirdAttention', 'LongformerAttention',
    'LinearAttention', 'PerformerAttention',
    'RoPE', 'apply_rotary_pos_emb',
    'ALiBi'
]
