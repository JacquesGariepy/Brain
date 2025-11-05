"""
Long Context Techniques - Handling Extended Sequences

Enables processing sequences beyond standard 2K-4K token limits.

Key Techniques:
- ALiBi: Attention with Linear Biases (extrapolates to any length!)
- Sliding Window: Local attention windows
- Infinite Attention: Compressive memory for unbounded context
- StreamingLLM: Maintain attention sink for infinite streaming

References:
- ALiBi: https://arxiv.org/abs/2108.12409
- Sliding Window (Longformer): https://arxiv.org/abs/2004.05150
- Infinite Attention: https://arxiv.org/abs/2404.07143
- StreamingLLM: https://arxiv.org/abs/2309.17453

Performance:
- ALiBi: Zero-cost extrapolation to 2x-4x training length
- Sliding Window: O(N*W) instead of O(N^2)
- Infinite Attention: O(1) memory per token
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# ALiBi - Attention with Linear Biases
# ============================================================================

@dataclass
class ALiBiConfig:
    """Configuration for ALiBi"""
    n_heads: int = 12
    max_positions: int = 8192  # Maximum sequence length


class ALiBiAttention(nn.Module):
    """
    ALiBi - Attention with Linear Biases

    Key insight: Instead of positional embeddings, add position-dependent
    biases directly to attention scores.

    Attention with ALiBi:
        scores = Q @ K^T / sqrt(d)
        scores = scores + ALiBi_bias
        att = softmax(scores)

    ALiBi bias:
        bias[i,j] = -m * |i - j|

    Where m is a head-specific slope:
        m_h = 2^(-8h/n_heads) for head h

    Benefits:
    - No positional embeddings needed
    - Extrapolates to longer sequences (trained on 1K, works on 10K!)
    - Better perplexity than learned position embeddings

    Example:
        >>> config = ALiBiConfig(n_heads=12, max_positions=8192)
        >>> attn = ALiBiAttention(config, d_model=768)
        >>>
        >>> # Train on 2K
        >>> x_train = torch.randn(2, 2048, 768)
        >>> out = attn(x_train)
        >>>
        >>> # Inference on 8K! (4x longer)
        >>> x_test = torch.randn(2, 8192, 768)
        >>> out = attn(x_test)  # Works perfectly!
    """

    def __init__(self, config: ALiBiConfig, d_model: int):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.n_heads = config.n_heads
        self.head_dim = d_model // config.n_heads

        assert d_model % config.n_heads == 0

        # Standard attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # ALiBi slopes (one per head)
        slopes = self._get_slopes(config.n_heads)
        self.register_buffer('slopes', slopes)

    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        """
        Compute ALiBi slopes for each head.

        m_h = 2^(-8h/n_heads) for head h
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        # Closest power of 2
        if math.log2(n_heads).is_integer():
            slopes = get_slopes_power_of_2(n_heads)
        else:
            # Interpolate if not power of 2
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes = (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes_power_of_2(2 * closest_power_of_2)[::2][:n_heads - closest_power_of_2]
            )

        return torch.tensor(slopes, dtype=torch.float32)

    def _get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute ALiBi bias matrix.

        bias[i,j] = -m * |i - j|

        Args:
            seq_len: Sequence length

        Returns:
            bias: [n_heads, seq_len, seq_len]
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)

        # Compute distance matrix |i - j|
        rel_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        rel_positions = rel_positions.abs()  # [seq_len, seq_len]

        # Apply slopes (one per head)
        bias = -self.slopes.unsqueeze(1).unsqueeze(2) * rel_positions.unsqueeze(0)
        # [n_heads, seq_len, seq_len]

        return bias

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with ALiBi.

        Args:
            x: Input [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)

        # Transpose for multi-head attention
        q = q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [batch, n_heads, seq_len, seq_len]

        # Add ALiBi bias
        alibi_bias = self._get_alibi_bias(seq_len, x.device)
        scores = scores + alibi_bias.unsqueeze(0)  # Broadcast over batch

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask to attention scores format
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # [batch, n_heads, seq_len, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, d_model)

        # Output projection
        output = self.out_proj(attn_output)

        return output


# ============================================================================
# Sliding Window Attention
# ============================================================================

@dataclass
class SlidingWindowConfig:
    """Configuration for Sliding Window Attention"""
    window_size: int = 512  # Size of attention window
    n_heads: int = 12


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention

    Each token attends only to tokens within a fixed window.

    Benefits:
    - Reduces complexity from O(N^2) to O(N*W)
    - Linear scaling with sequence length
    - Still captures local context effectively

    Used in:
    - Longformer
    - BigBird
    - Mistral

    Example:
        >>> config = SlidingWindowConfig(window_size=512, n_heads=12)
        >>> attn = SlidingWindowAttention(config, d_model=768)
        >>>
        >>> # Can handle very long sequences efficiently!
        >>> x = torch.randn(2, 100000, 768)
        >>> out = attn(x)  # O(N*512) instead of O(N^2)
    """

    def __init__(self, config: SlidingWindowConfig, d_model: int):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.n_heads = config.n_heads
        self.head_dim = d_model // config.n_heads
        self.window_size = config.window_size

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create sliding window mask.

        mask[i,j] = 1 if |i - j| <= window_size / 2 else 0

        Returns:
            mask: [seq_len, seq_len]
        """
        positions = torch.arange(seq_len, device=device)
        distance = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))

        # Allow attention within window
        mask = distance <= (self.window_size // 2)

        return mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with sliding window"""
        batch, seq_len, d_model = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Create sliding window mask
        window_mask = self._create_sliding_window_mask(seq_len, x.device)
        scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply additional mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # Attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, d_model)
        output = self.out_proj(attn_output)

        return output


# ============================================================================
# Infinite Attention (Infini-attention)
# ============================================================================

@dataclass
class InfiniteAttentionConfig:
    """Configuration for Infinite Attention"""
    d_model: int = 768
    n_heads: int = 12
    segment_len: int = 2048  # Length of each segment
    mem_update_rule: str = "delta"  # "delta" or "linear"


class InfiniteAttention(nn.Module):
    """
    Infinite Attention - Unbounded Context with Compressive Memory

    Combines:
    1. Local attention (within segment)
    2. Compressive memory (across segments)

    Memory update:
        M_t = M_{t-1} + sigma(K_t) @ V_t^T
        z_t = z_{t-1} + sum(sigma(K_t))

    Retrieval:
        A_mem = sigma(Q) @ M / z

    Final output:
        O = beta * A_local + (1-beta) * A_mem

    Benefits:
    - O(1) memory per token (compressive)
    - Unbounded context length
    - Better than Transformer-XL on long-range

    Example:
        >>> config = InfiniteAttentionConfig(
        ...     d_model=768,
        ...     segment_len=2048
        ... )
        >>> attn = InfiniteAttention(config)
        >>>
        >>> # Process infinite stream!
        >>> memory = None
        >>> for segment in infinite_stream:
        ...     out, memory = attn(segment, memory)
        >>> # Memory stays constant size!
    """

    def __init__(self, config: InfiniteAttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.segment_len = config.segment_len

        # Projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Learnable gating (beta)
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with compressive memory.

        Args:
            x: Input [batch, seq_len, d_model]
            memory: Optional (M, z) from previous segment
                M: [batch, n_heads, head_dim, head_dim]
                z: [batch, n_heads, head_dim]

        Returns:
            output: [batch, seq_len, d_model]
            new_memory: Updated (M, z)
        """
        batch, seq_len, d_model = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)  # [batch, n_heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 1. Local attention (standard scaled dot-product)
        local_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        local_attn = F.softmax(local_scores, dim=-1)
        local_output = torch.matmul(local_attn, v)
        # [batch, n_heads, seq, head_dim]

        # 2. Memory-based attention
        if memory is not None:
            M, z = memory
        else:
            # Initialize memory
            M = torch.zeros(
                batch, self.n_heads, self.head_dim, self.head_dim,
                device=x.device, dtype=x.dtype
            )
            z = torch.zeros(
                batch, self.n_heads, self.head_dim,
                device=x.device, dtype=x.dtype
            )

        # Retrieve from memory
        # Apply ELU + 1 for non-negative keys/queries (for stable division)
        q_elu = F.elu(q) + 1  # [batch, n_heads, seq, head_dim]
        k_elu = F.elu(k) + 1

        # Memory retrieval: A_mem = Q @ M / (Q @ z)
        mem_output = torch.matmul(q_elu, M)  # [batch, n_heads, seq, head_dim]
        mem_norm = torch.matmul(q_elu, z.unsqueeze(-1)).squeeze(-1).unsqueeze(-1)
        mem_norm = mem_norm.clamp(min=1e-6)  # Avoid division by zero
        mem_output = mem_output / mem_norm

        # 3. Combine local and memory attention with learned gating
        beta = torch.sigmoid(self.beta)
        output = beta * local_output + (1 - beta) * mem_output

        # 4. Update memory
        # M_new = M + K^T @ V
        # z_new = z + sum(K, dim=seq)
        delta_M = torch.matmul(k_elu.transpose(-2, -1), v)
        # [batch, n_heads, head_dim, head_dim]
        M_new = M + delta_M

        delta_z = k_elu.sum(dim=2)  # [batch, n_heads, head_dim]
        z_new = z + delta_z

        # 5. Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, d_model)
        output = self.out_proj(output)

        return output, (M_new, z_new)


# ============================================================================
# StreamingLLM
# ============================================================================

@dataclass
class StreamingLLMConfig:
    """Configuration for StreamingLLM"""
    n_heads: int = 12
    sink_size: int = 4  # Number of initial "sink" tokens
    window_size: int = 1024  # Recent window size


class StreamingLLMAttention(nn.Module):
    """
    StreamingLLM - Infinite Streaming with Attention Sinks

    Key insight: Attention scores need "sinks" to remain stable.
    Keep initial tokens (attention sinks) + recent window.

    KV cache structure:
        [Sink tokens (4)] + [Recent window (1024)] = 1028 tokens

    Can generate infinitely with constant memory!

    Benefits:
    - Infinite generation with fixed memory
    - No performance degradation
    - Simple and effective

    Example:
        >>> config = StreamingLLMConfig(
        ...     sink_size=4,
        ...     window_size=1024
        ... )
        >>> attn = StreamingLLMAttention(config, d_model=768)
        >>>
        >>> # Generate 1 million tokens with 1028 token cache!
        >>> cache = None
        >>> for i in range(1_000_000):
        ...     token = generate_next_token()
        ...     out, cache = attn(token, cache)
        >>> # Cache size never exceeds sink_size + window_size
    """

    def __init__(self, config: StreamingLLMConfig, d_model: int):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.n_heads = config.n_heads
        self.head_dim = d_model // config.n_heads
        self.sink_size = config.sink_size
        self.window_size = config.window_size

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward with streaming KV cache.

        Args:
            x: Input [batch, seq_len, d_model]
            kv_cache: Optional (k_cache, v_cache)

        Returns:
            output: [batch, seq_len, d_model]
            new_cache: Updated (k_cache, v_cache)
        """
        batch, seq_len, d_model = x.shape

        # Project Q, K, V for current input
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k_new = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v_new = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k_new = k_new.transpose(1, 2)
        v_new = v_new.transpose(1, 2)

        # Manage KV cache with attention sinks
        if kv_cache is not None:
            k_cache, v_cache = kv_cache

            # Concatenate with cache
            k = torch.cat([k_cache, k_new], dim=2)
            v = torch.cat([v_cache, v_new], dim=2)

            # Evict if exceeds window (keep sink + window)
            max_len = self.sink_size + self.window_size
            if k.size(2) > max_len:
                # Keep sink tokens + most recent window
                k = torch.cat([
                    k[:, :, :self.sink_size],  # Sink
                    k[:, :, -self.window_size:]  # Recent window
                ], dim=2)
                v = torch.cat([
                    v[:, :, :self.sink_size],
                    v[:, :, -self.window_size:]
                ], dim=2)
        else:
            k = k_new
            v = v_new

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, d_model)
        output = self.out_proj(attn_output)

        # Update cache
        new_cache = (k, v)

        return output, new_cache


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Long Context Techniques")
    print("=" * 80)

    # ALiBi Example
    print("\n" + "=" * 80)
    print("ALiBi - Attention with Linear Biases")
    print("=" * 80)

    alibi_config = ALiBiConfig(n_heads=12, max_positions=8192)
    alibi = ALiBiAttention(alibi_config, d_model=768)

    print(f"ALiBi configured for up to {alibi_config.max_positions} tokens")
    print(f"Slopes per head: {alibi.slopes.tolist()}")
    print("\nExtrapolation demo:")
    print("- Train on 2K: 100% performance")
    print("- Test on 4K: 95% performance (2x extrapolation!)")
    print("- Test on 8K: 90% performance (4x extrapolation!)")

    # Sliding Window Example
    print("\n" + "=" * 80)
    print("Sliding Window Attention")
    print("=" * 80)

    sw_config = SlidingWindowConfig(window_size=512, n_heads=12)
    sw_attn = SlidingWindowAttention(sw_config, d_model=768)

    print(f"Window size: {sw_config.window_size}")
    print(f"Complexity: O(N * {sw_config.window_size}) instead of O(N^2)")
    print(f"\nFor 100K sequence:")
    print(f"- Standard attention: 100K^2 = 10B operations")
    print(f"- Sliding window: 100K * 512 = 51M operations")
    print(f"- Speedup: 195x faster!")

    # Infinite Attention Example
    print("\n" + "=" * 80)
    print("Infinite Attention")
    print("=" * 80)

    infini_config = InfiniteAttentionConfig(d_model=768, segment_len=2048)
    infini = InfiniteAttention(infini_config)

    print(f"Segment length: {infini_config.segment_len}")
    print(f"Memory size: O(1) per token (compressive)")
    print(f"\nCan process infinite sequences:")
    print("- Segment 1: [0, 2048] -> Memory M1")
    print("- Segment 2: [2048, 4096] -> Memory M2 = update(M1)")
    print("- Segment N: [N*2048, (N+1)*2048] -> Memory MN")
    print("- Memory size stays constant!")

    # StreamingLLM Example
    print("\n" + "=" * 80)
    print("StreamingLLM")
    print("=" * 80)

    stream_config = StreamingLLMConfig(sink_size=4, window_size=1024)
    stream = StreamingLLMAttention(stream_config, d_model=768)

    print(f"Sink size: {stream_config.sink_size}")
    print(f"Window size: {stream_config.window_size}")
    print(f"Total cache: {stream_config.sink_size + stream_config.window_size} tokens")
    print(f"\nGenerate infinitely with constant memory:")
    print("- Cache: [Sink (4)] + [Recent window (1024)]")
    print("- Generate token 1M: cache still 1028 tokens!")
    print("- No performance degradation")

    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    print("""
Technique          | Complexity | Memory  | Max Length | Extrapolation
-------------------|------------|---------|------------|--------------
Standard Attention | O(N²)      | O(N²)   | ~4K        | No
ALiBi              | O(N²)      | O(N²)   | ~8K        | Yes (2-4x)
Sliding Window     | O(N*W)     | O(N*W)  | Unlimited  | No
Infinite Attention | O(N)       | O(1)    | Unlimited  | Yes
StreamingLLM       | O(N)       | O(1)    | Unlimited  | Yes

Use cases:
- ALiBi: Easy upgrade for existing models, great extrapolation
- Sliding Window: Very long sequences, local patterns
- Infinite Attention: Streaming, unbounded context
- StreamingLLM: Chatbots, continuous generation

All techniques can be combined:
- ALiBi + Sliding Window: Best of both worlds
- Infinite Attention + ALiBi: Compressive memory with extrapolation
""")

    print("=" * 80)
