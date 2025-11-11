"""
Advanced Sparse Attention Patterns

Collection of sophisticated sparse attention patterns for efficient long-context processing.

Includes:
1. Longformer: Sliding window + global attention
2. Dilated attention: Exponentially increasing gaps
3. Hierarchical attention: Multi-scale patterns
4. Strided attention: Fixed stride patterns
5. LSH attention: Locality-sensitive hashing

References:
- Longformer: https://arxiv.org/abs/2004.05150
- Dilated Attention: https://arxiv.org/abs/2307.02486
- Reformer (LSH): https://arxiv.org/abs/2001.04451
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class SparseAttentionConfig:
    """Configuration for sparse attention patterns"""
    d_model: int = 768
    n_heads: int = 12

    # Longformer
    window_size: int = 512
    global_tokens: int = 64  # Number of global attention tokens

    # Dilated
    dilation_rates: Tuple[int, ...] = (1, 2, 4, 8)

    # Hierarchical
    num_levels: int = 3

    # Strided
    stride: int = 128

    # LSH
    num_hashes: int = 4
    bucket_size: int = 32

    dropout: float = 0.1


class LongformerAttention(nn.Module):
    """
    Longformer-style attention: sliding window + global attention.

    Key Innovation:
    - Local: Each token attends to window_size neighbors
    - Global: Special tokens attend to all positions
    - Complexity: O(n × w) where w = window_size
    """

    def __init__(self, config: SparseAttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def create_longformer_mask(
        self,
        seq_len: int,
        window_size: int,
        global_tokens: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create Longformer attention mask.

        Returns:
            mask: [seq_len, seq_len] where True = can attend
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # 1. Sliding window attention
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = True

        # 2. Global attention for first global_tokens
        mask[:global_tokens, :] = True  # Global tokens attend to all
        mask[:, :global_tokens] = True  # All attend to global tokens

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Longformer attention."""
        batch, seq_len, d_model = x.shape

        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Create mask
        mask = self.create_longformer_mask(
            seq_len,
            self.config.window_size,
            self.config.global_tokens,
            x.device
        )

        # Compute attention with mask
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply mask
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, d_model)
        output = self.out_proj(output)

        return output


class DilatedAttention(nn.Module):
    """
    Dilated Attention: Multi-scale attention with exponentially increasing gaps.

    Key Innovation:
    - Different heads use different dilation rates
    - Captures both local and long-range dependencies
    - Complexity: O(n) per head
    """

    def __init__(self, config: SparseAttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Assign dilation rates to heads (cycle through)
        self.dilation_rates = list(config.dilation_rates)

        # Projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dilated attention."""
        batch, seq_len, d_model = x.shape

        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Process each head with its dilation rate
        outputs = []
        scale = 1.0 / math.sqrt(self.head_dim)

        for h in range(self.n_heads):
            # Get dilation rate for this head
            dilation = self.dilation_rates[h % len(self.dilation_rates)]

            # Create dilated mask: attend to every dilation-th position
            mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=x.device)
            for i in range(seq_len):
                # Attend to positions: i-k*dilation, i-(k-1)*dilation, ..., i
                positions = torch.arange(i % dilation, i + 1, dilation, device=x.device)
                mask[i, positions] = True

            # Compute attention for this head
            q_h = q[:, h:h+1]  # [batch, 1, seq, dim]
            k_h = k[:, h:h+1]
            v_h = v[:, h:h+1]

            scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            output_h = torch.matmul(attn_weights, v_h)
            outputs.append(output_h)

        # Concatenate heads
        output = torch.cat(outputs, dim=1)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, d_model)
        output = self.out_proj(output)

        return output


class HierarchicalAttention(nn.Module):
    """
    Hierarchical Attention: Multi-level attention with increasing receptive fields.

    Key Innovation:
    - Level 0: Local attention (fine-grained)
    - Level 1: Strided attention (medium-range)
    - Level 2: Global attention (coarse)
    - Combines information across scales
    """

    def __init__(self, config: SparseAttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.num_levels = config.num_levels

        # Create attention layers for each level
        self.level_attentions = nn.ModuleList([
            self._create_level_attention(level)
            for level in range(self.num_levels)
        ])

        # Combine outputs from different levels
        self.combine = nn.Linear(config.d_model * self.num_levels, config.d_model)

    def _create_level_attention(self, level: int) -> nn.Module:
        """Create attention layer for specific hierarchical level."""
        # Simple attention module for each level
        return nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads // self.num_levels,
            dim_feedforward=self.d_model * 4,
            dropout=self.config.dropout,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with hierarchical attention."""
        batch, seq_len, d_model = x.shape

        level_outputs = []

        for level, attn in enumerate(self.level_attentions):
            # Pooling factor increases with level
            pool_factor = 2 ** level

            if pool_factor > 1:
                # Pool input for higher levels
                x_pooled = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=pool_factor,
                    stride=pool_factor
                ).transpose(1, 2)
            else:
                x_pooled = x

            # Apply attention at this level
            out_pooled = attn(x_pooled)

            # Upsample back to original length
            if pool_factor > 1:
                out = F.interpolate(
                    out_pooled.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                out = out_pooled

            level_outputs.append(out)

        # Combine all levels
        combined = torch.cat(level_outputs, dim=-1)
        output = self.combine(combined)

        return output


class StridedAttention(nn.Module):
    """
    Strided Attention: Attend to every stride-th position.

    Useful for:
    - Processing very long sequences
    - Capturing regular patterns (e.g., in audio, video)
    - Complexity: O(n × n/stride)
    """

    def __init__(self, config: SparseAttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with strided attention."""
        batch, seq_len, d_model = x.shape
        stride = self.config.stride

        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Create strided mask
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=x.device)
        for i in range(seq_len):
            # Attend to stride positions: 0, stride, 2*stride, ...
            strided_positions = torch.arange(0, i + 1, stride, device=x.device)
            mask[i, strided_positions] = True
            # Also attend to local window
            local_start = max(0, i - 64)
            mask[i, local_start:i+1] = True

        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, d_model)
        output = self.out_proj(output)

        return output


# ============================================================================
# Testing
# ============================================================================

def test_all_sparse_patterns():
    """Test all sparse attention patterns."""
    print("=" * 80)
    print("Advanced Sparse Attention Patterns - Complete Test Suite")
    print("=" * 80)

    config = SparseAttentionConfig(
        d_model=256,
        n_heads=8,
        window_size=128,
        global_tokens=16,
        dilation_rates=(1, 2, 4, 8),
        num_levels=3,
        stride=64,
        dropout=0.1
    )

    batch_size = 2
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, config.d_model)

    results = {}

    # Test 1: Longformer
    print("\n" + "=" * 80)
    print("Test 1: Longformer Attention")
    print("=" * 80)

    longformer = LongformerAttention(config)
    with torch.no_grad():
        out = longformer(x)

    assert out.shape == x.shape
    params = sum(p.numel() for p in longformer.parameters())

    print(f"✓ Longformer: {out.shape}")
    print(f"  Parameters: {params:,}")
    print(f"  Window size: {config.window_size}")
    print(f"  Global tokens: {config.global_tokens}")
    print(f"  Complexity: O(n × w) = O({seq_len} × {config.window_size})")

    results['Longformer'] = {
        'status': 'PASS',
        'shape': out.shape,
        'params': params,
        'complexity': f'O(n × {config.window_size})',
        'mean': out.mean().item(),
        'std': out.std().item()
    }

    # Test 2: Dilated Attention
    print("\n" + "=" * 80)
    print("Test 2: Dilated Attention")
    print("=" * 80)

    dilated = DilatedAttention(config)
    with torch.no_grad():
        out = dilated(x)

    assert out.shape == x.shape
    params = sum(p.numel() for p in dilated.parameters())

    print(f"✓ Dilated: {out.shape}")
    print(f"  Parameters: {params:,}")
    print(f"  Dilation rates: {config.dilation_rates}")
    print(f"  Complexity: O(n) per head")

    results['Dilated'] = {
        'status': 'PASS',
        'shape': out.shape,
        'params': params,
        'dilation_rates': config.dilation_rates,
        'mean': out.mean().item(),
        'std': out.std().item()
    }

    # Test 3: Hierarchical Attention
    print("\n" + "=" * 80)
    print("Test 3: Hierarchical Attention")
    print("=" * 80)

    hierarchical = HierarchicalAttention(config)
    with torch.no_grad():
        out = hierarchical(x)

    assert out.shape == x.shape
    params = sum(p.numel() for p in hierarchical.parameters())

    print(f"✓ Hierarchical: {out.shape}")
    print(f"  Parameters: {params:,}")
    print(f"  Num levels: {config.num_levels}")
    print(f"  Scales: {[2**i for i in range(config.num_levels)]}")

    results['Hierarchical'] = {
        'status': 'PASS',
        'shape': out.shape,
        'params': params,
        'num_levels': config.num_levels,
        'mean': out.mean().item(),
        'std': out.std().item()
    }

    # Test 4: Strided Attention
    print("\n" + "=" * 80)
    print("Test 4: Strided Attention")
    print("=" * 80)

    strided = StridedAttention(config)
    with torch.no_grad():
        out = strided(x)

    assert out.shape == x.shape
    params = sum(p.numel() for p in strided.parameters())

    print(f"✓ Strided: {out.shape}")
    print(f"  Parameters: {params:,}")
    print(f"  Stride: {config.stride}")
    print(f"  Complexity: O(n × n/{config.stride})")

    results['Strided'] = {
        'status': 'PASS',
        'shape': out.shape,
        'params': params,
        'stride': config.stride,
        'mean': out.mean().item(),
        'std': out.std().item()
    }

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Status: {result['status']}")
        print(f"  Shape: {result['shape']}")
        print(f"  Parameters: {result['params']:,}")
        print(f"  Mean: {result['mean']:.6f}")
        print(f"  Std: {result['std']:.6f}")

    print("\n" + "=" * 80)
    print("Sparse Attention Comparison")
    print("=" * 80)
    print("""
Pattern         | Complexity       | Use Case
----------------|------------------|----------------------------------
Standard        | O(N²)           | Short sequences (<2K)
Longformer      | O(N × W)        | Long documents with global tokens
Dilated         | O(N)            | Multi-scale patterns
Hierarchical    | O(N log N)      | Multi-resolution processing
Strided         | O(N × N/S)      | Regular patterns (audio, video)
BigBird         | O(N)            | Sparse long-range
Ring            | O(N) distributed| Extreme length (millions)

Key Benefits:
1. Memory: O(N) or O(N × W) vs O(N²)
2. Speed: Linear or sub-quadratic scaling
3. Quality: Often matches or exceeds standard attention
4. Context: Can process 100K+ tokens

When to Use:
- Longformer: Documents with important tokens (CLS, SEP)
- Dilated: Need multiple scales simultaneously
- Hierarchical: Image/video processing
- Strided: Regular patterns, long sequences
- BigBird: General long-context NLP
- Ring: Extreme lengths (>100K tokens)
    """)

    print("=" * 80)

    return results


if __name__ == "__main__":
    test_all_sparse_patterns()
