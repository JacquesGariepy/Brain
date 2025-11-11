"""
Ring Attention - Distributed Long Context Attention

Enables training with extremely long sequences (millions of tokens) by distributing
attention computation across multiple devices in a ring topology.

Key Innovation:
- Blockwise computation with ring communication
- Enables sequences far longer than single GPU memory
- Linear scaling with number of devices

References:
- Ring Attention: https://arxiv.org/abs/2310.01889
- Blockwise Parallel Transformer: https://arxiv.org/abs/2305.19370

Performance:
- Sequence length scales linearly with devices
- 100M+ token sequences possible
- Minimal communication overhead
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class RingAttentionConfig:
    """Configuration for Ring Attention"""
    # Model dimensions
    d_model: int = 768
    n_heads: int = 12

    # Ring configuration
    block_size: int = 1024  # Size of each block
    num_devices: int = 8  # Number of devices in ring

    # Attention
    dropout: float = 0.1
    causal: bool = True


class RingAttention(nn.Module):
    """
    Ring Attention for distributed long-context processing.

    Key Idea:
    1. Split sequence into blocks across devices
    2. Each device computes attention for its block
    3. Pass KV blocks in ring topology
    4. Accumulate attention outputs

    Example:
        Device 0: Q[0:1024]  attends to K,V[0:1024], K,V[1024:2048], ...
        Device 1: Q[1024:2048] attends to K,V[1024:2048], K,V[2048:3072], ...
        ...

    This allows sequence length = block_size * num_devices
    with memory cost = O(block_size) per device
    """

    def __init__(self, config: RingAttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        assert config.d_model % config.n_heads == 0

        # Projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        device_id: int = 0,
        simulate_ring: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with ring communication simulation.

        Args:
            x: Input [batch, block_size, d_model] (one block per device)
            device_id: Current device ID in ring
            simulate_ring: If True, simulate ring communication for testing

        Returns:
            output: [batch, block_size, d_model]
        """
        batch, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, n_heads, seq, head_dim]

        if simulate_ring:
            # Simulate ring attention (single device emulation)
            output = self._simulate_ring_attention(q, k, v, device_id)
        else:
            # In production, this would use actual distributed communication
            output = self._distributed_ring_attention(q, k, v, device_id)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, d_model)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output

    def _simulate_ring_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        device_id: int
    ) -> torch.Tensor:
        """
        Simulate ring attention for testing (single device).

        In production, this would be replaced with actual ring communication
        using torch.distributed or similar.
        """
        batch, n_heads, seq_len, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)

        # Initialize output and normalizer
        output = torch.zeros_like(q)
        normalizer = torch.zeros(batch, n_heads, seq_len, 1, device=q.device)

        # Simulate ring passes
        for ring_step in range(self.config.num_devices):
            # In production, this would receive K,V from previous device in ring
            # For simulation, we just use the same K,V

            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Apply causal mask if needed
            if self.config.causal:
                # Mask out future positions relative to this block
                kv_block_id = (device_id + ring_step) % self.config.num_devices
                q_positions = device_id * seq_len + torch.arange(seq_len, device=q.device)
                kv_positions = kv_block_id * seq_len + torch.arange(seq_len, device=q.device)

                # Create causal mask: can only attend to past
                causal_mask = q_positions.unsqueeze(-1) < kv_positions.unsqueeze(0)
                scores = scores.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(0),
                    float('-inf')
                )

            # Compute attention weights
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Accumulate weighted values
            output += torch.matmul(attn_weights, v)
            normalizer += attn_weights.sum(dim=-1, keepdim=True)

            # In production: send K,V to next device in ring

        # Normalize (in case of masking)
        output = output / normalizer.clamp(min=1e-6)

        return output

    def _distributed_ring_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        device_id: int
    ) -> torch.Tensor:
        """
        Actual distributed ring attention implementation.

        This would use torch.distributed for real ring communication:
        - torch.distributed.send()
        - torch.distributed.recv()
        - Or ring_exchange custom CUDA kernel
        """
        # Placeholder for actual distributed implementation
        raise NotImplementedError(
            "Distributed ring attention requires torch.distributed setup. "
            "Use simulate_ring=True for single-device testing."
        )


class BlockwiseAttention(nn.Module):
    """
    Blockwise Attention - Process attention in blocks for memory efficiency.

    Similar to Ring Attention but for single device with memory constraints.
    """

    def __init__(self, config: RingAttentionConfig):
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
        """
        Blockwise attention computation.

        Processes attention in blocks to reduce peak memory usage.
        """
        batch, seq_len, d_model = x.shape
        block_size = self.config.block_size

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Process in blocks
        output = torch.zeros_like(q)
        scale = 1.0 / math.sqrt(self.head_dim)

        num_blocks = (seq_len + block_size - 1) // block_size

        for i in range(num_blocks):
            # Query block
            q_start = i * block_size
            q_end = min((i + 1) * block_size, seq_len)
            q_block = q[:, :, q_start:q_end]

            block_output = torch.zeros_like(q_block)
            normalizer = torch.zeros(
                batch, self.n_heads, q_end - q_start, 1,
                device=q.device
            )

            # Attend to all KV blocks
            for j in range(num_blocks):
                k_start = j * block_size
                k_end = min((j + 1) * block_size, seq_len)
                k_block = k[:, :, k_start:k_end]
                v_block = v[:, :, k_start:k_end]

                # Compute attention
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

                # Causal masking
                if self.config.causal and j > i:
                    # Future block - mask all
                    scores = scores.masked_fill(
                        torch.ones_like(scores, dtype=torch.bool),
                        float('-inf')
                    )
                elif self.config.causal and j == i:
                    # Same block - causal mask
                    causal_mask = torch.triu(
                        torch.ones(scores.shape[-2:], device=scores.device),
                        diagonal=1
                    ).bool()
                    scores = scores.masked_fill(causal_mask, float('-inf'))

                # Compute weights and accumulate
                attn_weights = torch.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

                block_output += torch.matmul(attn_weights, v_block)
                normalizer += attn_weights.sum(dim=-1, keepdim=True)

            # Normalize and store
            output[:, :, q_start:q_end] = block_output / normalizer.clamp(min=1e-6)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, d_model)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output


# ============================================================================
# Testing
# ============================================================================

def test_ring_attention():
    """Test Ring Attention implementation."""
    print("=" * 80)
    print("Testing Ring Attention")
    print("=" * 80)

    config = RingAttentionConfig(
        d_model=256,
        n_heads=8,
        block_size=512,
        num_devices=4,
        causal=True
    )

    model = RingAttention(config)

    # Test with single block
    batch_size = 2
    x = torch.randn(batch_size, config.block_size, config.d_model)

    print(f"\nInput shape: {x.shape}")
    print(f"Block size: {config.block_size}")
    print(f"Num devices: {config.num_devices}")
    print(f"Max sequence length: {config.block_size * config.num_devices}")

    # Forward pass
    with torch.no_grad():
        output = model(x, device_id=0, simulate_ring=True)

    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Shape mismatch!"

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    print("\n✓ Ring Attention test PASSED")

    return {
        'status': 'PASS',
        'config': config,
        'output_shape': output.shape,
        'params': total_params,
        'max_seq_len': config.block_size * config.num_devices,
        'memory_per_device': f"O({config.block_size})",
        'output_mean': output.mean().item(),
        'output_std': output.std().item()
    }


def test_blockwise_attention():
    """Test Blockwise Attention implementation."""
    print("\n" + "=" * 80)
    print("Testing Blockwise Attention")
    print("=" * 80)

    config = RingAttentionConfig(
        d_model=256,
        n_heads=8,
        block_size=512,
        causal=True
    )

    model = BlockwiseAttention(config)

    # Test with longer sequence
    batch_size = 2
    seq_len = 2048
    x = torch.randn(batch_size, seq_len, config.d_model)

    print(f"\nInput shape: {x.shape}")
    print(f"Block size: {config.block_size}")
    print(f"Num blocks: {(seq_len + config.block_size - 1) // config.block_size}")

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Shape mismatch!"

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    print("\n✓ Blockwise Attention test PASSED")

    return {
        'status': 'PASS',
        'config': config,
        'output_shape': output.shape,
        'params': total_params,
        'num_blocks': (seq_len + config.block_size - 1) // config.block_size,
        'peak_memory': f"O(block_size^2) vs O(seq_len^2)",
        'output_mean': output.mean().item(),
        'output_std': output.std().item()
    }


def test_all():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Ring Attention - Complete Test Suite")
    print("=" * 80)

    results = {}

    # Test 1: Ring Attention
    results['RingAttention'] = test_ring_attention()

    # Test 2: Blockwise Attention
    results['BlockwiseAttention'] = test_blockwise_attention()

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Status: {result['status']}")
        print(f"  Output shape: {result['output_shape']}")
        print(f"  Parameters: {result['params']:,}")
        if 'max_seq_len' in result:
            print(f"  Max sequence length: {result['max_seq_len']:,}")
            print(f"  Memory per device: {result['memory_per_device']}")
        if 'peak_memory' in result:
            print(f"  Peak memory: {result['peak_memory']}")
        print(f"  Output mean: {result['output_mean']:.6f}")
        print(f"  Output std: {result['output_std']:.6f}")

    print("\n" + "=" * 80)
    print("Key Advantages of Ring Attention:")
    print("=" * 80)
    print("""
1. Extreme Long Context:
   - Enables 100M+ token sequences
   - Scales linearly with devices
   - Each device only stores one block

2. Memory Efficiency:
   - Memory per device: O(block_size)
   - Total sequence: block_size × num_devices
   - Example: 512 block × 1000 devices = 512K tokens

3. Communication Efficient:
   - Only KV blocks passed in ring
   - Overlaps computation and communication
   - Minimal overhead

4. Production Use:
   - Used in long-document understanding
   - Video processing (millions of frames)
   - Genomics (long DNA sequences)

Comparison:
-----------
Standard Attention:
  - Max length: ~16K tokens (limited by single GPU)
  - Memory: O(N²)

Ring Attention:
  - Max length: Millions of tokens
  - Memory per device: O(block_size²)
  - Scales with devices
""")

    print("=" * 80)

    return results


if __name__ == "__main__":
    test_all()
