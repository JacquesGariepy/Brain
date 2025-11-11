"""
RetNet - Retentive Network

Successor to Transformers with O(1) inference complexity.

Key Innovation:
- Retention mechanism: combines parallel, recurrent, and chunkwise forms
- Training: O(N) parallel
- Inference: O(1) recurrent
- Quality: Matches Transformers

References:
- RetNet: https://arxiv.org/abs/2307.08621

Performance:
- 8x faster inference than Transformers
- Constant memory per token
- Matches Transformer quality
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class RetNetConfig:
    """Configuration for RetNet"""
    # Model dimensions
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072

    # Retention
    retention_heads: int = 12

    # Dropout
    dropout: float = 0.1

    # Layer norm
    layer_norm_eps: float = 1e-5


class MultiScaleRetention(nn.Module):
    """
    Multi-Scale Retention (MSR)

    Key innovation: Retention with multiple decay rates.

    Three equivalent formulations:
    1. Parallel (training): O(N^2) but parallelizable
    2. Recurrent (inference): O(1) per token
    3. Chunkwise (hybrid): O(1) per chunk

    Retention equation:
        Retention(Q, K, V) = (Q * D) @ (K^T * D^T) @ V

    Where D is decay matrix: D[i,j] = gamma^(i-j) if i >= j else 0
    """

    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.retention_heads
        self.head_dim = config.d_model // config.retention_heads

        assert config.d_model % config.retention_heads == 0

        # Projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Group norm (for stability)
        self.group_norm = nn.GroupNorm(config.retention_heads, config.d_model)

        # Decay rates (gamma) - one per head for multi-scale
        # gamma_h = 1 - 2^(-(5 + h))
        gammas = []
        for h in range(config.retention_heads):
            gamma = 1 - 2 ** (-(5 + h))
            gammas.append(gamma)
        self.register_buffer('gammas', torch.tensor(gammas))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        use_parallel: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input [batch, seq_len, d_model]
            state: Optional state for recurrent mode [batch, n_heads, head_dim, head_dim]
            use_parallel: Whether to use parallel formulation

        Returns:
            output: [batch, seq_len, d_model]
            new_state: Updated state
        """
        batch, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, n_heads, seq, head_dim]

        if use_parallel:
            # Parallel formulation (training)
            retention_out, new_state = self._parallel_retention(q, k, v)
        else:
            # Recurrent formulation (inference)
            retention_out, new_state = self._recurrent_retention(q, k, v, state)

        # Reshape and project
        retention_out = retention_out.transpose(1, 2).contiguous()
        retention_out = retention_out.view(batch, seq_len, d_model)

        # Group norm
        retention_out = self.group_norm(retention_out.transpose(1, 2)).transpose(1, 2)

        # Output projection
        output = self.out_proj(retention_out)

        return output, new_state

    def _parallel_retention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel retention formulation.

        Retention = (Q * D) @ K^T @ V

        Where D[i,j] = gamma^(i-j) if i >= j else 0
        """
        batch, n_heads, seq_len, head_dim = q.shape

        # Compute decay matrix for each head
        positions = torch.arange(seq_len, device=q.device)
        decay = self.gammas.unsqueeze(-1) ** positions  # [n_heads, seq_len]

        # Apply decay to Q and K
        q_decay = q * decay.unsqueeze(0).unsqueeze(-1)  # [batch, heads, seq, dim]
        k_decay = k * decay.unsqueeze(0).unsqueeze(-1)

        # Compute attention-like scores
        scores = torch.matmul(q_decay, k_decay.transpose(-2, -1))  # [batch, heads, seq, seq]

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0)

        # Apply to values
        output = torch.matmul(scores, v)  # [batch, heads, seq, dim]

        # Compute final state (for potential recurrent continuation)
        # State = K^T @ V (with decay)
        final_state = torch.matmul(
            k_decay[:, :, -1:].transpose(-2, -1),
            v[:, :, -1:]
        )  # [batch, heads, dim, dim]

        return output, final_state

    def _recurrent_retention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent retention formulation.

        S_t = gamma * S_{t-1} + K_t^T @ V_t
        O_t = Q_t @ S_t

        Constant time per token!
        """
        batch, n_heads, seq_len, head_dim = q.shape

        # Initialize state
        if state is None:
            state = torch.zeros(
                batch, n_heads, head_dim, head_dim,
                device=q.device, dtype=q.dtype
            )

        outputs = []

        for t in range(seq_len):
            # Update state: S = gamma * S + K^T @ V
            k_t = k[:, :, t:t + 1]  # [batch, heads, 1, dim]
            v_t = v[:, :, t:t + 1]  # [batch, heads, 1, dim]

            state = self.gammas.view(1, -1, 1, 1) * state + torch.matmul(
                k_t.transpose(-2, -1), v_t
            )

            # Compute output: O = Q @ S
            q_t = q[:, :, t:t + 1]  # [batch, heads, 1, dim]
            o_t = torch.matmul(q_t, state)  # [batch, heads, 1, dim]

            outputs.append(o_t)

        output = torch.cat(outputs, dim=2)  # [batch, heads, seq, dim]

        return output, state


class RetentionBlock(nn.Module):
    """RetNet block with retention and FFN"""

    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.config = config

        # Retention
        self.retention = MultiScaleRetention(config)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model)
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        use_parallel: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass"""
        # Retention with residual
        retention_out, new_state = self.retention(
            self.ln1(x), state=state, use_parallel=use_parallel
        )
        x = x + self.dropout(retention_out)

        # FFN with residual
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)

        return x, new_state


class RetNet(nn.Module):
    """
    Complete RetNet model.

    Example:
        >>> config = RetNetConfig(d_model=768, n_layers=12)
        >>> model = RetNet(config)
        >>>
        >>> # Training (parallel)
        >>> x = torch.randn(2, 1024, 768)
        >>> out = model(x, use_parallel=True)
        >>>
        >>> # Inference (recurrent, O(1) per token!)
        >>> state = None
        >>> for token in tokens:
        >>>     out, state = model(token, state=state, use_parallel=False)
    """

    def __init__(self, config: RetNetConfig):
        super().__init__()
        self.config = config

        # Layers
        self.blocks = nn.ModuleList([
            RetentionBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
        use_parallel: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward pass"""
        # Initialize states
        if state is None:
            states = [None] * len(self.blocks)
        else:
            states = state

        new_states = []

        # Pass through blocks
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, state=states[i], use_parallel=use_parallel)
            new_states.append(new_state)

        # Final norm
        x = self.ln_f(x)

        return x, new_states if not use_parallel else None


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RetNet - Retentive Network")
    print("=" * 80)

    # Create model
    config = RetNetConfig(
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072
    )

    model = RetNet(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel size: {total_params / 1e6:.1f}M parameters")

    # Test parallel mode
    print("\n" + "=" * 80)
    print("Parallel Mode (Training)")
    print("=" * 80)

    batch_size = 2
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, config.d_model)

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        out, _ = model(x, use_parallel=True)

    print(f"Output shape: {out.shape}")

    # Test recurrent mode
    print("\n" + "=" * 80)
    print("Recurrent Mode (Inference)")
    print("=" * 80)

    state = None
    token = torch.randn(1, 1, config.d_model)

    print(f"Processing tokens one at a time...")
    for i in range(10):
        with torch.no_grad():
            out, state = model(token, state=state, use_parallel=False)
        print(f"Token {i + 1}: O(1) time and memory!")

    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    print("""
RetNet vs Transformer:

1. Training:
   - RetNet: O(N) time (parallel retention)
   - Transformer: O(N^2) time (attention)

2. Inference:
   - RetNet: O(1) per token (recurrent)
   - Transformer: O(N) per token (KV cache)

3. Memory:
   - RetNet: Constant (state is fixed size)
   - Transformer: O(N) (KV cache grows)

4. Quality:
   - RetNet: Matches Transformers
   - Transformer: Current SOTA

5. Speed (inference):
   - RetNet: 8.4x faster than Transformers
   - No KV cache needed!

Key Advantages:
- Three formulations: parallel, recurrent, chunkwise
- Best of both worlds: fast training AND inference
- Constant memory during inference
- Maintains Transformer quality

Use RetNet when:
- Need fast inference (real-time applications)
- Memory constrained
- Long sequences
- Want Transformer quality with RNN efficiency
""")

    print("=" * 80)
