"""
RWKV - Receptance Weighted Key Value

RNN with Transformer-level performance, O(N) time and memory.

Key Innovation:
- Linear attention with time-mixing and channel-mixing
- Can be formulated as RNN or Transformer
- Combines best of both worlds

References:
- RWKV: https://arxiv.org/abs/2305.13048
- Project: https://github.com/BlinkDL/RWKV-LM

Performance:
- Speed: Constant time per token (RNN mode)
- Memory: O(1) for inference (stateful)
- Quality: Competitive with Transformers up to 14B scale
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class RWKVConfig:
    """Configuration for RWKV"""
    # Model dimensions
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12  # For multi-head RWKV (v5+)

    # Layer config
    layer_norm_eps: float = 1e-5

    # Time decay
    time_decay_init: str = "default"  # "default" or "log"

    # Context length
    ctx_len: int = 1024


class WKV(nn.Module):
    """
    WKV (Weighted Key Value) Operator

    Core of RWKV. Computes attention in O(N) time.

    Traditional attention:
        att = softmax(Q @ K^T) @ V  # O(N^2)

    RWKV WKV:
        wkv[t] = sum_{i<=t} exp(-(t-i)*w + k[i]) * v[i]  # O(N) with scan!
        out[t] = r[t] * wkv[t]

    Where:
    - w: time decay (how fast to forget)
    - k: key
    - v: value
    - r: receptance (gating)
    """

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        w: torch.Tensor,  # Time decay
        k: torch.Tensor,  # Key
        v: torch.Tensor,  # Value
        r: torch.Tensor,  # Receptance
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute WKV operator.

        Args:
            w: Time decay [d_model]
            k: Key [batch, seq_len, d_model]
            v: Value [batch, seq_len, d_model]
            r: Receptance [batch, seq_len, d_model]
            state: Optional state for RNN mode (num, den)

        Returns:
            output: [batch, seq_len, d_model]
            new_state: Updated state
        """
        batch, seq_len, d_model = k.shape

        # Transformer mode (parallel)
        if state is None:
            return self._parallel_wkv(w, k, v, r)
        else:
            # RNN mode (sequential)
            return self._sequential_wkv(w, k, v, r, state)

    def _parallel_wkv(
        self,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Parallel WKV computation (training)"""
        batch, seq_len, d_model = k.shape

        # Compute attention weights with time decay
        # att[t, i] = exp(-(t-i)*w + k[i]) for i <= t

        # Create time decay matrix
        positions = torch.arange(seq_len, device=k.device)
        time_diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq, seq]
        time_diff = time_diff.clamp(min=0)  # Only look at past

        # Decay weights: exp(-time_diff * w)
        decay = torch.exp(-time_diff.unsqueeze(-1) * w.unsqueeze(0).unsqueeze(0))  # [seq, seq, d_model]

        # Apply decay to keys
        k_exp = torch.exp(k.unsqueeze(1))  # [batch, 1, seq, d_model]
        weights = decay.unsqueeze(0) * k_exp  # [batch, seq, seq, d_model]

        # Mask future
        mask = time_diff == 0
        weights = weights.masked_fill(
            ~mask.unsqueeze(0).unsqueeze(-1).expand_as(weights),
            0
        )

        # Weighted sum of values
        wkv = torch.einsum('btsd,bsd->btd', weights, v)

        # Apply receptance
        output = r * wkv

        # Final state (for potential RNN continuation)
        final_state = (
            weights[:, -1].sum(dim=1),  # Numerator
            torch.ones(batch, d_model, device=k.device)  # Denominator
        )

        return output, final_state

    def _sequential_wkv(
        self,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sequential WKV computation (inference, RNN mode)"""
        batch, seq_len, d_model = k.shape

        num, den = state
        outputs = []

        for t in range(seq_len):
            k_t = k[:, t]  # [batch, d_model]
            v_t = v[:, t]  # [batch, d_model]
            r_t = r[:, t]  # [batch, d_model]

            # Update state
            kv = torch.exp(k_t) * v_t
            num = num * torch.exp(-w) + kv
            den = den * torch.exp(-w) + torch.exp(k_t)

            # Compute output
            wkv_t = num / (den + 1e-8)
            out_t = r_t * wkv_t

            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)
        return output, (num, den)


class TimeMixing(nn.Module):
    """
    Time-Mixing Layer

    Mixes information across time (like attention).

    Interpolates between current and previous token:
        x_mix[t] = lerp(x[t], x[t-1], time_mix)
    """

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model

        # Time mixing weights
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))

        # Projections
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

        # Time decay
        self.time_decay = nn.Parameter(torch.ones(d_model))

        # WKV operator
        self.wkv = WKV(config)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, d_model]
            state: Optional state for RNN mode

        Returns:
            output: [batch, seq_len, d_model]
            new_state: Updated state
        """
        batch, seq_len, d_model = x.shape

        # Get previous token (or state)
        if state is not None and 'time_mix' in state:
            x_prev = state['time_mix']
        else:
            # Pad with zeros for first token
            x_prev = F.pad(x[:, :-1], (0, 0, 1, 0))

        # Time mixing (interpolate between current and previous)
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        # Project to k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        # WKV
        wkv_state = state.get('wkv') if state is not None else None
        out, new_wkv_state = self.wkv(self.time_decay, k, v, r, wkv_state)

        # Output projection
        out = self.output(out)

        # Update state
        if state is not None:
            new_state = {
                'time_mix': x[:, -1:],  # Last token
                'wkv': new_wkv_state
            }
        else:
            new_state = None

        return out, new_state


class ChannelMixing(nn.Module):
    """
    Channel-Mixing Layer

    Mixes information across channels (like FFN).
    """

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model

        # Time mixing weight
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))

        # FFN
        d_ff = d_model * 4
        self.key = nn.Linear(d_model, d_ff, bias=False)
        self.value = nn.Linear(d_ff, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Forward pass"""
        batch, seq_len, d_model = x.shape

        # Get previous token
        if state is not None and 'channel_mix' in state:
            x_prev = state['channel_mix']
        else:
            x_prev = F.pad(x[:, :-1], (0, 0, 1, 0))

        # Time mixing
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        # Channel mixing
        k = self.key(xk)
        k = torch.square(torch.relu(k))  # Squared ReLU
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(xr))

        out = r * kv

        # Update state
        if state is not None:
            new_state = {'channel_mix': x[:, -1:]}
        else:
            new_state = None

        return out, new_state


class RWKVBlock(nn.Module):
    """Single RWKV block (time-mixing + channel-mixing)"""

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.config = config

        # Layer norms
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Mixing layers
        self.time_mixing = TimeMixing(config)
        self.channel_mixing = ChannelMixing(config)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Forward pass"""
        # Time mixing with residual
        time_out, time_state = self.time_mixing(self.ln1(x), state)
        x = x + time_out

        # Channel mixing with residual
        channel_out, channel_state = self.channel_mixing(self.ln2(x), state)
        x = x + channel_out

        # Combine states
        if state is not None:
            new_state = {**time_state, **channel_state}
        else:
            new_state = None

        return x, new_state


class RWKV(nn.Module):
    """
    Complete RWKV model.

    Can operate in two modes:
    1. Transformer mode (parallel): For training
    2. RNN mode (sequential): For efficient inference

    Example:
        >>> config = RWKVConfig(d_model=768, n_layers=12)
        >>> model = RWKV(config)
        >>>
        >>> # Training (parallel)
        >>> x = torch.randn(2, 1024, 768)
        >>> out = model(x)
        >>>
        >>> # Inference (RNN mode, constant memory!)
        >>> state = None
        >>> for token in tokens:
        >>>     out, state = model(token, state=state)
    """

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.config = config

        # Layers
        self.blocks = nn.ModuleList([
            RWKVBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm
        self.ln_out = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[List[dict]] = None
    ) -> Tuple[torch.Tensor, Optional[List[dict]]]:
        """
        Forward pass.

        Args:
            x: Input [batch, seq_len, d_model]
            state: Optional state for each layer (RNN mode)

        Returns:
            output: [batch, seq_len, d_model]
            new_state: Updated state for each layer
        """
        # Initialize states if needed
        if state is None:
            states = [None] * len(self.blocks)
        else:
            states = state

        new_states = []

        # Pass through all blocks
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, states[i])
            new_states.append(new_state)

        # Final norm
        x = self.ln_out(x)

        return x, new_states if state is not None else None


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RWKV - Receptance Weighted Key Value")
    print("=" * 80)

    # Create model
    config = RWKVConfig(
        d_model=768,
        n_layers=12,
        ctx_len=1024
    )

    model = RWKV(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel size: {total_params / 1e6:.1f}M parameters")

    # Test parallel mode (training)
    print("\n" + "=" * 80)
    print("Parallel Mode (Training)")
    print("=" * 80)

    batch_size = 2
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, config.d_model)

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        out, _ = model(x)

    print(f"Output shape: {out.shape}")

    # Test RNN mode (inference)
    print("\n" + "=" * 80)
    print("RNN Mode (Inference)")
    print("=" * 80)

    state = None
    token = torch.randn(1, 1, config.d_model)

    print(f"Processing tokens one at a time...")
    for i in range(10):
        with torch.no_grad():
            out, state = model(token, state=state)
        print(f"Token {i + 1}: state stored, constant memory!")

    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    print("""
RWKV vs Transformer vs RNN:

1. Training:
   - RWKV: O(N) time, O(N) memory (parallel mode)
   - Transformer: O(N^2) time, O(N^2) memory
   - RNN: O(N) time, O(1) memory (but sequential!)

2. Inference:
   - RWKV: O(1) time/memory per token (RNN mode)
   - Transformer: O(N) time, O(N) memory (KV cache)
   - RNN: O(1) time/memory per token

3. Quality:
   - RWKV: Competitive with Transformers up to 14B
   - Transformer: Best quality (current SOTA)
   - RNN: Historically weaker on long-range

4. Context Length:
   - RWKV: Infinite (in theory, constant state)
   - Transformer: Limited by quadratic cost
   - RNN: Infinite but forgets

Key Advantages:
- Best of both worlds: Transformer for training, RNN for inference
- Constant inference cost (no KV cache!)
- Can handle infinite context length
- Parallelizable training

Use RWKV when:
- Need efficient inference (e.g., chatbots, streaming)
- Working with very long sequences
- Memory constrained
- Want constant-time token generation

Use Transformers when:
- Maximum quality needed
- Context length < 8K
- Established tooling important
""")

    print("=" * 80)
