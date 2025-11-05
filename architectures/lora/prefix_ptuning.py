"""
Parameter-Efficient Fine-Tuning Methods

Beyond LoRA: Additional PEFT techniques:
1. Prefix Tuning: Prepend trainable prefix tokens
2. P-Tuning v2: Deep prompt tuning across all layers
3. Prompt Tuning: Simple learned prompt embeddings
4. Adapter Layers: Bottleneck adapters in transformers
5. BitFit: Bias-only fine-tuning

Key Benefits:
- Train <1% of parameters
- Similar performance to full fine-tuning
- Multiple tasks with same base model
- Reduced memory and compute

References:
- Prefix Tuning: https://arxiv.org/abs/2101.00190
- P-Tuning v2: https://arxiv.org/abs/2110.07602
- Prompt Tuning: https://arxiv.org/abs/2104.08691
- Adapter: https://arxiv.org/abs/1902.00751
"""

from dataclasses import dataclass
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Prefix Tuning
# ============================================================================

@dataclass
class PrefixTuningConfig:
    """Configuration for Prefix Tuning"""
    # Model
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12

    # Prefix
    prefix_length: int = 20  # Number of prefix tokens
    prefix_projection: bool = True  # Project prefix through MLP
    prefix_dropout: float = 0.1

    # Reparameterization
    prefix_hidden_size: int = 512  # MLP hidden size for prefix


class PrefixEncoder(nn.Module):
    """
    Prefix Encoder: Learns prefix representations.

    Key Innovation:
    - Instead of directly learning prefix embeddings, learn through MLP
    - Reparameterization: prefix = MLP(prefix_params)
    - More stable training

    Original:
        prefix_k, prefix_v = learnable parameters

    Reparameterized:
        prefix_k = MLP_k(prefix_params)
        prefix_v = MLP_v(prefix_params)
    """

    def __init__(self, config: PrefixTuningConfig):
        super().__init__()
        self.config = config

        if config.prefix_projection:
            # Reparameterization through MLP
            # Input: prefix_length × d_model
            # Output: prefix_length × n_layers × 2 × d_model
            #   (2 for key and value, n_layers for all layers)

            self.prefix_params = nn.Parameter(
                torch.randn(config.prefix_length, config.d_model)
            )

            # MLP for reparameterization
            self.mlp = nn.Sequential(
                nn.Linear(config.d_model, config.prefix_hidden_size),
                nn.Tanh(),
                nn.Linear(config.prefix_hidden_size, config.n_layers * 2 * config.d_model)
            )

            self.dropout = nn.Dropout(config.prefix_dropout)
        else:
            # Direct parameterization (simpler but less stable)
            self.prefix_kv = nn.Parameter(
                torch.randn(config.n_layers, 2, config.prefix_length, config.d_model)
            )

    def forward(self) -> torch.Tensor:
        """
        Generate prefix key-value pairs for all layers.

        Returns:
            prefix_kv: [n_layers, 2, prefix_length, d_model]
                where [:, 0] = keys, [:, 1] = values
        """
        if self.config.prefix_projection:
            # Reparameterization
            # prefix_params: [prefix_length, d_model]
            prefix_flat = self.mlp(self.prefix_params)  # [prefix_length, n_layers*2*d_model]
            prefix_flat = self.dropout(prefix_flat)

            # Reshape to [prefix_length, n_layers, 2, d_model]
            prefix = prefix_flat.view(
                self.config.prefix_length,
                self.config.n_layers,
                2,
                self.config.d_model
            )

            # Transpose to [n_layers, 2, prefix_length, d_model]
            prefix_kv = prefix.permute(1, 2, 0, 3)
        else:
            # Direct parameterization
            prefix_kv = self.prefix_kv

        return prefix_kv


class PrefixTuningAttention(nn.Module):
    """
    Attention layer with prefix tuning.

    Key Idea:
    - Prepend learned prefix to key and value
    - Prefix acts as "virtual" tokens that guide attention
    - Only train prefix, freeze base model

    Standard attention:
        Q = input @ W_q
        K = input @ W_k
        V = input @ W_v
        output = softmax(Q @ K^T) @ V

    With prefix tuning:
        K = concat([prefix_k, input @ W_k])
        V = concat([prefix_v, input @ W_v])
        output = softmax(Q @ K^T) @ V
    """

    def __init__(self, config: PrefixTuningConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Standard attention (frozen)
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            batch_first=True
        )

    def forward(
        self,
        x: torch.Tensor,
        prefix_kv: torch.Tensor
    ) -> torch.Tensor:
        """
        Attention with prefix.

        Args:
            x: Input [batch, seq, d_model]
            prefix_kv: Prefix key-value [n_layers, 2, prefix_len, d_model]

        Returns:
            output: [batch, seq, d_model]
        """
        batch, seq_len, d_model = x.shape

        # Get prefix for this layer
        prefix_k = prefix_kv[self.layer_idx, 0]  # [prefix_len, d_model]
        prefix_v = prefix_kv[self.layer_idx, 1]  # [prefix_len, d_model]

        # Expand for batch
        prefix_k = prefix_k.unsqueeze(0).expand(batch, -1, -1)  # [batch, prefix_len, d_model]
        prefix_v = prefix_v.unsqueeze(0).expand(batch, -1, -1)

        # Concatenate prefix to input for key and value
        k = torch.cat([prefix_k, x], dim=1)  # [batch, prefix_len + seq, d_model]
        v = torch.cat([prefix_v, x], dim=1)

        # Standard attention (query is just input, key/value include prefix)
        output, _ = self.attn(x, k, v)

        return output


class PrefixTuningModel(nn.Module):
    """
    Complete model with Prefix Tuning.

    Example:
        >>> config = PrefixTuningConfig(
        ...     d_model=768,
        ...     n_layers=12,
        ...     prefix_length=20
        ... )
        >>> model = PrefixTuningModel(config)
        >>>
        >>> # Only train prefix (0.1% of parameters!)
        >>> trainable = sum(p.numel() for p in model.prefix_encoder.parameters())
        >>> total = sum(p.numel() for p in model.parameters())
        >>> print(f"Trainable: {trainable/total:.2%}")
        >>>
        >>> x = torch.randn(2, 128, 768)
        >>> out = model(x)
    """

    def __init__(self, config: PrefixTuningConfig):
        super().__init__()
        self.config = config

        # Prefix encoder (TRAINABLE)
        self.prefix_encoder = PrefixEncoder(config)

        # Transformer layers with prefix attention (FROZEN)
        self.layers = nn.ModuleList([
            PrefixTuningAttention(config, i) for i in range(config.n_layers)
        ])

        # Layer norms and FFN (FROZEN)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.d_model) for _ in range(config.n_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 4),
                nn.GELU(),
                nn.Linear(config.d_model * 4, config.d_model)
            )
            for _ in range(config.n_layers)
        ])

        # Freeze base model (only train prefix)
        self._freeze_base_model()

    def _freeze_base_model(self):
        """Freeze all parameters except prefix encoder."""
        for name, param in self.named_parameters():
            if 'prefix_encoder' not in name:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with prefix."""
        # Generate prefix
        prefix_kv = self.prefix_encoder()  # [n_layers, 2, prefix_len, d_model]

        # Pass through layers
        for i, (layer, ln, ffn) in enumerate(zip(self.layers, self.layer_norms, self.ffns)):
            # Attention with prefix
            residual = x
            x = layer(ln(x), prefix_kv)
            x = residual + x

            # FFN
            residual = x
            x = ffn(ln(x))
            x = residual + x

        return x


# ============================================================================
# P-Tuning v2
# ============================================================================

@dataclass
class PTuningV2Config:
    """Configuration for P-Tuning v2"""
    d_model: int = 768
    n_layers: int = 12

    # Prompt
    prompt_length: int = 20  # Number of prompt tokens
    prompt_deep: bool = True  # Use deep prompt (across all layers)
    prompt_dropout: float = 0.1


class PTuningV2Model(nn.Module):
    """
    P-Tuning v2: Deep Prompt Tuning

    Key Innovation vs Prefix Tuning:
    - Simpler: Just prepend trainable embeddings
    - No reparameterization needed
    - Works across all layers (deep prompts)
    - Even more parameter-efficient

    Difference from Prefix Tuning:
    - Prefix Tuning: Learns K, V pairs through MLP
    - P-Tuning v2: Directly learns prompt embeddings, added to input

    Process:
    - Layer 0: [prompt_emb; input_emb]
    - Layer i: [prompt_emb_i; hidden_i]

    Example:
        >>> config = PTuningV2Config(
        ...     d_model=768,
        ...     n_layers=12,
        ...     prompt_length=20,
        ...     prompt_deep=True
        ... )
        >>> model = PTuningV2Model(config)
        >>> x = torch.randn(2, 128, 768)
        >>> out = model(x)
    """

    def __init__(self, config: PTuningV2Config):
        super().__init__()
        self.config = config

        if config.prompt_deep:
            # Deep prompts: one per layer
            self.prompt_embeddings = nn.ParameterList([
                nn.Parameter(torch.randn(config.prompt_length, config.d_model))
                for _ in range(config.n_layers)
            ])
        else:
            # Shallow prompt: only at input
            self.prompt_embeddings = nn.ParameterList([
                nn.Parameter(torch.randn(config.prompt_length, config.d_model))
            ])

        self.dropout = nn.Dropout(config.prompt_dropout)

        # Transformer layers (FROZEN)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=12,
                dim_feedforward=config.d_model * 4,
                batch_first=True
            )
            for _ in range(config.n_layers)
        ])

        # Freeze base model
        self._freeze_base_model()

    def _freeze_base_model(self):
        """Freeze all except prompt embeddings."""
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with deep prompts.

        Args:
            x: Input [batch, seq, d_model]

        Returns:
            output: [batch, seq, d_model] (without prompt tokens)
        """
        batch, seq_len, d_model = x.shape

        for i, layer in enumerate(self.layers):
            # Get prompt for this layer
            if self.config.prompt_deep:
                prompt = self.prompt_embeddings[i]
            else:
                prompt = self.prompt_embeddings[0] if i == 0 else None

            if prompt is not None:
                # Prepend prompt
                prompt = prompt.unsqueeze(0).expand(batch, -1, -1)  # [batch, prompt_len, d_model]
                prompt = self.dropout(prompt)

                x_with_prompt = torch.cat([prompt, x], dim=1)  # [batch, prompt_len + seq, d_model]

                # Apply layer
                output_with_prompt = layer(x_with_prompt)

                # Remove prompt from output
                x = output_with_prompt[:, self.config.prompt_length:]
            else:
                # No prompt for this layer
                x = layer(x)

        return x


# ============================================================================
# Adapter Layers
# ============================================================================

@dataclass
class AdapterConfig:
    """Configuration for Adapter layers"""
    d_model: int = 768
    adapter_size: int = 64  # Bottleneck dimension
    adapter_dropout: float = 0.1


class AdapterLayer(nn.Module):
    """
    Adapter Layer: Bottleneck adapter in transformer.

    Key Innovation:
    - Small bottleneck layers inserted in transformer
    - Down-project → nonlinearity → up-project
    - Residual connection

    Process:
        h = LayerNorm(x)
        down = W_down @ h  # d_model → adapter_size
        up = W_up @ ReLU(down)  # adapter_size → d_model
        output = x + up

    Example:
        >>> config = AdapterConfig(d_model=768, adapter_size=64)
        >>> adapter = AdapterLayer(config)
        >>> x = torch.randn(2, 128, 768)
        >>> out = adapter(x)
        >>>
        >>> # Only 64*768*2 = 98K parameters (vs 768*768 = 590K)
    """

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config

        # Down-projection
        self.down_proj = nn.Linear(config.d_model, config.adapter_size)

        # Up-projection
        self.up_proj = nn.Linear(config.adapter_size, config.d_model)

        # Layer norm
        self.ln = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.adapter_dropout)

        # Initialize to near-identity (important!)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through adapter.

        Args:
            x: Input [batch, seq, d_model]

        Returns:
            output: [batch, seq, d_model]
        """
        residual = x

        # Adapter transformation
        x = self.ln(x)
        x = self.down_proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.up_proj(x)

        # Residual
        return residual + x


# ============================================================================
# BitFit
# ============================================================================

class BitFit:
    """
    BitFit: Bias-only Fine-tuning

    Key Innovation:
    - Only train bias parameters
    - Freeze all weights
    - Extremely parameter-efficient (<0.1%)

    Surprisingly effective for many tasks!

    Example:
        >>> model = TransformerModel()
        >>> BitFit.apply(model)  # Freeze weights, enable biases
        >>> # Now only biases are trainable
    """

    @staticmethod
    def apply(model: nn.Module):
        """
        Apply BitFit to model: freeze weights, enable biases.

        Args:
            model: Model to apply BitFit to
        """
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        print(f"BitFit: {trainable:,} / {total:,} parameters ({trainable/total:.2%})")


# ============================================================================
# Testing
# ============================================================================

def test_prefix_tuning():
    """Test Prefix Tuning."""
    print("=" * 80)
    print("Test 1: Prefix Tuning")
    print("=" * 80)

    config = PrefixTuningConfig(
        d_model=256,
        n_heads=8,
        n_layers=4,
        prefix_length=10,
        prefix_projection=True
    )

    model = PrefixTuningModel(config)

    batch_size = 2
    seq_len = 128
    x = torch.randn(batch_size, seq_len, config.d_model)

    print(f"Input shape: {x.shape}")
    print(f"Prefix length: {config.prefix_length}")

    with torch.no_grad():
        output = model(x)

    assert output.shape == x.shape

    # Count trainable vs total parameters
    trainable = sum(p.numel() for p in model.prefix_encoder.parameters())
    total = sum(p.numel() for p in model.parameters())

    print(f"\n✓ Prefix Tuning test PASSED")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {trainable:,} ({trainable/total:.2%})")
    print(f"Total parameters: {total:,}")

    return {
        'status': 'PASS',
        'output_shape': output.shape,
        'trainable_params': trainable,
        'total_params': total,
        'trainable_ratio': trainable/total
    }


def test_ptuning_v2():
    """Test P-Tuning v2."""
    print("\n" + "=" * 80)
    print("Test 2: P-Tuning v2")
    print("=" * 80)

    config = PTuningV2Config(
        d_model=256,
        n_layers=4,
        prompt_length=10,
        prompt_deep=True
    )

    model = PTuningV2Model(config)

    batch_size = 2
    seq_len = 128
    x = torch.randn(batch_size, seq_len, config.d_model)

    print(f"Input shape: {x.shape}")
    print(f"Prompt length: {config.prompt_length}")
    print(f"Deep prompts: {config.prompt_deep}")

    with torch.no_grad():
        output = model(x)

    assert output.shape == x.shape

    # Count trainable
    trainable = sum(p.numel() for p in model.prompt_embeddings.parameters())
    total = sum(p.numel() for p in model.parameters())

    print(f"\n✓ P-Tuning v2 test PASSED")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {trainable:,} ({trainable/total:.2%})")

    return {
        'status': 'PASS',
        'output_shape': output.shape,
        'trainable_params': trainable,
        'total_params': total,
        'trainable_ratio': trainable/total
    }


def test_adapter():
    """Test Adapter layers."""
    print("\n" + "=" * 80)
    print("Test 3: Adapter Layers")
    print("=" * 80)

    config = AdapterConfig(
        d_model=256,
        adapter_size=32
    )

    adapter = AdapterLayer(config)

    batch_size = 2
    seq_len = 128
    x = torch.randn(batch_size, seq_len, config.d_model)

    print(f"Input shape: {x.shape}")
    print(f"Adapter size: {config.adapter_size}")
    print(f"Compression ratio: {config.d_model / config.adapter_size:.1f}x")

    with torch.no_grad():
        output = adapter(x)

    assert output.shape == x.shape

    adapter_params = sum(p.numel() for p in adapter.parameters())
    full_ffn_params = config.d_model * config.d_model * 2  # Two linear layers

    print(f"\n✓ Adapter test PASSED")
    print(f"Output shape: {output.shape}")
    print(f"Adapter parameters: {adapter_params:,}")
    print(f"vs Full FFN: {full_ffn_params:,}")
    print(f"Reduction: {full_ffn_params / adapter_params:.1f}x")

    return {
        'status': 'PASS',
        'output_shape': output.shape,
        'adapter_params': adapter_params,
        'full_ffn_params': full_ffn_params
    }


def test_all():
    """Run all PEFT tests."""
    print("\n" + "=" * 80)
    print("Parameter-Efficient Fine-Tuning - Complete Test Suite")
    print("=" * 80)

    results = {}

    # Test 1: Prefix Tuning
    results['PrefixTuning'] = test_prefix_tuning()

    # Test 2: P-Tuning v2
    results['PTuningV2'] = test_ptuning_v2()

    # Test 3: Adapter
    results['Adapter'] = test_adapter()

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for name, result in results.items():
        print(f"\n{name}: {result['status']}")

    print("\n" + "=" * 80)
    print("PEFT Methods Comparison")
    print("=" * 80)
    print("""
Method          | Trainable   | Memory  | Quality | Best For
----------------|-------------|---------|---------|------------------
Full FT         | 100%        | High    | Best    | Unlimited resources
LoRA            | 0.1-1%      | Low     | Great   | General use
Prefix Tuning   | 0.1-0.5%    | Low     | Good    | Seq2seq tasks
P-Tuning v2     | <0.1%       | Lowest  | Good    | NLU tasks
Adapter         | 0.5-2%      | Low     | Great   | Multi-task
BitFit          | <0.1%       | Lowest  | Fair    | Simple tasks

Key Advantages by Method:

1. Prefix Tuning:
   - Prepends learned "prefix" tokens
   - Guides attention through virtual tokens
   - Good for generation tasks
   - 0.1-0.5% of parameters

2. P-Tuning v2:
   - Simpler than Prefix Tuning
   - Deep prompts across all layers
   - Very parameter-efficient (<0.1%)
   - Strong on NLU benchmarks

3. Adapter Layers:
   - Bottleneck layers in transformer
   - Easy to implement
   - Can train multiple adapters for multiple tasks
   - 0.5-2% of parameters

4. BitFit:
   - Only train biases
   - <0.1% of parameters
   - Surprisingly effective
   - Easiest to implement

Performance Comparison:
----------------------
Task             | LoRA | Prefix | P-Tuning | Adapter | BitFit
-----------------|------|--------|----------|---------|-------
GLUE (NLU)       | 98%  | 96%    | 98%      | 97%     | 94%
SQuAD (QA)       | 99%  | 97%    | 98%      | 98%     | 95%
MT (Translation) | 98%  | 99%    | 97%      | 98%     | 93%

(% = relative to full fine-tuning)

When to Use:
-----------
- LoRA: Default choice, best overall
- Prefix Tuning: Generation tasks (translation, summarization)
- P-Tuning v2: NLU tasks, minimal parameters
- Adapter: Multiple tasks on same model
- BitFit: Extremely limited resources

Production Usage:
----------------
- Most common: LoRA (easiest, best quality)
- Growing: Prefix/P-Tuning (more efficient)
- Multi-task: Adapters (swap adapters per task)
- Research: All methods active

Memory Savings:
--------------
Full FT: 16 GB
LoRA: 4 GB (4x savings)
Prefix/P-Tuning: 2 GB (8x savings)
BitFit: 1 GB (16x savings)

All PEFT methods enable:
- Fine-tuning on consumer GPUs
- Multiple task-specific models from one base
- Faster experimentation
- Lower costs
    """)

    print("=" * 80)

    return results


if __name__ == "__main__":
    test_all()
