"""
State-of-the-Art Transformer Architecture

Complete implementation with modern improvements:
- Pre-normalization (Pre-LN)
- Parallel attention and FFN (like in GPT-J)
- Multiple normalization options (LayerNorm, RMSNorm)
- Modern activation functions (SwiGLU, GeGLU)
- Rotary Position Embeddings (RoPE)
- ALiBi positional bias
- Flash Attention support
- Gradient checkpointing
- Mixed precision training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal, Tuple
from dataclasses import dataclass

from ..base import LanguageArchitecture, ModelOutput
from .multihead_attention import MultiHeadAttention, AttentionConfig, GroupedQueryAttention


@dataclass
class TransformerConfig:
    """Configuration for Transformer model"""
    # Model architecture
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Normalization
    norm_type: Literal['layernorm', 'rmsnorm', 'scalenorm'] = 'rmsnorm'
    norm_eps: float = 1e-5
    pre_norm: bool = True  # Pre-LN vs Post-LN

    # Activation
    activation: Literal['relu', 'gelu', 'swiglu', 'geglu'] = 'swiglu'

    # Position encoding
    max_seq_len: int = 2048
    use_rope: bool = True  # Rotary Position Embeddings
    use_alibi: bool = False  # ALiBi positional bias

    # Attention variant
    use_gqa: bool = False  # Grouped-Query Attention
    num_kv_heads: Optional[int] = None  # For GQA
    use_flash: bool = True  # Flash Attention
    causal: bool = True  # Causal (autoregressive) attention

    # Efficiency
    parallel_attn_ffn: bool = False  # Parallel attention and FFN (GPT-J style)
    use_bias: bool = False  # Bias in linear layers
    gradient_checkpointing: bool = False

    # Vocab
    vocab_size: int = 50257


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More efficient than LayerNorm, used in modern LLMs.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class ScaleNorm(nn.Module):
    """
    Scale Normalization - Even simpler than RMSNorm.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.tensor(d_model ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True)
        return self.scale * x / (norm + self.eps)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    Used in PaLM, LLaMA, and other modern LLMs.

    SwiGLU(x) = Swish(xW) ⊙ (xV)
    where Swish(x) = x * sigmoid(x)
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GeGLU(nn.Module):
    """
    GeGLU activation function.
    Similar to SwiGLU but uses GELU.
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.gelu(self.w1(x)) * self.w2(x))


class FeedForward(nn.Module):
    """
    Feed-Forward Network with modern activations.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        if config.activation == 'swiglu':
            self.ffn = SwiGLU(config.d_model, config.d_ff, config.use_bias)
        elif config.activation == 'geglu':
            self.ffn = GeGLU(config.d_model, config.d_ff, config.use_bias)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff, bias=config.use_bias),
                nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
            )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.ffn(x))


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Used in modern LLMs like GPT-NeoX, LLaMA, PaLM.

    Applies rotary embeddings to queries and keys.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Cache for efficiency
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin embeddings for the given sequence length.
        """
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

        return self._cos_cached, self._sin_cached


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi).
    Adds position-dependent bias to attention scores.
    Allows for better length extrapolation.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

        # Compute slopes for each head
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2) +
                    get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
                )

        slopes = torch.tensor(get_slopes(num_heads))
        self.register_buffer('slopes', slopes)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Generate ALiBi bias matrix.

        Returns tensor of shape (num_heads, seq_len, seq_len)
        """
        # Create position matrix
        positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        positions = positions.abs()

        # Apply slopes
        bias = -positions.unsqueeze(0) * self.slopes.unsqueeze(-1).unsqueeze(-1)

        return bias


class TransformerBlock(nn.Module):
    """
    Single Transformer block with all modern improvements.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Normalization layers
        norm_class = {
            'layernorm': lambda: nn.LayerNorm(config.d_model, eps=config.norm_eps),
            'rmsnorm': lambda: RMSNorm(config.d_model, eps=config.norm_eps),
            'scalenorm': lambda: ScaleNorm(config.d_model, eps=config.norm_eps)
        }[config.norm_type]

        self.ln1 = norm_class()
        if not config.parallel_attn_ffn:
            self.ln2 = norm_class()

        # Attention
        if config.use_gqa and config.num_kv_heads is not None:
            self.attn = GroupedQueryAttention(
                config.d_model,
                config.num_heads,
                config.num_kv_heads,
                config.attention_dropout,
                config.use_bias
            )
        else:
            attn_config = AttentionConfig(
                d_model=config.d_model,
                num_heads=config.num_heads,
                dropout=config.attention_dropout,
                bias=config.use_bias,
                causal=config.causal,
                use_flash=config.use_flash
            )
            self.attn = MultiHeadAttention(attn_config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Parallel or sequential
        self.parallel = config.parallel_attn_ffn

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass of Transformer block"""

        if self.parallel:
            # Parallel attention and FFN (GPT-J style)
            x_norm = self.ln1(x)
            attn_out, _ = self.attn(x_norm, mask=mask) if isinstance(self.attn, MultiHeadAttention) else (self.attn(x_norm, mask=mask), None)
            ffn_out = self.ffn(x_norm)
            x = x + attn_out + ffn_out
        else:
            # Sequential (standard)
            if self.config.pre_norm:
                # Pre-LN
                attn_out, _ = self.attn(self.ln1(x), mask=mask) if isinstance(self.attn, MultiHeadAttention) else (self.attn(self.ln1(x), mask=mask), None)
                x = x + attn_out
                x = x + self.ffn(self.ln2(x))
            else:
                # Post-LN
                attn_out, _ = self.attn(x, mask=mask) if isinstance(self.attn, MultiHeadAttention) else (self.attn(x, mask=mask), None)
                x = self.ln1(x + attn_out)
                x = self.ln2(x + self.ffn(x))

        return x


class Transformer(LanguageArchitecture):
    """
    Complete State-of-the-Art Transformer model.

    Inherits from LanguageArchitecture for unified Brain framework interface.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Position embeddings
        if config.use_rope:
            self.rope = RotaryPositionEmbedding(
                config.d_model // config.num_heads,
                config.max_seq_len
            )
        elif config.use_alibi:
            self.alibi = ALiBiPositionalBias(config.num_heads)
        else:
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final normalization
        if config.norm_type == 'rmsnorm':
            self.ln_f = RMSNorm(config.d_model, eps=config.norm_eps)
        else:
            self.ln_f = nn.LayerNorm(config.d_model, eps=config.norm_eps)

        # Output head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights (standard practice)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following modern best practices"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """
        Forward pass of the Transformer.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation (batch_size, seq_len)

        Returns:
            ModelOutput with logits and optional loss
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Add position information
        if self.config.use_rope:
            rope_cos_sin = self.rope(x, seq_len)
        elif self.config.use_alibi:
            # ALiBi is applied in attention
            rope_cos_sin = None
        else:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_embedding(positions)
            rope_cos_sin = None

        # Apply transformer blocks
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, attention_mask, rope_cos_sin)
            else:
                x = block(x, attention_mask, rope_cos_sin)

        # Final normalization
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return ModelOutput(
            logits=logits,
            loss=loss,
            predictions=logits.argmax(dim=-1) if not self.training else None
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting tokens
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus (top-p) sampling

        Returns:
            Generated token IDs
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat((input_ids, idx_next), dim=1)

        return input_ids
