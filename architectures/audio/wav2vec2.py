"""
Wav2Vec 2.0 - Self-Supervised Speech Representation Learning

Facebook/Meta's breakthrough model for learning speech representations
from unlabeled audio data (2020-2025).

Key features:
- Self-supervised pre-training on raw audio
- Contrastive learning with quantization
- Fine-tuning for ASR with minimal labeled data
- Multilingual support
- SOTA on many speech benchmarks

Architecture:
- Feature encoder: 7-layer CNN
- Transformer: 12 or 24 layers
- Quantization module: Gumbel softmax
- Contrastive loss: InfoNCE

References:
- "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (Baevski et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class Wav2Vec2Config:
    """Configuration for Wav2Vec2 model"""
    # Feature encoder (CNN)
    conv_layers: list = None  # [(channels, kernel, stride)]
    conv_dropout: float = 0.0

    # Transformer
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1

    # Quantization
    num_codevector_groups: int = 2
    num_codevectors_per_group: int = 320
    codevector_dim: int = 256

    # Contrastive loss
    contrastive_logits_temperature: float = 0.1
    num_negatives: int = 100
    negative_sampling: str = "uniform"  # uniform or same_speaker

    # Masking (for pre-training)
    mask_prob: float = 0.065  # Probability of masking a time step
    mask_length: int = 10  # Length of mask span
    min_masks: int = 2

    # Diversity loss
    diversity_loss_weight: float = 0.1

    def __post_init__(self):
        if self.conv_layers is None:
            # Default: 7 CNN layers like in wav2vec 2.0 base
            self.conv_layers = [
                (512, 10, 5),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 2, 2),
                (512, 2, 2)
            ]


class FeatureEncoder(nn.Module):
    """
    CNN-based feature encoder.

    Converts raw waveform to features with ~20ms frame shift.
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config

        # Build CNN layers
        layers = []
        in_channels = 1  # Raw audio

        for i, (out_channels, kernel_size, stride) in enumerate(config.conv_layers):
            layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=False
                ),
                nn.Dropout(config.conv_dropout),
                nn.GroupNorm(out_channels, out_channels),  # Group norm
                nn.GELU()
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Projection to transformer dimension
        self.projection = nn.Linear(config.conv_layers[-1][0], config.d_model)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode raw audio to features.

        Args:
            audio: Raw waveform (batch, time)

        Returns:
            Features (batch, seq_len, d_model)
        """
        # Add channel dimension
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # (batch, 1, time)

        # Pass through CNN
        features = self.conv_layers(audio)  # (batch, channels, seq_len)

        # Transpose and project
        features = features.transpose(1, 2)  # (batch, seq_len, channels)
        features = self.projection(features)

        return features


class FeatureProjection(nn.Module):
    """Layer norm and projection for encoder output"""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GumbelVectorQuantizer(nn.Module):
    """
    Gumbel softmax-based vector quantizer.

    Quantizes continuous features to discrete codes for contrastive learning.
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config

        # Codebooks
        self.codevectors = nn.Parameter(
            torch.randn(
                1,
                config.num_codevector_groups * config.num_codevectors_per_group,
                config.codevector_dim
            )
        )

        # Projection layers
        self.weight_proj = nn.Linear(config.d_model, config.num_codevector_groups * config.num_codevectors_per_group)

        # Temperature for Gumbel softmax
        self.temperature = 2.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask_time_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize hidden states.

        Args:
            hidden_states: (batch, seq_len, d_model)
            mask_time_indices: Boolean mask (batch, seq_len)

        Returns:
            quantized: Quantized vectors (batch, seq_len, codevector_dim)
            perplexity: Perplexity of code usage
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to logits
        logits = self.weight_proj(hidden_states)  # (batch, seq_len, num_codes)
        logits = logits.view(
            batch_size, seq_len,
            self.config.num_codevector_groups,
            self.config.num_codevectors_per_group
        )

        # Apply mask if provided
        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.unsqueeze(-1).unsqueeze(-1)
            logits = logits * mask_time_indices

        # Gumbel softmax sampling
        if self.training:
            # Add Gumbel noise
            gumbels = -torch.log(
                -torch.log(torch.rand_like(logits) + 1e-8) + 1e-8
            )
            logits = (logits + gumbels) / self.temperature

        # Softmax over codevectors
        probs = F.softmax(logits, dim=-1)

        # Hard assignment (one-hot)
        if self.training:
            # Straight-through estimator
            indices = probs.argmax(dim=-1)
            hard = F.one_hot(indices, self.config.num_codevectors_per_group).float()
            probs = hard - probs.detach() + probs
        else:
            indices = probs.argmax(dim=-1)
            probs = F.one_hot(indices, self.config.num_codevectors_per_group).float()

        # Compute perplexity (diversity metric)
        avg_probs = probs.view(-1, self.config.num_codevectors_per_group).mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        # Get codevectors
        codevector_idx = probs.view(batch_size * seq_len, -1)
        codevectors_flat = self.codevectors.view(-1, self.config.codevector_dim)
        quantized = codevector_idx @ codevectors_flat

        # Reshape
        quantized = quantized.view(batch_size, seq_len, -1)

        return quantized, perplexity


class Wav2Vec2Encoder(nn.Module):
    """Transformer encoder for Wav2Vec2"""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config

        # Position embeddings (relative or absolute)
        self.pos_conv = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=128,
            padding=128 // 2,
            groups=16
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            Wav2Vec2EncoderLayer(config)
            for _ in range(config.num_layers)
        ])

        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward through transformer encoder"""
        # Add positional encoding
        hidden_states = hidden_states.transpose(1, 2)  # (batch, d_model, seq_len)
        hidden_states = hidden_states + self.pos_conv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)  # (batch, seq_len, d_model)

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class Wav2Vec2EncoderLayer(nn.Module):
    """Single transformer encoder layer"""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.activation_dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward with pre-norm"""
        # Self-attention
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=attention_mask,
            need_weights=False
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Wav2Vec2(nn.Module):
    """
    Complete Wav2Vec 2.0 model.

    Pre-training: Self-supervised with contrastive loss
    Fine-tuning: Add CTC head for ASR
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config

        # Feature encoder (CNN)
        self.feature_encoder = FeatureEncoder(config)

        # Feature projection
        self.feature_projection = FeatureProjection(config)

        # Quantizer (for pre-training)
        self.quantizer = GumbelVectorQuantizer(config)

        # Context encoder (Transformer)
        self.encoder = Wav2Vec2Encoder(config)

        # For fine-tuning
        self.lm_head = None  # Added during fine-tuning

    def _mask_hidden_states(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask hidden states for contrastive learning.

        Returns:
            masked_hidden_states: Hidden states with masks applied
            mask_time_indices: Boolean mask indicating which steps are masked
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Compute number of masks
        num_masked_spans = int(self.config.mask_prob * seq_len / self.config.mask_length)
        num_masked_spans = max(num_masked_spans, self.config.min_masks)

        # Create mask
        mask_time_indices = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)

        for batch_idx in range(batch_size):
            # Sample start indices
            mask_starts = torch.randperm(seq_len - self.config.mask_length + 1)[:num_masked_spans]

            # Create spans
            for start in mask_starts:
                mask_time_indices[batch_idx, start:start + self.config.mask_length] = True

        # Apply mask (replace with learnable mask embedding)
        masked_hidden_states = hidden_states.clone()
        mask_embedding = nn.Parameter(torch.randn(hidden_size))
        masked_hidden_states[mask_time_indices] = mask_embedding.to(hidden_states.device)

        return masked_hidden_states, mask_time_indices

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pre-training.

        Args:
            audio: Raw waveform (batch, time)
            attention_mask: Padding mask
            mask_time_indices: Pre-computed mask indices

        Returns:
            Dictionary with loss and metrics
        """
        # Extract features
        features = self.feature_encoder(audio)

        # Project features
        features = self.feature_projection(features)

        # Quantize (before masking)
        quantized_features, perplexity = self.quantizer(features, mask_time_indices)

        # Mask features (for contrastive learning)
        if mask_time_indices is None:
            features, mask_time_indices = self._mask_hidden_states(features)
        else:
            # Use provided mask
            mask_embedding = nn.Parameter(torch.randn(features.shape[-1]))
            features[mask_time_indices] = mask_embedding.to(features.device)

        # Encode with transformer
        hidden_states = self.encoder(features, attention_mask)

        # Compute contrastive loss
        contrastive_loss = self._compute_contrastive_loss(
            hidden_states,
            quantized_features,
            mask_time_indices
        )

        # Diversity loss (encourage using all codebook entries)
        diversity_loss = (self.config.num_codevector_groups * self.config.num_codevectors_per_group - perplexity) / \
                        (self.config.num_codevector_groups * self.config.num_codevectors_per_group)

        # Total loss
        loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'diversity_loss': diversity_loss,
            'perplexity': perplexity,
            'hidden_states': hidden_states
        }

    def _compute_contrastive_loss(
        self,
        context: torch.Tensor,
        quantized: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss (InfoNCE).

        Predict quantized targets from context.
        """
        # Only compute loss for masked positions
        context_masked = context[mask]  # (num_masked, d_model)
        quantized_masked = quantized[mask]  # (num_masked, codevector_dim)

        # Sample negatives
        batch_size = quantized.shape[0]
        negatives = self._sample_negatives(quantized, mask)  # (num_masked, num_negatives, codevector_dim)

        # Compute similarities
        positive_logits = torch.sum(context_masked * quantized_masked, dim=-1)  # (num_masked,)
        negative_logits = torch.bmm(
            negatives,
            context_masked.unsqueeze(-1)
        ).squeeze(-1)  # (num_masked, num_negatives)

        # Concatenate and apply temperature
        logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
        logits = logits / self.config.contrastive_logits_temperature

        # Targets are always 0 (positive is first)
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, targets)

        return loss

    def _sample_negatives(
        self,
        quantized: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Sample negative examples for contrastive learning"""
        batch_size, seq_len, hidden_size = quantized.shape
        num_masked = mask.sum()

        # Flatten
        quantized_flat = quantized.view(-1, hidden_size)

        # Sample negatives uniformly
        negative_indices = torch.randint(
            0, batch_size * seq_len,
            (num_masked, self.config.num_negatives),
            device=quantized.device
        )

        negatives = quantized_flat[negative_indices]

        return negatives


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Wav2Vec 2.0 - Self-Supervised Speech Learning")
    print("="*80)

    # Create Wav2Vec2 model (base configuration)
    config = Wav2Vec2Config(
        d_model=768,
        num_layers=12,
        num_heads=12
    )

    model = Wav2Vec2(config)

    # Example: Pre-training on raw audio
    batch_size = 4
    audio_length = 16000 * 10  # 10 seconds at 16kHz
    audio = torch.randn(batch_size, audio_length)

    print(f"\nInput audio shape: {audio.shape}")

    # Forward pass
    outputs = model(audio)

    print(f"\nOutputs:")
    print(f"  Total loss: {outputs['loss'].item():.4f}")
    print(f"  Contrastive loss: {outputs['contrastive_loss'].item():.4f}")
    print(f"  Diversity loss: {outputs['diversity_loss'].item():.4f}")
    print(f"  Perplexity: {outputs['perplexity'].item():.2f}")
    print(f"  Hidden states shape: {outputs['hidden_states'].shape}")

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "="*80)
