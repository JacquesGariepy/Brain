"""
Flamingo - Few-Shot Visual Reasoning Architecture

DeepMind's architecture for few-shot learning on vision-language tasks.
Uses cross-attention to condition frozen LM on visual information.

Key innovations:
- Perceiver Resampler: Compresses variable-size visual inputs
- Gated cross-attention: Inserted into frozen LM layers
- Few-shot in-context learning with images
- Interleaved image-text sequences

References:
- "Flamingo: a Visual Language Model for Few-Shot Learning" (Alayrac et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class FlamingoConfig:
    """Configuration for Flamingo model"""
    # Vision
    vision_encoder: str = "clip_vit_l"
    vision_hidden_size: int = 1024
    freeze_vision: bool = True

    # Perceiver Resampler
    num_latents: int = 64  # Number of latent queries
    resampler_depth: int = 6
    resampler_dim: int = 1024
    resampler_heads: int = 16
    resampler_dim_head: int = 64

    # Language Model
    language_model_dim: int = 4096
    freeze_language: bool = True

    # Gated Cross-Attention
    cross_attention_freq: int = 4  # Insert every N layers


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler compresses variable-size visual inputs to fixed latents.

    Uses cross-attention from learned latent queries to image features,
    producing a fixed number of visual tokens regardless of input size.
    """

    def __init__(self, config: FlamingoConfig):
        super().__init__()
        self.config = config

        # Learned latent queries
        self.latents = nn.Parameter(
            torch.randn(config.num_latents, config.resampler_dim)
        )

        # Transformer blocks with cross-attention
        self.layers = nn.ModuleList([
            PerceiverResamplerLayer(
                dim=config.resampler_dim,
                num_heads=config.resampler_heads,
                dim_head=config.resampler_dim_head
            )
            for _ in range(config.resampler_depth)
        ])

        # Projection from vision to resampler dim
        self.proj_in = nn.Linear(config.vision_hidden_size, config.resampler_dim)

        # Final norm
        self.norm = nn.LayerNorm(config.resampler_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Resample visual features to fixed number of tokens.

        Args:
            visual_features: (batch, num_patches, vision_hidden_size)

        Returns:
            Resampled features: (batch, num_latents, resampler_dim)
        """
        batch_size = visual_features.shape[0]

        # Project visual features
        visual_features = self.proj_in(visual_features)

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply perceiver layers
        for layer in self.layers:
            latents = layer(latents, visual_features)

        # Final norm
        latents = self.norm(latents)

        return latents


class PerceiverResamplerLayer(nn.Module):
    """Single layer of Perceiver Resampler"""

    def __init__(self, dim: int, num_heads: int, dim_head: int):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Cross-attention from latents to visual features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            kdim=dim,
            vdim=dim,
            batch_first=True
        )

        # Self-attention among latents
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.norm_ffn = nn.LayerNorm(dim)

    def forward(
        self,
        latents: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward with cross-attention to visual features.

        Args:
            latents: (batch, num_latents, dim)
            visual_features: (batch, num_patches, dim)

        Returns:
            Updated latents: (batch, num_latents, dim)
        """
        # Cross-attention: latents attend to visual features
        normed_latents = self.norm1(latents)
        cross_out, _ = self.cross_attn(
            normed_latents, visual_features, visual_features,
            need_weights=False
        )
        latents = latents + cross_out

        # Self-attention among latents
        normed_latents = self.norm2(latents)
        self_out, _ = self.self_attn(
            normed_latents, normed_latents, normed_latents,
            need_weights=False
        )
        latents = latents + self_out

        # FFN
        normed_latents = self.norm_ffn(latents)
        latents = latents + self.ffn(normed_latents)

        return latents


class GatedCrossAttentionBlock(nn.Module):
    """
    Gated Cross-Attention block inserted into frozen LM layers.

    Allows LM to attend to visual features with learnable gating.
    The gate starts near zero to preserve pre-trained LM behavior.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        visual_dim: Optional[int] = None
    ):
        super().__init__()

        if visual_dim is None:
            visual_dim = dim

        self.norm = nn.LayerNorm(dim)

        # Cross-attention to visual features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            kdim=visual_dim,
            vdim=visual_dim,
            batch_first=True
        )

        # Tanh gating (starts near 0 to preserve LM)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply gated cross-attention to visual features.

        Args:
            hidden_states: LM hidden states (batch, seq_len, dim)
            visual_features: Visual tokens (batch, num_visual, visual_dim)
            attention_mask: Attention mask

        Returns:
            Updated hidden states
        """
        if visual_features is None:
            return hidden_states

        # Normalize
        normed = self.norm(hidden_states)

        # Cross-attention
        attn_out, _ = self.cross_attn(
            normed, visual_features, visual_features,
            need_weights=False
        )

        # Gated addition
        gate_value = torch.tanh(self.gate)
        hidden_states = hidden_states + gate_value * attn_out

        return hidden_states


class Flamingo(nn.Module):
    """
    Flamingo: Few-shot visual reasoning model.

    Architecture:
    1. Vision encoder (frozen)
    2. Perceiver Resampler
    3. Language model with gated cross-attention layers (frozen LM + trainable gates)

    Key capability: Process interleaved sequences of images and text.
    Can do few-shot learning by conditioning on examples.
    """

    def __init__(
        self,
        config: FlamingoConfig,
        vision_encoder: Optional[nn.Module] = None,
        language_model: Optional[nn.Module] = None
    ):
        super().__init__()
        self.config = config

        # Vision encoder (frozen)
        if vision_encoder is None:
            from ..vision.vision_transformer import VisionTransformer, ViTConfig
            vit_config = ViTConfig(
                image_size=224,
                d_model=config.vision_hidden_size,
                num_layers=24,
                num_heads=16
            )
            self.vision_encoder = VisionTransformer(vit_config)
        else:
            self.vision_encoder = vision_encoder

        if config.freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # Perceiver Resampler
        self.perceiver = PerceiverResampler(config)

        # Projection to language model dimension
        self.visual_proj = nn.Linear(
            config.resampler_dim,
            config.language_model_dim
        )

        # Language model with gated cross-attention
        self.language_model = language_model

        # Inject gated cross-attention layers into LM
        # (In practice, this modifies the LM architecture)
        self.gated_cross_attn_layers = nn.ModuleList()
        if language_model is not None and hasattr(language_model, 'layers'):
            num_layers = len(language_model.layers)
            for i in range(num_layers):
                if i % config.cross_attention_freq == 0:
                    self.gated_cross_attn_layers.append(
                        GatedCrossAttentionBlock(
                            dim=config.language_model_dim,
                            visual_dim=config.language_model_dim
                        )
                    )

        if config.freeze_language and language_model is not None:
            for param in language_model.parameters():
                param.requires_grad = False

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to visual tokens via Perceiver Resampler.

        Args:
            images: (batch, 3, H, W) or (batch, num_images, 3, H, W)

        Returns:
            Visual tokens: (batch, num_latents, lm_dim) or
                          (batch, num_images, num_latents, lm_dim)
        """
        # Handle multiple images per example
        original_shape = images.shape
        if len(original_shape) == 5:
            # (batch, num_images, 3, H, W)
            batch_size, num_images = original_shape[:2]
            images = images.view(-1, *original_shape[2:])
        else:
            num_images = 1

        # Extract vision features (frozen)
        with torch.no_grad() if self.config.freeze_vision else torch.enable_grad():
            visual_features = self.vision_encoder(images, return_all_tokens=True)

        # Resample with Perceiver
        visual_tokens = self.perceiver(visual_features)

        # Project to LM space
        visual_tokens = self.visual_proj(visual_tokens)

        # Reshape if multiple images
        if num_images > 1:
            visual_tokens = visual_tokens.view(
                batch_size, num_images, self.config.num_latents, -1
            )

        return visual_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        image_positions: Optional[List[int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with interleaved image-text sequence.

        Args:
            input_ids: Text tokens (batch, seq_len)
            images: Images (batch, num_images, 3, H, W) or (batch, 3, H, W)
            image_positions: Positions to insert images in sequence
            attention_mask: Attention mask
            labels: Labels for LM loss

        Returns:
            Dictionary with loss and logits
        """
        # Encode images
        visual_tokens = None
        if images is not None:
            visual_tokens = self.encode_vision(images)

        # Forward through LM with visual conditioning
        if self.language_model is not None:
            # This is simplified - actual implementation would properly
            # interleave visual tokens at specified positions
            outputs = self._forward_with_visual_conditioning(
                input_ids=input_ids,
                visual_tokens=visual_tokens,
                attention_mask=attention_mask,
                labels=labels
            )
            return outputs
        else:
            return {'visual_tokens': visual_tokens}

    def _forward_with_visual_conditioning(
        self,
        input_ids: torch.Tensor,
        visual_tokens: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward through LM with gated cross-attention to visual tokens.

        In actual implementation, this would:
        1. Get LM embeddings
        2. Pass through LM layers
        3. At layers with gated cross-attention, condition on visual tokens
        4. Return logits and loss
        """
        # Placeholder - needs actual LM integration
        outputs = {
            'logits': None,
            'loss': None
        }
        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text conditioned on images (few-shot).

        Args:
            input_ids: Prompt tokens
            images: Context images for few-shot learning
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated token IDs
        """
        # Encode visual context
        visual_tokens = None
        if images is not None:
            visual_tokens = self.encode_vision(images)

        # Generate with LM
        if self.language_model is not None:
            # Use LM generation with visual conditioning
            pass

        return input_ids  # Placeholder


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Flamingo - Few-Shot Visual Reasoning")
    print("="*80)

    # Create Flamingo
    config = FlamingoConfig(
        vision_hidden_size=1024,
        num_latents=64,
        resampler_depth=6,
        language_model_dim=4096
    )

    model = Flamingo(config)

    # Example: Few-shot visual QA
    # Images: [example1_image, example2_image, query_image]
    batch_size = 1
    num_images = 3
    images = torch.randn(batch_size, num_images, 3, 224, 224)

    # Encode images
    visual_tokens = model.encode_vision(images)

    print(f"\nModel Statistics:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Perceiver params: {sum(p.numel() for p in model.perceiver.parameters()):,}")
    print(f"Visual tokens shape: {visual_tokens.shape}")
    print(f"  -> {visual_tokens.shape[1]} images")
    print(f"  -> {visual_tokens.shape[2]} latent tokens per image")
    print(f"  -> {visual_tokens.shape[3]} dimensions")

    print("\n" + "="*80)
