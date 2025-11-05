"""
LLaVA (Large Language and Vision Assistant)

One of the most successful open-source vision-language models (2023-2025).
Simple yet effective approach: connects vision encoder to LLM via projection layer.

Key innovations:
- Simple projection layer instead of complex Q-Former
- Instruction-following training with GPT-4 generated data
- Excellent performance on visual reasoning tasks
- Highly scalable and easy to train

Variants:
- LLaVA-1.5 (improved training)
- LLaVA-NeXT (higher resolution, better reasoning)
- LLaVA-OneVision (unified image/video understanding)

References:
- "Visual Instruction Tuning" (Liu et al., 2023)
- LLaVA-1.5, LLaVA-NeXT papers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class LLaVAConfig:
    """Configuration for LLaVA model"""
    # Vision encoder
    vision_encoder: str = "clip_vit_l"  # clip_vit_l, clip_vit_g, siglip
    image_size: int = 336  # LLaVA-1.5 uses 336x336
    freeze_vision_encoder: bool = True

    # Projection
    projection_type: str = "mlp"  # linear, mlp (2-layer MLP in LLaVA-1.5)
    mm_hidden_size: int = 4096  # Projection hidden size

    # Language model
    language_model: str = "vicuna-7b"  # or llama-2-7b, mistral-7b, etc.
    language_model_hidden_size: int = 4096
    freeze_language_model: bool = False  # Usually fine-tune

    # Special tokens
    image_token_index: int = -200  # Special token for <image>
    ignore_index: int = -100  # For loss computation

    # Training
    tune_mm_mlp_adapter: bool = True
    tune_language_model: bool = True


class MultimodalProjector(nn.Module):
    """
    Projection layer that connects vision encoder to language model.

    LLaVA-1.5 uses a 2-layer MLP with GELU activation.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        language_hidden_size: int,
        projection_type: str = "mlp"
    ):
        super().__init__()
        self.projection_type = projection_type

        if projection_type == "linear":
            self.projector = nn.Linear(vision_hidden_size, language_hidden_size)
        elif projection_type == "mlp":
            # 2-layer MLP (LLaVA-1.5 approach)
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, language_hidden_size),
                nn.GELU(),
                nn.Linear(language_hidden_size, language_hidden_size)
            )
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to language model space.

        Args:
            x: Vision features (batch, num_patches, vision_hidden_size)

        Returns:
            Projected features (batch, num_patches, language_hidden_size)
        """
        return self.projector(x)


class LLaVA(nn.Module):
    """
    LLaVA: Large Language and Vision Assistant.

    Simple and effective architecture:
    1. Vision encoder extracts image features
    2. Projection layer aligns features to LM space
    3. Visual tokens are inserted into text sequence
    4. LM processes the combined sequence

    Usage:
    - Visual question answering
    - Image captioning
    - Visual reasoning
    - Instruction following with images
    """

    def __init__(
        self,
        config: LLaVAConfig,
        vision_encoder: Optional[nn.Module] = None,
        language_model: Optional[nn.Module] = None
    ):
        super().__init__()
        self.config = config

        # Vision encoder (typically CLIP ViT)
        if vision_encoder is None:
            from ..vision.vision_transformer import VisionTransformer, ViTConfig
            vit_config = ViTConfig(
                image_size=config.image_size,
                patch_size=14,  # CLIP uses 14x14 patches
                d_model=1024,  # CLIP-ViT-L
                num_layers=24,
                num_heads=16
            )
            self.vision_encoder = VisionTransformer(vit_config)
            vision_hidden_size = 1024
        else:
            self.vision_encoder = vision_encoder
            vision_hidden_size = vision_encoder.config.d_model

        # Freeze vision encoder if specified
        if config.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # Multimodal projector
        self.mm_projector = MultimodalProjector(
            vision_hidden_size=vision_hidden_size,
            language_hidden_size=config.language_model_hidden_size,
            projection_type=config.projection_type
        )

        # Language model (can be any decoder-only LM)
        self.language_model = language_model
        if config.freeze_language_model and language_model is not None:
            for param in self.language_model.parameters():
                param.requires_grad = False

        # Special token embeddings
        self.image_newline = nn.Parameter(
            torch.randn(config.language_model_hidden_size)
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to features aligned with language model.

        Args:
            images: Input images (batch, 3, H, W)

        Returns:
            Image features in LM space (batch, num_patches, lm_hidden_size)
        """
        # Extract vision features (frozen or trainable)
        with torch.set_grad_enabled(not self.config.freeze_vision_encoder):
            if hasattr(self.vision_encoder, 'forward_features'):
                image_features = self.vision_encoder.forward_features(images)
            else:
                image_features = self.vision_encoder(images, return_all_tokens=True)

        # Project to language model space
        image_features = self.mm_projector(image_features)

        return image_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare inputs by inserting image features into text sequence.

        This is the key function that merges vision and language.
        Image tokens in the text are replaced with actual visual features.

        Args:
            input_ids: Text token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            images: Optional images (batch, 3, H, W)
            labels: Optional labels for training (batch, seq_len)

        Returns:
            inputs_embeds: Combined embeddings (batch, new_seq_len, hidden_size)
            attention_mask: Updated attention mask (batch, new_seq_len)
            labels: Updated labels (batch, new_seq_len) or None
        """
        if images is None:
            # Text-only mode
            if self.language_model is not None:
                inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            else:
                inputs_embeds = input_ids
            return inputs_embeds, attention_mask, labels

        # Encode images
        image_features = self.encode_images(images)
        batch_size, num_patches, hidden_size = image_features.shape

        # Get text embeddings
        if self.language_model is not None:
            text_embeds = self.language_model.get_input_embeddings()(input_ids)
        else:
            # Placeholder if LM not provided
            text_embeds = torch.randn(
                input_ids.shape[0], input_ids.shape[1],
                self.config.language_model_hidden_size,
                device=input_ids.device
            )

        # Find positions of image tokens
        image_token_mask = input_ids == self.config.image_token_index

        # Replace image tokens with image features
        new_inputs_embeds = []
        new_labels = [] if labels is not None else None

        for batch_idx in range(batch_size):
            # Get positions where image tokens appear
            image_positions = torch.where(image_token_mask[batch_idx])[0]

            if len(image_positions) == 0:
                # No image tokens, use text only
                new_inputs_embeds.append(text_embeds[batch_idx])
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue

            # Build new sequence with image features inserted
            cur_text_embeds = text_embeds[batch_idx]
            cur_image_features = image_features[batch_idx]

            # Split text at image token position
            image_pos = image_positions[0].item()

            # Concatenate: text_before + image_features + text_after
            new_embed = torch.cat([
                cur_text_embeds[:image_pos],
                cur_image_features,
                cur_text_embeds[image_pos + 1:]
            ], dim=0)

            new_inputs_embeds.append(new_embed)

            # Update labels if provided
            if labels is not None:
                cur_labels = labels[batch_idx]
                # Image tokens don't contribute to loss
                image_labels = torch.full(
                    (num_patches,),
                    self.config.ignore_index,
                    device=cur_labels.device,
                    dtype=cur_labels.dtype
                )
                new_label = torch.cat([
                    cur_labels[:image_pos],
                    image_labels,
                    cur_labels[image_pos + 1:]
                ], dim=0)
                new_labels.append(new_label)

        # Pad sequences to same length
        max_len = max(x.shape[0] for x in new_inputs_embeds)

        inputs_embeds = torch.stack([
            F.pad(x, (0, 0, 0, max_len - x.shape[0]), value=0)
            for x in new_inputs_embeds
        ])

        # Update attention mask
        new_attention_mask = torch.zeros(
            batch_size, max_len,
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        for i, x in enumerate(new_inputs_embeds):
            new_attention_mask[i, :x.shape[0]] = 1

        # Update labels
        if labels is not None:
            new_labels_padded = torch.stack([
                F.pad(x, (0, max_len - x.shape[0]), value=self.config.ignore_index)
                for x in new_labels
            ])
        else:
            new_labels_padded = None

        return inputs_embeds, new_attention_mask, new_labels_padded

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of LLaVA.

        Args:
            input_ids: Text token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            images: Images (batch, 3, H, W)
            labels: Labels for language modeling (batch, seq_len)
            return_dict: Whether to return dictionary

        Returns:
            Dictionary with outputs including loss and logits
        """
        # Prepare multimodal inputs
        inputs_embeds, attention_mask, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            labels=labels
        )

        # Forward through language model
        if self.language_model is not None:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

            return {
                'loss': outputs.get('loss', None),
                'logits': outputs.get('logits', None),
                'hidden_states': outputs.get('hidden_states', None)
            }
        else:
            # Return embeddings if no LM
            return {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
                'labels': labels
            }

    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text conditioned on images and text prompt.

        Args:
            input_ids: Input text tokens (batch, seq_len)
            images: Input images (batch, 3, H, W)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample

        Returns:
            Generated token IDs (batch, seq_len + new_tokens)
        """
        # Prepare inputs with images
        attention_mask = torch.ones_like(input_ids)
        inputs_embeds, attention_mask, _ = self.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images
        )

        # Generate with language model
        if self.language_model is not None and hasattr(self.language_model, 'generate'):
            outputs = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **kwargs
            )
            return outputs
        else:
            # Placeholder: implement basic greedy generation
            return self._greedy_generate(inputs_embeds, attention_mask, max_new_tokens)

    def _greedy_generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int
    ) -> torch.Tensor:
        """Placeholder for basic greedy generation"""
        # This would need a full implementation with the language model
        batch_size = inputs_embeds.shape[0]
        return torch.zeros(batch_size, max_new_tokens, dtype=torch.long)


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("LLaVA - Large Language and Vision Assistant")
    print("="*80)

    # Create LLaVA model
    config = LLaVAConfig(
        vision_encoder="clip_vit_l",
        image_size=336,
        projection_type="mlp",
        language_model_hidden_size=4096
    )

    model = LLaVA(config)

    # Example inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 336, 336)

    # Text with image token: "USER: <image>\nWhat is in this image? ASSISTANT:"
    input_ids = torch.randint(0, 32000, (batch_size, 50))
    input_ids[:, 5] = config.image_token_index  # Insert image token

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        images=images
    )

    print(f"\nModel Statistics:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vision encoder params: {sum(p.numel() for p in model.vision_encoder.parameters()):,}")
    print(f"MM projector params: {sum(p.numel() for p in model.mm_projector.parameters()):,}")

    if 'inputs_embeds' in outputs:
        print(f"Output embeddings shape: {outputs['inputs_embeds'].shape}")

    print("\n" + "="*80)
