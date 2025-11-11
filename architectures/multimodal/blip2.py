"""
BLIP-2 (Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models)

SOTA vision-language model that efficiently connects frozen vision encoders with frozen LLMs
using a lightweight Q-Former architecture.

Key innovations:
- Q-Former: Lightweight transformer that bridges vision and language modalities
- Frozen encoders: Leverages pre-trained vision and language models without fine-tuning
- Two-stage training: Vision-language representation learning + Vision-to-language generative learning
- Highly efficient compared to full fine-tuning approaches

Used as foundation for:
- InstructBLIP
- X-InstructBLIP
- Many multimodal LLMs

References:
- "BLIP-2: Bootstrapping Language-Image Pre-training" (Li et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class BLIP2Config:
    """Configuration for BLIP-2 model"""
    # Vision encoder (e.g., ViT)
    vision_encoder: str = "vit_l"  # vit_b, vit_l, vit_g
    image_size: int = 224
    freeze_vision_encoder: bool = True

    # Q-Former
    num_query_tokens: int = 32  # Learnable query tokens
    qformer_hidden_size: int = 768
    qformer_num_layers: int = 12
    qformer_num_heads: int = 12
    qformer_ffn_dim: int = 3072
    qformer_max_position_embeddings: int = 512
    qformer_dropout: float = 0.1

    # Language model (e.g., OPT, Flan-T5)
    language_model: str = "opt-2.7b"
    freeze_language_model: bool = True

    # Projection
    projection_dim: int = 768  # Dim to project to LM space


class QFormer(nn.Module):
    """
    Querying Transformer (Q-Former) - The core innovation of BLIP-2.

    A lightweight transformer that:
    1. Extracts visual features via learnable query tokens
    2. Bridges the modality gap between vision and language
    3. Performs cross-attention to image features
    4. Self-attention among queries and text

    The Q-Former is trained while keeping vision encoder and LM frozen,
    making it extremely parameter-efficient.
    """

    def __init__(self, config: BLIP2Config):
        super().__init__()
        self.config = config

        # Learnable query tokens - These are the key to Q-Former
        # They extract the most relevant visual information
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_hidden_size)
        )
        nn.init.normal_(self.query_tokens, std=0.02)

        # Position embeddings for text tokens
        self.position_embeddings = nn.Embedding(
            config.qformer_max_position_embeddings,
            config.qformer_hidden_size
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            QFormerLayer(
                hidden_size=config.qformer_hidden_size,
                num_heads=config.qformer_num_heads,
                ffn_dim=config.qformer_ffn_dim,
                dropout=config.qformer_dropout
            )
            for _ in range(config.qformer_num_layers)
        ])

        # Layer norm
        self.layernorm = nn.LayerNorm(config.qformer_hidden_size)

    def forward(
        self,
        text_embeddings: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of Q-Former.

        Args:
            text_embeddings: Optional text embeddings (batch, text_len, hidden_size)
            image_features: Image features from vision encoder (batch, num_patches, vision_hidden_size)
            text_attention_mask: Attention mask for text
            return_all_tokens: Whether to return all query tokens or just average

        Returns:
            Query representations (batch, num_queries, hidden_size) or (batch, hidden_size)
        """
        batch_size = image_features.shape[0] if image_features is not None else text_embeddings.shape[0]

        # Expand query tokens for batch
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        # Optionally concatenate with text embeddings
        if text_embeddings is not None:
            # Add position embeddings to text
            seq_length = text_embeddings.shape[1]
            position_ids = torch.arange(seq_length, device=text_embeddings.device)
            position_embeddings = self.position_embeddings(position_ids)
            text_embeddings = text_embeddings + position_embeddings.unsqueeze(0)

            # Concatenate queries and text
            embeddings = torch.cat([query_tokens, text_embeddings], dim=1)

            # Update attention mask
            query_mask = torch.ones(
                batch_size, self.config.num_query_tokens,
                device=embeddings.device
            )
            if text_attention_mask is not None:
                attention_mask = torch.cat([query_mask, text_attention_mask], dim=1)
            else:
                attention_mask = None
        else:
            embeddings = query_tokens
            attention_mask = None

        # Pass through Q-Former layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states=image_features,
                attention_mask=attention_mask
            )

        # Apply final layer norm
        hidden_states = self.layernorm(hidden_states)

        # Extract query representations (first num_query_tokens)
        query_output = hidden_states[:, :self.config.num_query_tokens, :]

        if return_all_tokens:
            return query_output
        else:
            # Return mean-pooled representation
            return query_output.mean(dim=1)


class QFormerLayer(nn.Module):
    """Single layer of Q-Former with self-attention and cross-attention"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_size)

        # Cross-attention to image features
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

        self.dropout = dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with self-attention, cross-attention, and FFN.

        Args:
            hidden_states: Query and text embeddings
            encoder_hidden_states: Image features for cross-attention
            attention_mask: Attention mask

        Returns:
            Updated hidden states
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=None if attention_mask is None else (attention_mask == 0),
            need_weights=False
        )
        hidden_states = residual + F.dropout(hidden_states, p=self.dropout, training=self.training)

        # Cross-attention to image features
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_norm(hidden_states)
            hidden_states, _ = self.cross_attn(
                hidden_states, encoder_hidden_states, encoder_hidden_states,
                need_weights=False
            )
            hidden_states = residual + F.dropout(hidden_states, p=self.dropout, training=self.training)

        # Feed-forward network
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class BLIP2(nn.Module):
    """
    Complete BLIP-2 model for vision-language understanding and generation.

    Architecture:
    1. Frozen vision encoder (e.g., ViT-g)
    2. Q-Former that bridges vision and language
    3. Frozen language model (e.g., OPT, Flan-T5)

    Training stages:
    1. Vision-language representation learning with ITC, ITM, ITG losses
    2. Vision-to-language generative learning

    Can be used for:
    - Visual question answering
    - Image captioning
    - Visual reasoning
    - Instruction following (InstructBLIP)
    """

    def __init__(
        self,
        config: BLIP2Config,
        vision_encoder: Optional[nn.Module] = None,
        language_model: Optional[nn.Module] = None
    ):
        super().__init__()
        self.config = config

        # Vision encoder (frozen)
        if vision_encoder is None:
            # Default: Use a vision transformer
            from ..vision.vision_transformer import VisionTransformer, ViTConfig
            vit_config = ViTConfig(
                image_size=config.image_size,
                patch_size=16,
                d_model=1024 if config.vision_encoder == "vit_l" else 1408,
                num_layers=24 if config.vision_encoder == "vit_l" else 40,
                num_heads=16 if config.vision_encoder == "vit_l" else 16
            )
            self.vision_encoder = VisionTransformer(vit_config)
        else:
            self.vision_encoder = vision_encoder

        if config.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # Q-Former - The core of BLIP-2
        self.qformer = QFormer(config)

        # Projection layer to match language model dimension
        vision_hidden_size = self.vision_encoder.config.d_model if hasattr(self.vision_encoder, 'config') else 1024
        self.vision_projection = nn.Linear(vision_hidden_size, config.qformer_hidden_size)

        # Language model projection
        self.language_projection = nn.Linear(
            config.qformer_hidden_size,
            config.projection_dim
        )

        # Language model (frozen)
        self.language_model = language_model
        if config.freeze_language_model and language_model is not None:
            for param in self.language_model.parameters():
                param.requires_grad = False

        # Token embeddings for Q-Former text input
        self.text_embeddings = nn.Embedding(30522, config.qformer_hidden_size)  # BERT vocab size

    def extract_visual_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features using frozen vision encoder.

        Args:
            images: Input images (batch, 3, H, W)

        Returns:
            Visual features (batch, num_patches, vision_hidden_size)
        """
        with torch.no_grad() if self.config.freeze_vision_encoder else torch.enable_grad():
            # Get patch features from vision encoder
            vision_outputs = self.vision_encoder(images, return_all_tokens=True)

        # Project to Q-Former dimension
        visual_features = self.vision_projection(vision_outputs)

        return visual_features

    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of BLIP-2.

        Args:
            images: Input images (batch, 3, H, W)
            text_input_ids: Optional text input IDs
            text_attention_mask: Optional text attention mask
            return_dict: Whether to return dictionary

        Returns:
            Dictionary containing:
            - visual_features: Features from Q-Former
            - language_model_outputs: Optional LM outputs
        """
        # Extract visual features
        visual_features = self.extract_visual_features(images)

        # Get text embeddings if text is provided
        text_embeddings = None
        if text_input_ids is not None:
            text_embeddings = self.text_embeddings(text_input_ids)

        # Pass through Q-Former
        query_outputs = self.qformer(
            text_embeddings=text_embeddings,
            image_features=visual_features,
            text_attention_mask=text_attention_mask,
            return_all_tokens=True
        )

        # Project to language model space
        language_model_inputs = self.language_projection(query_outputs)

        outputs = {
            'visual_features': query_outputs,
            'language_model_inputs': language_model_inputs
        }

        # Optionally pass through language model
        if self.language_model is not None and text_input_ids is not None:
            # Concatenate visual features with text embeddings
            # and pass through frozen LM
            lm_outputs = self._generate_with_language_model(
                language_model_inputs,
                text_input_ids
            )
            outputs['language_model_outputs'] = lm_outputs

        return outputs if return_dict else (query_outputs, language_model_inputs)

    def _generate_with_language_model(
        self,
        visual_inputs: torch.Tensor,
        text_input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate text using frozen language model conditioned on visual inputs.

        Args:
            visual_inputs: Visual features from Q-Former
            text_input_ids: Text input IDs

        Returns:
            Language model outputs
        """
        # This is a placeholder - actual implementation depends on the LM architecture
        # For OPT/Llama: prepend visual tokens to text tokens
        # For T5: use visual tokens as encoder outputs
        if self.language_model is not None:
            with torch.no_grad() if self.config.freeze_language_model else torch.enable_grad():
                outputs = self.language_model(text_input_ids)
            return outputs
        return None

    def generate(
        self,
        images: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        max_length: int = 50,
        num_beams: int = 1,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate text captions or responses for images.

        Args:
            images: Input images (batch, 3, H, W)
            prompt: Optional text prompt
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature

        Returns:
            Generated token IDs
        """
        # Extract visual features
        visual_features = self.extract_visual_features(images)

        # Process through Q-Former
        query_outputs = self.qformer(
            image_features=visual_features,
            return_all_tokens=True
        )

        # Project to LM space
        language_model_inputs = self.language_projection(query_outputs)

        # Generate with language model
        # (Placeholder - needs actual generation implementation)
        if self.language_model is not None:
            # Use LM's generate method with visual conditioning
            pass

        return None

    def compute_contrastive_loss(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute contrastive loss between image and text (ITC loss in BLIP-2).

        Args:
            images: Input images
            text_input_ids: Text token IDs
            text_attention_mask: Text attention mask

        Returns:
            loss: Contrastive loss
            metrics: Logging metrics
        """
        batch_size = images.shape[0]

        # Get visual and text features
        outputs = self.forward(
            images,
            text_input_ids,
            text_attention_mask,
            return_dict=True
        )

        visual_features = outputs['visual_features'].mean(dim=1)  # Pool query tokens
        text_features = outputs['language_model_inputs'].mean(dim=1)

        # Normalize
        visual_features = F.normalize(visual_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute similarity
        logits = visual_features @ text_features.T

        # Labels are diagonal (matching pairs)
        labels = torch.arange(batch_size, device=logits.device)

        # Symmetric loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2

        metrics = {
            'contrastive_loss': loss.item(),
            'loss_i2t': loss_i2t.item(),
            'loss_t2i': loss_t2i.item()
        }

        return loss, metrics


# Example usage
if __name__ == "__main__":
    # Create BLIP-2 model
    config = BLIP2Config(
        vision_encoder="vit_l",
        num_query_tokens=32,
        qformer_num_layers=12
    )

    model = BLIP2(config)

    # Example inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    text_ids = torch.randint(0, 30522, (batch_size, 32))

    # Forward pass
    outputs = model(images, text_ids)

    print(f"BLIP-2 Model")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters (Q-Former): {sum(p.numel() for p in model.qformer.parameters()):,}")
    print(f"Visual features shape: {outputs['visual_features'].shape}")
    print(f"Language model inputs shape: {outputs['language_model_inputs'].shape}")
