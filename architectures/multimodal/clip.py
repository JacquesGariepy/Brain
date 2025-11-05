"""
CLIP (Contrastive Language-Image Pre-training)

SOTA vision-language model that learns visual concepts from natural language supervision.
Introduced by OpenAI in 2021, remains foundational for multimodal AI in 2025-2026.

Key features:
- Contrastive learning between image and text embeddings
- Zero-shot image classification
- Image-text retrieval
- Foundation for many downstream multimodal tasks

References:
- "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class CLIPConfig:
    """Configuration for CLIP model"""
    # Vision encoder
    image_size: int = 224
    patch_size: int = 16
    vision_width: int = 768
    vision_layers: int = 12
    vision_heads: int = 12

    # Text encoder
    vocab_size: int = 49408
    context_length: int = 77
    text_width: int = 512
    text_layers: int = 12
    text_heads: int = 8

    # Embedding
    embed_dim: int = 512

    # Training
    logit_scale_init: float = 2.6592  # ln(1/0.07)
    dropout: float = 0.0


class CLIPVisionEncoder(nn.Module):
    """
    Vision encoder for CLIP using Vision Transformer.

    Processes images into visual embeddings compatible with text embeddings.
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=config.vision_width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False
        )

        # Calculate number of patches
        self.num_patches = (config.image_size // config.patch_size) ** 2

        # Class token
        self.class_embedding = nn.Parameter(
            torch.randn(config.vision_width)
        )

        # Position embeddings
        self.positional_embedding = nn.Parameter(
            torch.randn(self.num_patches + 1, config.vision_width)
        )

        # Pre-LayerNorm
        self.ln_pre = nn.LayerNorm(config.vision_width)

        # Transformer blocks
        self.transformer = nn.ModuleList([
            VisionTransformerBlock(
                d_model=config.vision_width,
                num_heads=config.vision_heads,
                dropout=config.dropout
            )
            for _ in range(config.vision_layers)
        ])

        # Post-LayerNorm
        self.ln_post = nn.LayerNorm(config.vision_width)

        # Projection to embedding space
        self.proj = nn.Parameter(
            torch.randn(config.vision_width, config.embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for vision encoder.

        Args:
            x: Images (batch, 3, H, W)

        Returns:
            Image embeddings (batch, embed_dim)
        """
        # Patch embedding
        x = self.conv1(x)  # (batch, width, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, width)

        # Add class token
        batch_size = x.shape[0]
        class_token = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(
            batch_size, 1, -1
        )
        x = torch.cat([class_token, x], dim=1)  # (batch, num_patches+1, width)

        # Add position embeddings
        x = x + self.positional_embedding

        # Pre-norm
        x = self.ln_pre(x)

        # Transformer blocks
        for block in self.transformer:
            x = block(x)

        # Extract class token
        x = x[:, 0, :]  # (batch, width)

        # Post-norm
        x = self.ln_post(x)

        # Project to embedding space
        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionTransformerBlock(nn.Module):
    """Transformer block for vision encoder"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections"""
        # Self-attention
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        # MLP
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPTextEncoder(nn.Module):
    """
    Text encoder for CLIP using Transformer.

    Processes text into embeddings compatible with visual embeddings.
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.text_width)

        # Position embeddings
        self.positional_embedding = nn.Parameter(
            torch.randn(config.context_length, config.text_width)
        )

        # Transformer blocks
        self.transformer = nn.ModuleList([
            TextTransformerBlock(
                d_model=config.text_width,
                num_heads=config.text_heads,
                dropout=config.dropout
            )
            for _ in range(config.text_layers)
        ])

        # Final LayerNorm
        self.ln_final = nn.LayerNorm(config.text_width)

        # Projection to embedding space
        self.text_projection = nn.Parameter(
            torch.randn(config.text_width, config.embed_dim)
        )

    def forward(
        self,
        text: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for text encoder.

        Args:
            text: Text tokens (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Text embeddings (batch, embed_dim)
        """
        batch_size, seq_len = text.shape

        # Token embedding
        x = self.token_embedding(text)  # (batch, seq_len, text_width)

        # Add position embeddings
        x = x + self.positional_embedding[:seq_len]

        # Transformer blocks
        for block in self.transformer:
            x = block(x, attention_mask)

        # Final LayerNorm
        x = self.ln_final(x)

        # Extract features from EOT token (last token for each sequence)
        # Find the position of EOT token for each sequence
        if attention_mask is not None:
            # EOT is the last attended token
            eot_indices = attention_mask.sum(dim=1) - 1
        else:
            # Assume EOT is at the end
            eot_indices = torch.full((batch_size,), seq_len - 1, device=text.device)

        # Gather EOT token features
        x = x[torch.arange(batch_size), eot_indices]  # (batch, text_width)

        # Project to embedding space
        if self.text_projection is not None:
            x = x @ self.text_projection

        return x


class TextTransformerBlock(nn.Module):
    """Transformer block for text encoder with causal masking"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with causal attention"""
        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )

        # Combine with attention mask if provided
        if attention_mask is not None:
            # Convert padding mask to attention mask
            attn_mask = attention_mask.float().masked_fill(
                attention_mask == 0, float('-inf')
            )
            attn_mask = attn_mask.unsqueeze(1).expand(-1, seq_len, -1)
            causal_mask = causal_mask + attn_mask

        # Self-attention with causal mask
        x = x + self.attn(
            self.ln_1(x), self.ln_1(x), self.ln_1(x),
            attn_mask=causal_mask,
            need_weights=False
        )[0]

        # MLP
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIP(nn.Module):
    """
    Complete CLIP model for vision-language learning.

    Learns joint embeddings for images and text through contrastive learning.
    Can be used for:
    - Zero-shot image classification
    - Image-text retrieval
    - Visual reasoning
    - Image captioning
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        # Vision and text encoders
        self.visual = CLIPVisionEncoder(config)
        self.text = CLIPTextEncoder(config)

        # Learnable temperature parameter for contrastive loss
        self.logit_scale = nn.Parameter(
            torch.ones([]) * config.logit_scale_init
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            image: Images (batch, 3, H, W)

        Returns:
            Normalized image embeddings (batch, embed_dim)
        """
        image_features = self.visual(image)
        image_features = F.normalize(image_features, dim=-1)
        return image_features

    def encode_text(
        self,
        text: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text to embeddings.

        Args:
            text: Text tokens (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Normalized text embeddings (batch, embed_dim)
        """
        text_features = self.text(text, attention_mask)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing image and text features plus logits.

        Args:
            image: Images (batch, 3, H, W)
            text: Text tokens (batch, seq_len)
            attention_mask: Text attention mask (batch, seq_len)

        Returns:
            image_features: Image embeddings (batch, embed_dim)
            text_features: Text embeddings (batch, embed_dim)
            logit_scale: Temperature-scaled logits
        """
        # Get embeddings
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, attention_mask)

        # Return features and logit scale
        return image_features, text_features, self.logit_scale.exp()

    def compute_contrastive_loss(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute CLIP contrastive loss.

        Args:
            image: Images (batch, 3, H, W)
            text: Text tokens (batch, seq_len)
            attention_mask: Text attention mask

        Returns:
            loss: Contrastive loss
            metrics: Dictionary of metrics for logging
        """
        # Get features
        image_features, text_features, logit_scale = self.forward(
            image, text, attention_mask
        )

        # Compute similarity matrix
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        # Symmetric cross-entropy loss
        batch_size = image.shape[0]
        labels = torch.arange(batch_size, device=image.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        # Compute accuracy
        with torch.no_grad():
            pred_i = logits_per_image.argmax(dim=1)
            pred_t = logits_per_text.argmax(dim=1)
            acc_i = (pred_i == labels).float().mean()
            acc_t = (pred_t == labels).float().mean()

        metrics = {
            'loss': loss.item(),
            'loss_image': loss_i.item(),
            'loss_text': loss_t.item(),
            'accuracy_image': acc_i.item(),
            'accuracy_text': acc_t.item(),
            'logit_scale': logit_scale.item()
        }

        return loss, metrics

    def zero_shot_classifier(
        self,
        class_names: list,
        templates: list = None
    ) -> torch.Tensor:
        """
        Create zero-shot classifier from class names.

        Args:
            class_names: List of class names
            templates: List of prompt templates (default: ["a photo of a {}"])

        Returns:
            Classifier weights (num_classes, embed_dim)
        """
        if templates is None:
            templates = ["a photo of a {}."]

        # Tokenize all prompts (pseudo-code, needs actual tokenizer)
        # In practice, you'd use CLIP's tokenizer
        classifier_weights = []

        for class_name in class_names:
            # Average embeddings over all templates for this class
            class_embeddings = []
            for template in templates:
                # text = tokenize(template.format(class_name))
                # text_features = self.encode_text(text)
                # class_embeddings.append(text_features)
                pass

            # class_embedding = torch.stack(class_embeddings).mean(dim=0)
            # class_embedding = F.normalize(class_embedding, dim=-1)
            # classifier_weights.append(class_embedding)

        # classifier_weights = torch.stack(classifier_weights)
        # return classifier_weights
        pass

    def compute_similarity(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute cosine similarity between images and texts.

        Args:
            image: Images (batch_i, 3, H, W)
            text: Text tokens (batch_t, seq_len)
            attention_mask: Text attention mask

        Returns:
            Similarity matrix (batch_i, batch_t)
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, attention_mask)

        # Cosine similarity (features are already normalized)
        similarity = image_features @ text_features.T

        return similarity


# Example usage
if __name__ == "__main__":
    # Create CLIP model
    config = CLIPConfig(
        image_size=224,
        patch_size=16,
        vision_layers=12,
        text_layers=12,
        embed_dim=512
    )

    model = CLIP(config)

    # Example forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    text = torch.randint(0, config.vocab_size, (batch_size, config.context_length))

    # Compute loss
    loss, metrics = model.compute_contrastive_loss(images, text)

    print(f"CLIP Model")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
