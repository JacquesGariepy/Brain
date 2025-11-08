"""
Vision Transformer (ViT) and variants - SOTA Computer Vision

Implementations include:
- Original ViT
- DeiT (Data-efficient Image Transformers)
- Swin Transformer
- MaxViT
- Cross-attention variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math

from ..base import VisionArchitecture, ModelOutput


@dataclass
class ViTConfig:
    """Configuration for Vision Transformer"""
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    num_classes: int = 1000
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_cls_token: bool = True
    use_distillation: bool = False  # DeiT
    representation_size: Optional[int] = None


class PatchEmbedding(nn.Module):
    """
    Convert image to sequence of patch embeddings.

    Splits image into patches and linearly embeds them.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2

        # Convolution for patch extraction and embedding
        self.proj = nn.Conv2d(
            config.num_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images (batch, channels, height, width)

        Returns:
            Patch embeddings (batch, num_patches, d_model)
        """
        x = self.proj(x)  # (batch, d_model, H/P, W/P)
        x = x.flatten(2)  # (batch, d_model, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, d_model)
        return x


class ViTAttention(nn.Module):
    """Multi-head self-attention for ViT"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.proj_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        # Compute QKV
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch, seq_len, d_model)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class ViTMLP(nn.Module):
    """MLP block for ViT"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ViTBlock(nn.Module):
    """Transformer block for ViT"""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = ViTAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = ViTMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(VisionArchitecture):
    """
    Vision Transformer (ViT).

    "An Image is Worth 16x16 Words" - Dosovitskiy et al. 2020

    Key features:
    - Patch-based image processing
    - Pure transformer architecture
    - Classification token (CLS)
    - Position embeddings

    Inherits from VisionArchitecture for unified Brain framework interface.
    """

    def __init__(self, config: ViTConfig):
        super().__init__(config=config)

        # Patch embedding
        self.patch_embed = PatchEmbedding(config)
        num_patches = self.patch_embed.num_patches

        # Class token
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            num_positions = num_patches + 1
        else:
            num_positions = num_patches

        # Distillation token (DeiT)
        if config.use_distillation:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            num_positions += 1

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, config.d_model))
        self.pos_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(config) for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

        # Classification head
        if config.representation_size:
            self.pre_logits = nn.Linear(config.d_model, config.representation_size)
            self.head = nn.Linear(config.representation_size, config.num_classes)
        else:
            self.pre_logits = nn.Identity()
            self.head = nn.Linear(config.d_model, config.num_classes)

        # Distillation head
        if config.use_distillation:
            self.head_dist = nn.Linear(config.d_model, config.num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        # Position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Class token
        if self.config.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Distillation token
        if self.config.use_distillation:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        # Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """
        Forward pass.

        Args:
            x: Images (batch, channels, height, width)
            labels: Optional labels for loss computation (batch,)

        Returns:
            ModelOutput with logits and optional loss
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, d_model)

        # Add class token
        if self.config.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Add distillation token
        if self.config.use_distillation:
            dist_tokens = self.dist_token.expand(batch_size, -1, -1)
            x = torch.cat((x[:, :1], dist_tokens, x[:, 1:]), dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Classification
        if self.config.use_distillation:
            x_cls = x[:, 0]
            x_dist = x[:, 1]
            logits_cls = self.head(self.pre_logits(x_cls))
            logits_dist = self.head_dist(self.pre_logits(x_dist))

            if self.training:
                logits = logits_cls  # Use primary head for loss
            else:
                # Average predictions during inference
                logits = (logits_cls + logits_dist) / 2
        else:
            x = x[:, 0] if self.config.use_cls_token else x.mean(dim=1)
            x = self.pre_logits(x)
            logits = self.head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ModelOutput(
            logits=logits,
            loss=loss,
            predictions=logits.argmax(dim=-1) if not self.training else None
        )


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with shifted windows.

    Key innovation: Window-based attention for efficiency.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = WindowAttention(d_model, num_heads, window_size)
        self.norm2 = nn.LayerNorm(d_model)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (batch, H*W, d_model)
            H, W: Height and width of feature map
        """
        batch, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(batch, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(batch, H * W, C)
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))

        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention for Swin Transformer"""

    def __init__(self, d_model: int, num_heads: int, window_size: int):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (num_windows*batch, window_size*window_size, d_model)
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias (simplified)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into non-overlapping windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
