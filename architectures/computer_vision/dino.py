"""
DINOv2 - Self-Distillation with No Labels v2

SOTA self-supervised vision learning (Meta AI, 2023).

Key features:
- Vision Transformer (ViT) backbone
- Self-supervised learning (no labels needed)
- Student-teacher framework with momentum
- Multi-crop training strategy
- Sinkhorn-Knopp centering for stability
- Strong feature representations
- Works at multiple scales (small, base, large, giant)

Architecture:
- Student: ViT that learns from teacher
- Teacher: EMA of student weights
- Projection head: Maps features to embedding space
- Loss: Cross-entropy between student and teacher

Applications:
- Feature extraction for downstream tasks
- Zero-shot classification
- Dense prediction (segmentation, depth)
- Image retrieval

References:
- "DINOv2: Learning Robust Visual Features without Supervision" (Oquab et al., 2023)
- Meta AI Research (FAIR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


@dataclass
class DINOv2Config:
    """Configuration for DINOv2"""
    # Architecture
    model_size: str = 'base'  # small, base, large, giant
    image_size: int = 518  # DINOv2 uses 518x518
    patch_size: int = 14
    num_classes: int = 0  # For self-supervised learning

    # Vision Transformer
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0

    # Self-supervised learning
    out_dim: int = 65536  # Output dimension for DINO head
    use_bn_in_head: bool = False
    norm_last_layer: bool = True

    # Teacher
    momentum_teacher: float = 0.996  # EMA momentum
    teacher_temp: float = 0.04  # Teacher temperature
    warmup_teacher_temp: float = 0.04
    warmup_teacher_temp_epochs: int = 30

    # Student
    student_temp: float = 0.1

    # Multi-crop
    local_crops_number: int = 8
    global_crops_scale: Tuple[float, float] = (0.4, 1.0)
    local_crops_scale: Tuple[float, float] = (0.05, 0.4)

    def __post_init__(self):
        # Set dimensions based on model size
        size_configs = {
            'small': (384, 6, 12),    # (embed_dim, num_heads, num_layers)
            'base': (768, 12, 12),
            'large': (1024, 16, 24),
            'giant': (1536, 24, 40)
        }

        if self.model_size in size_configs:
            self.embed_dim, self.num_heads, self.num_layers = size_configs[self.model_size]


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    Splits image into patches and linearly embeds them.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images (batch, 3, H, W)

        Returns:
            Patches (batch, num_patches, embed_dim)
        """
        x = self.proj(x)  # (batch, embed_dim, H', W')
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention.

    Standard transformer attention with optional QKV bias.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)

        Returns:
            Output (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch, seq_len, dim)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    MLP block with GELU activation.

    Standard transformer FFN.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Transformer block.

    Self-attention + MLP with residual connections and LayerNorm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)

        Returns:
            Output (batch, seq_len, dim)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer backbone for DINOv2.

    Standard ViT with:
    - Patch embedding
    - Positional embedding
    - CLS token
    - Transformer blocks
    """

    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbed(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=3,
            embed_dim=config.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embed_dim)
        )
        self.pos_drop = nn.Dropout(p=config.drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate
            )
            for _ in range(config.num_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(config.embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images (batch, 3, H, W)

        Returns:
            CLS token features (batch, embed_dim)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        # Return CLS token
        return x[:, 0]

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 1
    ) -> List[torch.Tensor]:
        """
        Get features from last n layers.

        Useful for dense prediction tasks.
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Collect outputs from last n layers
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i >= len(self.blocks) - n:
                outputs.append(self.norm(x))

        return outputs


class DINOHead(nn.Module):
    """
    Projection head for DINO.

    Maps backbone features to embedding space for self-supervised learning.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256
    ):
        super().__init__()
        nlayers = max(nlayers, 1)

        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())

            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (batch, in_dim)

        Returns:
            Embeddings (batch, out_dim)
        """
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOv2(nn.Module):
    """
    Complete DINOv2 model.

    Self-supervised visual learning with:
    - Student-teacher framework
    - Multi-crop training
    - Strong feature representations
    """

    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.config = config

        # Student network
        self.student_backbone = VisionTransformer(config)
        self.student_head = DINOHead(
            in_dim=config.embed_dim,
            out_dim=config.out_dim,
            use_bn=config.use_bn_in_head,
            norm_last_layer=config.norm_last_layer
        )

        # Teacher network (no gradients)
        self.teacher_backbone = VisionTransformer(config)
        self.teacher_head = DINOHead(
            in_dim=config.embed_dim,
            out_dim=config.out_dim,
            use_bn=config.use_bn_in_head
        )

        # Teacher is initialized with student weights
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        # Disable gradients for teacher
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """
        Update teacher with EMA of student weights.

        Args:
            momentum: EMA momentum (typically 0.996-0.999)
        """
        for param_student, param_teacher in zip(
            self.student_backbone.parameters(),
            self.teacher_backbone.parameters()
        ):
            param_teacher.data.mul_(momentum).add_(
                param_student.data,
                alpha=1 - momentum
            )

        for param_student, param_teacher in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters()
        ):
            param_teacher.data.mul_(momentum).add_(
                param_student.data,
                alpha=1 - momentum
            )

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass (for inference).

        Args:
            images: Images (batch, 3, H, W)
            return_features: Return backbone features instead of head output

        Returns:
            Features or embeddings (batch, dim)
        """
        features = self.student_backbone(images)

        if return_features:
            return features
        else:
            return self.student_head(features)

    def forward_train(
        self,
        global_crops: torch.Tensor,
        local_crops: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with multi-crop.

        Args:
            global_crops: Global views (batch, 2, 3, H, W)
            local_crops: Local views (batch, N, 3, H, W)

        Returns:
            Dictionary with student and teacher outputs
        """
        batch_size = global_crops.shape[0]

        # Flatten crops
        student_crops = global_crops.flatten(0, 1)  # (batch*2, 3, H, W)
        if local_crops is not None:
            num_local = local_crops.shape[1]
            local_crops_flat = local_crops.flatten(0, 1)
            student_crops = torch.cat([student_crops, local_crops_flat], dim=0)

        # Student forward
        student_features = self.student_backbone(student_crops)
        student_output = self.student_head(student_features)

        # Teacher forward (only global crops)
        with torch.no_grad():
            teacher_crops = global_crops.flatten(0, 1)
            teacher_features = self.teacher_backbone(teacher_crops)
            teacher_output = self.teacher_head(teacher_features)

        return {
            'student_output': student_output,
            'teacher_output': teacher_output,
            'student_features': student_features,
            'teacher_features': teacher_features
        }

    def compute_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        student_temp: float,
        teacher_temp: float,
        epoch: int
    ) -> torch.Tensor:
        """
        Compute DINO loss (cross-entropy between student and teacher).

        Args:
            student_output: Student predictions (batch*(2+local), out_dim)
            teacher_output: Teacher predictions (batch*2, out_dim)
            student_temp: Student temperature
            teacher_temp: Teacher temperature
            epoch: Current epoch

        Returns:
            Loss scalar
        """
        # Softmax with temperature
        student_out = student_output / student_temp
        student_out = student_out.chunk(2 + self.config.local_crops_number)

        teacher_out = F.softmax(teacher_output / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        # Compute cross-entropy
        total_loss = 0
        n_loss_terms = 0

        for t_idx, teacher_view in enumerate(teacher_out):
            for s_idx, student_view in enumerate(student_out):
                if t_idx == s_idx:
                    # Don't compare same view
                    continue

                loss = torch.sum(
                    -teacher_view * F.log_softmax(student_view, dim=-1),
                    dim=-1
                )
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms

        return total_loss

    def extract_features(
        self,
        images: torch.Tensor,
        layers: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Extract features for downstream tasks.

        Args:
            images: Images (batch, 3, H, W)
            layers: Which layers to extract (default: last layer)

        Returns:
            Features (batch, embed_dim)
        """
        self.eval()
        with torch.no_grad():
            if layers is None:
                return self.student_backbone(images)
            else:
                return self.student_backbone.get_intermediate_layers(
                    images,
                    n=len(layers)
                )


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("DINOv2 - Self-Supervised Visual Learning")
    print("="*80)

    # Create DINOv2 models of different sizes
    for size in ['small', 'base']:
        config = DINOv2Config(
            model_size=size,
            image_size=518,
            patch_size=14
        )

        model = DINOv2(config)

        print(f"\nDINOv2-{size}:")
        print(f"  Embed dim: {config.embed_dim}")
        print(f"  Num layers: {config.num_layers}")
        print(f"  Num heads: {config.num_heads}")

        # Test inference
        batch_size = 2
        images = torch.randn(batch_size, 3, 518, 518)

        print(f"\n  Input shape: {images.shape}")

        # Extract features
        features = model(images, return_features=True)
        print(f"  Features shape: {features.shape}")

        # Get embeddings
        embeddings = model(images, return_features=False)
        print(f"  Embeddings shape: {embeddings.shape}")

        num_params = sum(p.numel() for p in model.student_backbone.parameters())
        print(f"  Backbone parameters: {num_params:,}")

    print("\n" + "="*80)
