"""
SAM (Segment Anything Model)

Meta's foundation model for image segmentation (2023-2025).
Zero-shot transfer to new visual concepts and tasks.

Key features:
- Promptable segmentation (points, boxes, masks, text)
- Zero-shot generalization
- Real-time performance
- High-quality masks
- Foundation model for segmentation

Architecture:
- Image encoder: ViT-H/L/B (MAE pre-trained)
- Prompt encoder: Sparse (points/boxes) + Dense (masks)
- Mask decoder: Transformer with IoU prediction

References:
- "Segment Anything" (Kirillov et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class SAMConfig:
    """Configuration for SAM model"""
    # Image encoder (ViT)
    image_size: int = 1024
    patch_size: int = 16
    encoder_embed_dim: int = 768  # 768 (Base), 1024 (Large), 1280 (Huge)
    encoder_depth: int = 12
    encoder_num_heads: int = 12
    encoder_global_attn_indexes: List[int] = None  # Layers with global attention

    # Prompt encoder
    prompt_embed_dim: int = 256
    mask_in_chans: int = 16

    # Mask decoder
    num_multimask_outputs: int = 3  # Number of mask predictions
    iou_head_depth: int = 3
    iou_head_hidden_dim: int = 256

    def __post_init__(self):
        if self.encoder_global_attn_indexes is None:
            # Add global attention at specific layers (2, 5, 8, 11 for base)
            self.encoder_global_attn_indexes = [2, 5, 8, 11]


class ImageEncoder(nn.Module):
    """
    Vision Transformer image encoder for SAM.

    Uses MAE (Masked Autoencoder) pre-trained ViT.
    """

    def __init__(self, config: SAMConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, config.encoder_embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # Position embeddings
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, config.encoder_embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=config.encoder_embed_dim,
                num_heads=config.encoder_num_heads,
                use_global_attn=(i in config.encoder_global_attn_indexes)
            )
            for i in range(config.encoder_depth)
        ])

        # Neck (output projection)
        self.neck = nn.Sequential(
            nn.Conv2d(
                config.encoder_embed_dim,
                config.prompt_embed_dim,
                kernel_size=1,
                bias=False
            ),
            nn.LayerNorm(config.prompt_embed_dim),
            nn.Conv2d(
                config.prompt_embed_dim,
                config.prompt_embed_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.LayerNorm(config.prompt_embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image.

        Args:
            x: Image (batch, 3, H, W)

        Returns:
            Image features (batch, prompt_embed_dim, H/16, W/16)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (batch, embed_dim, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)

        # Add position embeddings
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Reshape to spatial
        batch_size, num_patches, embed_dim = x.shape
        h = w = int(num_patches ** 0.5)
        x = x.transpose(1, 2).reshape(batch_size, embed_dim, h, w)

        # Neck
        x = self.neck(x)

        return x


class ViTBlock(nn.Module):
    """Vision Transformer block with optional global attention"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_global_attn: bool = False
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.use_global_attn = use_global_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = shortcut + x

        # MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x

        return x


class PromptEncoder(nn.Module):
    """
    Encodes prompts (points, boxes, masks) to embeddings.

    Supports:
    - Point prompts (click coordinates)
    - Box prompts (bounding boxes)
    - Mask prompts (coarse masks)
    """

    def __init__(self, config: SAMConfig):
        super().__init__()
        self.config = config

        # Embeddings for point prompts
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, config.prompt_embed_dim)  # Foreground point
            for _ in range(2)  # Foreground and background
        ])

        # Embedding for boxes (represented as point pairs)
        self.box_embed = nn.Embedding(4, config.prompt_embed_dim)

        # Mask downsampling for mask prompts
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, config.mask_in_chans // 4, kernel_size=2, stride=2),
            nn.LayerNorm(config.mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(config.mask_in_chans // 4, config.mask_in_chans, kernel_size=2, stride=2),
            nn.LayerNorm(config.mask_in_chans),
            nn.GELU(),
            nn.Conv2d(config.mask_in_chans, config.prompt_embed_dim, kernel_size=1)
        )

        # No-mask embedding
        self.no_mask_embed = nn.Embedding(1, config.prompt_embed_dim)

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts.

        Args:
            points: (coords, labels) where coords is (batch, num_points, 2)
                   and labels is (batch, num_points) with 0=bg, 1=fg
            boxes: Bounding boxes (batch, 4) as [x1, y1, x2, y2]
            masks: Coarse masks (batch, 1, H, W)

        Returns:
            sparse_embeddings: Sparse prompt embeddings (batch, num_sparse, embed_dim)
            dense_embeddings: Dense prompt embeddings (batch, embed_dim, H, W)
        """
        batch_size = 1
        sparse_embeddings = []

        # Encode points
        if points is not None:
            coords, labels = points
            batch_size = coords.shape[0]

            for i in range(coords.shape[1]):
                coord = coords[:, i, :]
                label = labels[:, i]

                # Get embedding based on label
                point_embed = torch.zeros(
                    batch_size, self.config.prompt_embed_dim,
                    device=coords.device
                )

                for b in range(batch_size):
                    if label[b] == 1:  # Foreground
                        point_embed[b] = self.point_embeddings[0].weight[0]
                    else:  # Background
                        point_embed[b] = self.point_embeddings[1].weight[0]

                # Add coordinate encoding
                coord_embed = self._encode_coords(coord)
                point_embed = point_embed + coord_embed

                sparse_embeddings.append(point_embed)

        # Encode boxes
        if boxes is not None:
            batch_size = boxes.shape[0]
            # Convert box to 4 corner points
            corners = boxes  # (batch, 4)

            for i in range(4):
                corner_embed = self.box_embed.weight[i].unsqueeze(0).expand(batch_size, -1)
                sparse_embeddings.append(corner_embed)

        # Stack sparse embeddings
        if sparse_embeddings:
            sparse_embeddings = torch.stack(sparse_embeddings, dim=1)
        else:
            sparse_embeddings = torch.zeros(
                batch_size, 0, self.config.prompt_embed_dim,
                device=next(self.parameters()).device
            )

        # Encode dense mask
        if masks is not None:
            dense_embeddings = self.mask_downscaling(masks)
        else:
            # No mask - use learned embedding
            dense_embeddings = self.no_mask_embed.weight.reshape(
                1, -1, 1, 1
            ).expand(
                batch_size, -1,
                self.config.image_size // 64,
                self.config.image_size // 64
            )

        return sparse_embeddings, dense_embeddings

    def _encode_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates with positional encoding"""
        # Simple learned encoding for now
        # In practice, use sinusoidal or learnable position encoding
        return torch.zeros(
            coords.shape[0], self.config.prompt_embed_dim,
            device=coords.device
        )


class MaskDecoder(nn.Module):
    """
    Decoder that predicts masks from image and prompt embeddings.

    Predicts multiple masks + IoU scores.
    """

    def __init__(self, config: SAMConfig):
        super().__init__()
        self.config = config

        # Transformer for fusing image and prompt features
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=config.prompt_embed_dim,
            num_heads=8,
            mlp_dim=2048
        )

        # Output upscaling
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(config.prompt_embed_dim, config.prompt_embed_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm(config.prompt_embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(config.prompt_embed_dim // 4, config.prompt_embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU()
        )

        # Mask prediction heads
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(config.prompt_embed_dim, config.prompt_embed_dim, config.prompt_embed_dim // 8, 3)
            for _ in range(config.num_multimask_outputs + 1)  # +1 for single mask
        ])

        # IoU prediction head
        self.iou_prediction_head = MLP(
            config.prompt_embed_dim,
            config.iou_head_hidden_dim,
            config.num_multimask_outputs + 1,
            config.iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks.

        Args:
            image_embeddings: Image features (batch, embed_dim, H, W)
            sparse_prompt_embeddings: Sparse prompts (batch, num_sparse, embed_dim)
            dense_prompt_embeddings: Dense prompts (batch, embed_dim, H, W)
            multimask_output: Whether to output multiple masks

        Returns:
            masks: Predicted masks (batch, num_masks, H*4, W*4)
            iou_predictions: IoU scores (batch, num_masks)
        """
        # Combine dense embeddings
        dense_embeddings = image_embeddings + dense_prompt_embeddings

        # Transformer
        output_tokens, _ = self.transformer(
            sparse_prompt_embeddings,
            dense_embeddings
        )

        # Upscale features
        upscaled_embedding = self.output_upscaling(dense_embeddings)

        # Predict masks
        hyper_in_list = []
        for i, mlp in enumerate(self.output_hypernetworks_mlps):
            hyper_in_list.append(mlp(output_tokens[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)  # (batch, num_masks, dim)

        # Generate masks
        batch_size, num_masks, c = hyper_in.shape
        masks = []

        for i in range(num_masks):
            mask_tokens = hyper_in[:, i, :].unsqueeze(-1).unsqueeze(-1)  # (batch, dim, 1, 1)
            mask = (mask_tokens * upscaled_embedding).sum(dim=1, keepdim=True)
            masks.append(mask)

        masks = torch.cat(masks, dim=1)  # (batch, num_masks, H, W)

        # Predict IoU
        iou_token = output_tokens[:, 0, :]  # Use first token
        iou_predictions = self.iou_prediction_head(iou_token)

        # Select masks
        if multimask_output:
            # Return all masks except the first (single mask)
            masks = masks[:, 1:, :, :]
            iou_predictions = iou_predictions[:, 1:]
        else:
            # Return only the first mask
            masks = masks[:, 0:1, :, :]
            iou_predictions = iou_predictions[:, 0:1]

        return masks, iou_predictions


class TwoWayTransformer(nn.Module):
    """Two-way transformer for fusing image and prompt features"""

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim
            )
            for _ in range(depth)
        ])

        self.final_attn_token_to_image = nn.MultiheadAttention(
            embedding_dim, num_heads, batch_first=True
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        point_embedding: torch.Tensor,
        image_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            point_embedding: (batch, num_points, embed_dim)
            image_embedding: (batch, embed_dim, H, W)

        Returns:
            point_embedding: Updated (batch, num_points, embed_dim)
            image_embedding: Updated (batch, embed_dim, H, W)
        """
        # Flatten image
        batch_size, c, h, w = image_embedding.shape
        image_embedding_flat = image_embedding.flatten(2).transpose(1, 2)  # (batch, h*w, c)

        # Apply layers
        for layer in self.layers:
            point_embedding, image_embedding_flat = layer(
                point_embedding, image_embedding_flat
            )

        # Final attention from points to image
        q = point_embedding + self.final_attn_token_to_image(
            self.norm_final_attn(point_embedding),
            self.norm_final_attn(image_embedding_flat),
            image_embedding_flat,
            need_weights=False
        )[0]

        # Reshape image back
        image_embedding = image_embedding_flat.transpose(1, 2).reshape(batch_size, c, h, w)

        return q, image_embedding


class TwoWayAttentionBlock(nn.Module):
    """Attention block with queries and keys from different sources"""

    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, 2)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.cross_attn_image_to_token = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention on queries
        q = queries + self.self_attn(self.norm1(queries), self.norm1(queries), self.norm1(queries), need_weights=False)[0]

        # Cross attention from queries to keys
        q = q + self.cross_attn_token_to_image(self.norm2(q), self.norm2(keys), keys, need_weights=False)[0]

        # MLP
        q = q + self.mlp(self.norm3(q))

        # Cross attention from keys to queries
        k = keys + self.cross_attn_image_to_token(self.norm4(keys), self.norm4(q), q, need_weights=False)[0]

        return q, k


class MLP(nn.Module):
    """Simple MLP"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Identity()
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SAM(nn.Module):
    """
    Complete Segment Anything Model.

    Zero-shot segmentation with various prompt types.
    """

    def __init__(self, config: SAMConfig):
        super().__init__()
        self.config = config

        self.image_encoder = ImageEncoder(config)
        self.prompt_encoder = PromptEncoder(config)
        self.mask_decoder = MaskDecoder(config)

    def forward(
        self,
        images: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        multimask_output: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Segment image with prompts.

        Args:
            images: Input images (batch, 3, 1024, 1024)
            points: Point prompts (coords, labels)
            boxes: Box prompts (batch, 4)
            masks: Mask prompts (batch, 1, H, W)
            multimask_output: Whether to output multiple masks

        Returns:
            Dictionary with masks and IoU predictions
        """
        # Encode image
        image_embeddings = self.image_encoder(images)

        # Encode prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks
        )

        # Decode masks
        masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )

        return {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_masks': masks  # Would include low-res version
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("SAM (Segment Anything Model)")
    print("="*80)

    # Create SAM model (base configuration)
    config = SAMConfig(
        image_size=1024,
        encoder_embed_dim=768,
        encoder_depth=12
    )

    model = SAM(config)

    # Example: Segment with point prompt
    batch_size = 2
    images = torch.randn(batch_size, 3, 1024, 1024)

    # Point prompts: click on foreground
    point_coords = torch.tensor([[[512, 512]], [[256, 256]]])  # (batch, num_points, 2)
    point_labels = torch.tensor([[1], [1]])  # (batch, num_points) - 1 for foreground

    print(f"\nInput image shape: {images.shape}")
    print(f"Point prompts: {point_coords.shape}")

    # Forward pass
    outputs = model(
        images=images,
        points=(point_coords, point_labels),
        multimask_output=True
    )

    print(f"\nOutputs:")
    print(f"  Masks shape: {outputs['masks'].shape}")
    print(f"  IoU predictions shape: {outputs['iou_predictions'].shape}")

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "="*80)
