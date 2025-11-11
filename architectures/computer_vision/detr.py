"""
DETR - DEtection TRansformer

SOTA end-to-end object detection with Transformers (2020+).

Key innovations:
- End-to-end detection (no NMS, no anchors)
- Transformer encoder-decoder architecture
- Object queries (learned embeddings)
- Bipartite matching with Hungarian algorithm
- Set prediction loss
- Direct set prediction

Architecture:
- Backbone: CNN (ResNet) for feature extraction
- Encoder: Transformer encoder on flattened features
- Decoder: Transformer decoder with object queries
- FFN: Prediction heads for class and bbox

References:
- "End-to-End Object Detection with Transformers" (Carion et al., 2020)
- Facebook AI Research (FAIR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class DETRConfig:
    """Configuration for DETR"""
    # Architecture
    backbone: str = 'resnet50'
    hidden_dim: int = 256
    num_queries: int = 100  # Number of object queries

    # Transformer
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # Detection
    num_classes: int = 91  # COCO (80 + 1 for background)

    # Loss weights
    loss_ce: float = 1.0  # Classification loss weight
    loss_bbox: float = 5.0  # L1 bbox loss weight
    loss_giou: float = 2.0  # GIoU loss weight

    # Matcher
    cost_class: float = 1.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0

    # Aux losses
    aux_loss: bool = True


class PositionEmbeddingSine(nn.Module):
    """
    2D positional encoding using sine/cosine functions.

    Similar to the one used in Transformer "Attention is All You Need".
    """

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Features (batch, channels, H, W)
            mask: Optional mask (batch, H, W)

        Returns:
            Positional encoding (batch, channels, H, W)
        """
        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool, device=x.device)

        # Cumulative sum to get positions
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # Normalize to [0, 1]
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps)
        x_embed = x_embed / (x_embed[:, :, -1:] + eps)

        # Create frequency bands
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Apply sine/cosine
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack([
            pos_x[:, :, :, 0::2].sin(),
            pos_x[:, :, :, 1::2].cos()
        ], dim=4).flatten(3)

        pos_y = torch.stack([
            pos_y[:, :, :, 0::2].sin(),
            pos_y[:, :, :, 1::2].cos()
        ], dim=4).flatten(3)

        # Concatenate x and y
        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)

        return pos


class DETRTransformer(nn.Module):
    """
    Transformer for DETR.

    Standard Transformer encoder-decoder with:
    - Multi-head self-attention
    - Cross-attention in decoder
    - FFN after each attention layer
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            src: Source features (HW, batch, d_model)
            query_embed: Object query embeddings (num_queries, batch, d_model)
            pos_embed: Positional embeddings (HW, batch, d_model)

        Returns:
            Decoder output (num_queries, batch, d_model)
        """
        # Encoder: process image features with self-attention
        memory = self.encoder(
            src + pos_embed
        )

        # Decoder: process object queries with cross-attention to image
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(
            tgt,
            memory,
            query_pos=query_embed,
            pos=pos_embed
        )

        return hs


class DETRBackbone(nn.Module):
    """
    CNN backbone for DETR.

    Typically ResNet-50 with modifications:
    - Remove stride from last block
    - Add dilation to maintain resolution
    """

    def __init__(
        self,
        name: str = 'resnet50',
        pretrained: bool = True,
        return_interm_layers: bool = False
    ):
        super().__init__()

        # Import torchvision for ResNet
        try:
            from torchvision import models

            if name == 'resnet50':
                backbone = models.resnet50(pretrained=pretrained)
            elif name == 'resnet101':
                backbone = models.resnet101(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported backbone: {name}")

            # Remove avgpool and fc
            self.body = nn.Sequential(*list(backbone.children())[:-2])

            # Output channels
            if name in ['resnet50', 'resnet101']:
                self.num_channels = 2048

        except ImportError:
            # Fallback: simple CNN backbone
            self.body = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),

                self._make_layer(64, 256, 3),
                self._make_layer(256, 512, 4, stride=2),
                self._make_layer(512, 1024, 6, stride=2),
                self._make_layer(1024, 2048, 3, stride=2),
            )
            self.num_channels = 2048

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Module:
        """Helper to create ResNet layer"""
        layers = []

        # Downsample if needed
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

        # First block with stride
        layers.append(self._make_block(in_channels, out_channels, stride, downsample))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _make_block(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> nn.Module:
        """Create a basic ResNet block"""
        class BasicBlock(nn.Module):
            def __init__(self):
                super().__init__()
                mid_channels = out_channels // 4

                self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1)
                self.bn1 = nn.BatchNorm2d(mid_channels)
                self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1)
                self.bn2 = nn.BatchNorm2d(mid_channels)
                self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1)
                self.bn3 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample

            def forward(self, x):
                identity = x

                out = self.relu(self.bn1(self.conv1(x)))
                out = self.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)

                return out

        return BasicBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images (batch, 3, H, W)

        Returns:
            Features (batch, num_channels, H', W')
        """
        return self.body(x)


class MLP(nn.Module):
    """Multi-Layer Perceptron for prediction heads"""

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

            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DETR(nn.Module):
    """
    Complete DETR model.

    End-to-end object detection with Transformers:
    - No anchors
    - No NMS
    - Direct set prediction
    - Bipartite matching loss
    """

    def __init__(self, config: DETRConfig):
        super().__init__()
        self.config = config

        # Backbone
        self.backbone = DETRBackbone(
            name=config.backbone,
            pretrained=False
        )

        # Input projection
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels,
            config.hidden_dim,
            kernel_size=1
        )

        # Positional encoding
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=config.hidden_dim // 2
        )

        # Object queries
        self.query_embed = nn.Embedding(config.num_queries, config.hidden_dim)

        # Transformer
        self.transformer = DETRTransformer(
            d_model=config.hidden_dim,
            num_heads=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )

        # Prediction heads
        self.class_embed = nn.Linear(config.hidden_dim, config.num_classes + 1)
        self.bbox_embed = MLP(
            config.hidden_dim,
            config.hidden_dim,
            4,
            num_layers=3
        )

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Input images (batch, 3, H, W)
            targets: Optional targets for training

        Returns:
            Dictionary with:
            - pred_logits: Class predictions (batch, num_queries, num_classes+1)
            - pred_boxes: Box predictions (batch, num_queries, 4) in [cx, cy, w, h] format
            - loss: If targets provided
        """
        # Extract features
        features = self.backbone(images)

        # Project to hidden dimension
        src = self.input_proj(features)

        # Get positional embeddings
        pos = self.position_embedding(src)

        # Flatten spatial dimensions
        batch_size, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # (HW, batch, c)
        pos = pos.flatten(2).permute(2, 0, 1)  # (HW, batch, c)

        # Get query embeddings
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # Transformer
        hs = self.transformer(src, query_embed, pos)

        # Predictions
        # hs shape: (num_queries, batch, hidden_dim)
        hs = hs.permute(1, 0, 2)  # (batch, num_queries, hidden_dim)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }

        # Compute loss if training
        if targets is not None and self.training:
            loss_dict = self._compute_loss(outputs_class, outputs_coord, targets)
            out['loss'] = loss_dict

        return out

    def _compute_loss(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DETR loss with bipartite matching.

        Steps:
        1. Hungarian matching between predictions and targets
        2. Compute classification loss (CE)
        3. Compute box regression loss (L1 + GIoU)
        """
        # Placeholder for loss computation
        # In practice, this involves:
        # 1. HungarianMatcher to find optimal assignment
        # 2. Cross-entropy loss for classification
        # 3. L1 loss + GIoU loss for boxes

        batch_size = pred_logits.shape[0]

        # Dummy loss for structure
        loss_dict = {
            'loss_ce': torch.tensor(0.0, device=pred_logits.device),
            'loss_bbox': torch.tensor(0.0, device=pred_logits.device),
            'loss_giou': torch.tensor(0.0, device=pred_logits.device)
        }

        total_loss = (
            self.config.loss_ce * loss_dict['loss_ce'] +
            self.config.loss_bbox * loss_dict['loss_bbox'] +
            self.config.loss_giou * loss_dict['loss_giou']
        )

        loss_dict['total_loss'] = total_loss

        return loss_dict

    def predict(
        self,
        images: torch.Tensor,
        threshold: float = 0.7
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference and filter predictions.

        Args:
            images: Input images (batch, 3, H, W)
            threshold: Confidence threshold

        Returns:
            List of predictions per image with:
            - boxes: (N, 4) in [cx, cy, w, h] format
            - scores: (N,)
            - labels: (N,)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images)

            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']

            # Get probabilities (softmax over classes)
            prob = F.softmax(pred_logits, dim=-1)

            # Get scores and labels (exclude "no-object" class)
            scores, labels = prob[..., :-1].max(-1)

            # Filter by threshold
            predictions = []
            for i in range(images.shape[0]):
                keep = scores[i] > threshold

                predictions.append({
                    'boxes': pred_boxes[i][keep],
                    'scores': scores[i][keep],
                    'labels': labels[i][keep]
                })

            return predictions


# Hungarian Matcher (for loss computation)
class HungarianMatcher(nn.Module):
    """
    Computes optimal assignment between predictions and ground truth.

    Uses Hungarian algorithm (scipy.optimize.linear_sum_assignment).
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute optimal assignment.

        Args:
            pred_logits: (batch, num_queries, num_classes)
            pred_boxes: (batch, num_queries, 4)
            targets: List of target dicts with 'labels' and 'boxes'

        Returns:
            List of (pred_indices, target_indices) per batch
        """
        batch_size, num_queries = pred_logits.shape[:2]

        # Flatten predictions
        out_prob = pred_logits.flatten(0, 1).softmax(-1)
        out_bbox = pred_boxes.flatten(0, 1)

        # Concatenate target labels and boxes
        tgt_ids = torch.cat([t['labels'] for t in targets])
        tgt_bbox = torch.cat([t['boxes'] for t in targets])

        # Compute costs
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -self._generalized_box_iou(
            self._box_cxcywh_to_xyxy(out_bbox),
            self._box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox +
            self.cost_class * cost_class +
            self.cost_giou * cost_giou
        )
        C = C.view(batch_size, num_queries, -1)

        # Split by batch and find optimal assignment
        sizes = [len(t['boxes']) for t in targets]
        indices = []

        # Would use scipy.optimize.linear_sum_assignment here
        # For now, return dummy indices
        for i, c in enumerate(C.split(sizes, -1)):
            indices.append((
                torch.arange(num_queries),
                torch.arange(sizes[i])
            ))

        return indices

    def _box_cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]"""
        cx, cy, w, h = boxes.unbind(-1)
        b = [
            cx - 0.5 * w,
            cy - 0.5 * h,
            cx + 0.5 * w,
            cy + 0.5 * h
        ]
        return torch.stack(b, dim=-1)

    def _generalized_box_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """Compute generalized IoU"""
        # Placeholder - would implement GIoU
        return torch.zeros(boxes1.shape[0], boxes2.shape[0])


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("DETR - End-to-End Object Detection with Transformers")
    print("="*80)

    # Create DETR model
    config = DETRConfig(
        backbone='resnet50',
        hidden_dim=256,
        num_queries=100,
        num_classes=80,
        num_encoder_layers=6,
        num_decoder_layers=6
    )

    model = DETR(config)

    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 640, 640)

    print(f"\nInput shape: {images.shape}")

    # Forward pass
    outputs = model(images)

    print(f"\nPredictions:")
    print(f"  Class logits: {outputs['pred_logits'].shape}")
    print(f"  Boxes: {outputs['pred_boxes'].shape}")

    # Inference
    predictions = model.predict(images, threshold=0.7)
    print(f"\nDetections (threshold=0.7):")
    for i, pred in enumerate(predictions):
        print(f"  Image {i}: {len(pred['boxes'])} objects")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {num_params:,}")

    print("\n" + "="*80)
