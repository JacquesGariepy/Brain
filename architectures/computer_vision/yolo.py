"""
YOLOv8 - You Only Look Once v8 (Ultralytics)

SOTA real-time object detection (2023+).

Key features:
- Anchor-free detection
- CSPDarknet backbone with C2f blocks
- PAN (Path Aggregation Network) neck
- Decoupled head (separate cls/box branches)
- Task-aligned assigner for label assignment
- Distribution Focal Loss for bounding boxes
- Multiple scales (n, s, m, l, x)

Architecture:
- Backbone: CSPDarknet with C2f modules
- Neck: PAN with feature pyramid
- Head: Decoupled detection head
- Loss: DFL + CIOU + BCE

References:
- "YOLOv8: Ultralytics YOLO" (Glenn Jocher et al., 2023)
- https://github.com/ultralytics/ultralytics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math


@dataclass
class YOLOv8Config:
    """Configuration for YOLOv8"""
    # Model size: 'n', 's', 'm', 'l', 'x'
    model_size: str = 'n'

    # Input
    image_size: int = 640
    num_classes: int = 80  # COCO

    # Architecture scaling factors per model size
    # (depth_multiple, width_multiple, max_channels)
    size_configs: Dict[str, Tuple[float, float, int]] = None

    # Detection
    num_levels: int = 3  # P3, P4, P5
    stride: List[int] = None  # [8, 16, 32]

    # Training
    iou_threshold: float = 0.7
    conf_threshold: float = 0.25
    max_det: int = 300

    # Loss weights
    box_weight: float = 7.5
    cls_weight: float = 0.5
    dfl_weight: float = 1.5

    # DFL (Distribution Focal Loss)
    reg_max: int = 16  # Maximum value for distribution

    def __post_init__(self):
        if self.stride is None:
            self.stride = [8, 16, 32]

        if self.size_configs is None:
            # (depth_multiple, width_multiple, max_channels)
            self.size_configs = {
                'n': (0.33, 0.25, 1024),  # Nano
                's': (0.33, 0.50, 1024),  # Small
                'm': (0.67, 0.75, 768),   # Medium
                'l': (1.00, 1.00, 512),   # Large
                'x': (1.00, 1.25, 512),   # Extra Large
            }


class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation: bool = True
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 3, 1)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions.

    Faster implementation than C3 in YOLOv5.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int = 1,
        shortcut: bool = False,
        expansion: float = 0.5
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, 2 * hidden_channels, 1, 1)
        self.conv2 = Conv(
            (2 + num_bottlenecks) * hidden_channels,
            out_channels,
            1,
            1
        )

        self.bottlenecks = nn.ModuleList(
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0)
            for _ in range(num_bottlenecks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into two paths
        y = list(self.conv1(x).chunk(2, 1))

        # Apply bottlenecks
        y.extend(m(y[-1]) for m in self.bottlenecks)

        # Concatenate and final conv
        return self.conv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF).

    Equivalent to SPP but faster using sequential pooling.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2

        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], 1))


class YOLOv8Backbone(nn.Module):
    """
    YOLOv8 Backbone - CSPDarknet with C2f blocks.

    Outputs features at 3 scales: P3, P4, P5
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        depth_multiple: float = 0.33,
        width_multiple: float = 0.25
    ):
        super().__init__()

        def make_divisible(x: float, divisor: int = 8) -> int:
            """Make channels divisible by divisor"""
            return math.ceil(x / divisor) * divisor

        def scale_channels(channels: int) -> int:
            return make_divisible(channels * width_multiple)

        def scale_depth(depth: int) -> int:
            return max(round(depth * depth_multiple), 1)

        # Stem
        self.stem = Conv(in_channels, scale_channels(base_channels), 3, 2)

        # Stage 1: 640 -> 320
        self.stage1 = nn.Sequential(
            Conv(scale_channels(64), scale_channels(128), 3, 2),
            C2f(scale_channels(128), scale_channels(128), scale_depth(3))
        )

        # Stage 2: 320 -> 160 (P3)
        self.stage2 = nn.Sequential(
            Conv(scale_channels(128), scale_channels(256), 3, 2),
            C2f(scale_channels(256), scale_channels(256), scale_depth(6))
        )

        # Stage 3: 160 -> 80 (P4)
        self.stage3 = nn.Sequential(
            Conv(scale_channels(256), scale_channels(512), 3, 2),
            C2f(scale_channels(512), scale_channels(512), scale_depth(6))
        )

        # Stage 4: 80 -> 40 (P5)
        self.stage4 = nn.Sequential(
            Conv(scale_channels(512), scale_channels(1024), 3, 2),
            C2f(scale_channels(1024), scale_channels(1024), scale_depth(3)),
            SPPF(scale_channels(1024), scale_channels(1024))
        )

        # Output channels for each scale
        self.out_channels = [
            scale_channels(256),  # P3
            scale_channels(512),  # P4
            scale_channels(1024)  # P5
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.

        Returns:
            [P3, P4, P5] features
        """
        x = self.stem(x)
        x = self.stage1(x)

        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)

        return [p3, p4, p5]


class YOLOv8Neck(nn.Module):
    """
    YOLOv8 Neck - PAN (Path Aggregation Network).

    Fuses features from different scales.
    """

    def __init__(
        self,
        in_channels: List[int],
        depth_multiple: float = 0.33,
        width_multiple: float = 0.25
    ):
        super().__init__()

        def make_divisible(x: float, divisor: int = 8) -> int:
            return math.ceil(x / divisor) * divisor

        def scale_channels(channels: int) -> int:
            return make_divisible(channels * width_multiple)

        def scale_depth(depth: int) -> int:
            return max(round(depth * depth_multiple), 1)

        # Top-down pathway
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # P5 -> P4
        self.c2f_p4 = C2f(
            in_channels[2] + in_channels[1],
            scale_channels(512),
            scale_depth(3),
            shortcut=False
        )

        # P4 -> P3
        self.c2f_p3 = C2f(
            scale_channels(512) + in_channels[0],
            scale_channels(256),
            scale_depth(3),
            shortcut=False
        )

        # Bottom-up pathway
        self.conv_p3 = Conv(scale_channels(256), scale_channels(256), 3, 2)
        self.c2f_p4_out = C2f(
            scale_channels(256) + scale_channels(512),
            scale_channels(512),
            scale_depth(3),
            shortcut=False
        )

        self.conv_p4 = Conv(scale_channels(512), scale_channels(512), 3, 2)
        self.c2f_p5_out = C2f(
            scale_channels(512) + in_channels[2],
            scale_channels(1024),
            scale_depth(3),
            shortcut=False
        )

        # Output channels
        self.out_channels = [
            scale_channels(256),  # P3
            scale_channels(512),  # P4
            scale_channels(1024)  # P5
        ]

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [P3, P4, P5] from backbone

        Returns:
            [P3_out, P4_out, P5_out] fused features
        """
        p3, p4, p5 = features

        # Top-down
        p5_up = self.upsample(p5)
        p4_fused = self.c2f_p4(torch.cat([p5_up, p4], 1))

        p4_up = self.upsample(p4_fused)
        p3_out = self.c2f_p3(torch.cat([p4_up, p3], 1))

        # Bottom-up
        p3_down = self.conv_p3(p3_out)
        p4_out = self.c2f_p4_out(torch.cat([p3_down, p4_fused], 1))

        p4_down = self.conv_p4(p4_out)
        p5_out = self.c2f_p5_out(torch.cat([p4_down, p5], 1))

        return [p3_out, p4_out, p5_out]


class DFL(nn.Module):
    """
    Distribution Focal Loss module.

    Represents bounding box as a distribution over discrete bins.
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer(
            'project',
            torch.linspace(0, reg_max, reg_max + 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 4*(reg_max+1), H, W)

        Returns:
            Box deltas (batch, 4, H, W)
        """
        batch, _, h, w = x.shape

        # Reshape to (batch, 4, reg_max+1, H, W)
        x = x.view(batch, 4, self.reg_max + 1, h, w)

        # Apply softmax over distribution
        x = F.softmax(x, dim=2)

        # Compute expected value
        x = (x * self.project.view(1, 1, -1, 1, 1)).sum(dim=2)

        return x


class YOLOHead(nn.Module):
    """
    YOLOv8 Detection Head.

    Decoupled head with separate branches for:
    - Classification
    - Bounding box regression (using DFL)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: List[int],
        reg_max: int = 16
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_levels = len(in_channels)

        # Shared convs for each level
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        for in_c in in_channels:
            # Stem
            self.stems.append(Conv(in_c, in_c, 1, 1))

            # Classification branch
            self.cls_convs.append(
                nn.Sequential(
                    Conv(in_c, in_c, 3, 1),
                    Conv(in_c, in_c, 3, 1)
                )
            )
            self.cls_preds.append(
                nn.Conv2d(in_c, num_classes, 1)
            )

            # Regression branch
            self.reg_convs.append(
                nn.Sequential(
                    Conv(in_c, in_c, 3, 1),
                    Conv(in_c, in_c, 3, 1)
                )
            )
            self.reg_preds.append(
                nn.Conv2d(in_c, 4 * (reg_max + 1), 1)
            )

        # DFL module
        self.dfl = DFL(reg_max)

        # Initialize biases
        self._initialize_biases()

    def _initialize_biases(self):
        """Initialize biases for better initial predictions"""
        for cls_pred in self.cls_preds:
            # Focal loss initialization
            b = cls_pred.bias.view(-1, )
            b.data.fill_(-math.log((1 - 0.01) / 0.01))
            cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            features: [P3, P4, P5] features from neck

        Returns:
            cls_scores: List of (batch, num_classes, H, W)
            bbox_preds: List of (batch, 4, H, W)
        """
        cls_scores = []
        bbox_preds = []

        for i, x in enumerate(features):
            x = self.stems[i](x)

            # Classification
            cls_feat = self.cls_convs[i](x)
            cls_score = self.cls_preds[i](cls_feat)
            cls_scores.append(cls_score)

            # Regression
            reg_feat = self.reg_convs[i](x)
            reg_dist = self.reg_preds[i](reg_feat)
            bbox_pred = self.dfl(reg_dist)
            bbox_preds.append(bbox_pred)

        return cls_scores, bbox_preds


class YOLOv8(nn.Module):
    """
    Complete YOLOv8 model.

    Real-time object detection with:
    - Anchor-free detection
    - CSPDarknet + PAN architecture
    - Decoupled head
    - Distribution Focal Loss
    """

    def __init__(self, config: YOLOv8Config):
        super().__init__()
        self.config = config

        # Get scaling factors
        depth_mult, width_mult, _ = config.size_configs[config.model_size]

        # Backbone
        self.backbone = YOLOv8Backbone(
            in_channels=3,
            base_channels=64,
            depth_multiple=depth_mult,
            width_multiple=width_mult
        )

        # Neck
        self.neck = YOLOv8Neck(
            in_channels=self.backbone.out_channels,
            depth_multiple=depth_mult,
            width_multiple=width_mult
        )

        # Head
        self.head = YOLOHead(
            num_classes=config.num_classes,
            in_channels=self.neck.out_channels,
            reg_max=config.reg_max
        )

        # Create anchors
        self.register_buffer(
            'stride',
            torch.tensor(config.stride, dtype=torch.float32)
        )

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Images (batch, 3, H, W)
            targets: Optional targets for training

        Returns:
            Dictionary with:
            - cls_scores: Classification scores
            - bbox_preds: Bounding box predictions
            - loss: If targets provided
        """
        # Backbone
        features = self.backbone(x)

        # Neck
        features = self.neck(features)

        # Head
        cls_scores, bbox_preds = self.head(features)

        outputs = {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds
        }

        # Compute loss if training
        if targets is not None and self.training:
            loss = self._compute_loss(cls_scores, bbox_preds, targets)
            outputs['loss'] = loss

        return outputs

    def _compute_loss(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute YOLOv8 loss.

        Uses:
        - BCE loss for classification
        - CIOU loss for boxes
        - DFL loss for distribution
        """
        # Placeholder for loss computation
        # In practice, this involves:
        # 1. Task-aligned assigner to match predictions to targets
        # 2. BCE loss for classification
        # 3. CIOU loss for bounding boxes
        # 4. DFL loss for box distribution

        total_loss = torch.tensor(0.0, device=cls_scores[0].device)

        # This would be implemented with proper label assignment
        # and loss computation for production use

        return total_loss

    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Inference with NMS post-processing.

        Args:
            x: Images (batch, 3, H, W)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            List of detections per image with:
            - boxes: (N, 4) in xyxy format
            - scores: (N,)
            - labels: (N,)
        """
        conf_threshold = conf_threshold or self.config.conf_threshold
        iou_threshold = iou_threshold or self.config.iou_threshold

        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            cls_scores = outputs['cls_scores']
            bbox_preds = outputs['bbox_preds']

            # Convert predictions to boxes
            predictions = self._decode_predictions(cls_scores, bbox_preds)

            # Apply NMS per image
            detections = []
            for pred in predictions:
                det = self._apply_nms(
                    pred,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                detections.append(det)

            return detections

    def _decode_predictions(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Decode predictions to boxes.

        Returns:
            List of (batch, num_boxes, 4+num_classes)
        """
        # Placeholder - would implement anchor-free decoding
        # Converting relative predictions to absolute boxes
        return []

    def _apply_nms(
        self,
        predictions: torch.Tensor,
        conf_threshold: float,
        iou_threshold: float
    ) -> Dict[str, torch.Tensor]:
        """
        Apply Non-Maximum Suppression.

        Returns:
            Dictionary with boxes, scores, labels
        """
        # Placeholder - would implement NMS
        return {
            'boxes': torch.empty(0, 4),
            'scores': torch.empty(0),
            'labels': torch.empty(0, dtype=torch.long)
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("YOLOv8 - Real-Time Object Detection")
    print("="*80)

    # Create YOLOv8 models of different sizes
    for size in ['n', 's', 'm']:
        config = YOLOv8Config(
            model_size=size,
            num_classes=80,
            image_size=640
        )

        model = YOLOv8(config)

        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 3, 640, 640)

        print(f"\nYOLOv8-{size.upper()}:")
        print(f"  Input shape: {x.shape}")

        outputs = model(x)
        print(f"  Classification outputs: {len(outputs['cls_scores'])} levels")
        for i, cls_score in enumerate(outputs['cls_scores']):
            print(f"    Level {i}: {cls_score.shape}")

        print(f"  Box regression outputs: {len(outputs['bbox_preds'])} levels")
        for i, bbox_pred in enumerate(outputs['bbox_preds']):
            print(f"    Level {i}: {bbox_pred.shape}")

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

    print("\n" + "="*80)
