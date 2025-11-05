"""
PatchTST - Patch Time Series Transformer

SOTA time series forecasting using patches (2023).

Key innovations:
- Patching time series (like ViT for images)
- Channel independence (process each variable separately)
- Transformer encoder with self-attention
- Better than traditional point-wise models
- Efficient for long sequences
- Strong transfer learning capabilities

Architecture:
- Patch embedding: Convert time series to patches
- Positional encoding
- Transformer encoder (from LLMs)
- Flatten and predict

Applications:
- Long-term forecasting
- Multivariate time series
- Transfer learning across domains
- Energy, weather, traffic prediction

References:
- "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (Nie et al., 2023)
- IBM Research, MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class PatchTSTConfig:
    """Configuration for PatchTST"""
    # Data
    num_variables: int = 7  # Number of time series variables
    seq_len: int = 336  # Input sequence length
    pred_len: int = 96  # Prediction horizon

    # Patching
    patch_len: int = 16  # Length of each patch
    stride: int = 8  # Stride for patches

    # Architecture
    d_model: int = 128  # Model dimension
    n_heads: int = 8  # Number of attention heads
    e_layers: int = 3  # Number of encoder layers
    d_ff: int = 256  # Feedforward dimension
    dropout: float = 0.2
    activation: str = 'gelu'

    # Channel independence
    channel_independence: bool = True

    # Positional encoding
    use_positional_encoding: bool = True

    # Normalization
    norm_type: str = 'batch'  # 'batch', 'layer', or 'none'

    # Head
    head_type: str = 'flatten'  # 'flatten' or 'regression'

    # Mask ratio for pre-training (0.0 = no masking)
    mask_ratio: float = 0.0

    def __post_init__(self):
        # Calculate number of patches
        self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1


class Patching(nn.Module):
    """
    Convert time series to patches.

    Similar to image patching in ViT, but for 1D time series.
    """

    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Time series (batch, num_vars, seq_len)

        Returns:
            Patches (batch, num_vars, num_patches, patch_len)
        """
        batch_size, num_vars, seq_len = x.shape

        # Unfold to create patches
        # x: (batch, num_vars, seq_len)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # patches: (batch, num_vars, num_patches, patch_len)

        return patches


class PatchEmbedding(nn.Module):
    """
    Embed patches into model dimension.

    Linear projection from patch_len to d_model.
    """

    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model

        # Linear projection
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (batch, num_vars, num_patches, patch_len)

        Returns:
            Embeddings (batch, num_vars, num_patches, d_model)
        """
        return self.proj(patches)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for patches.

    Standard sinusoidal encoding from Transformer.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_vars, num_patches, d_model)

        Returns:
            x with positional encoding added
        """
        num_patches = x.shape[2]
        return x + self.pe[:, :, :num_patches, :]


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer.

    Self-attention + FFN with residual connections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=False  # Will use (seq, batch, d_model) format
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: (seq, batch, d_model)
            src_mask: Optional attention mask

        Returns:
            Output (seq, batch, d_model)
        """
        # Self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class FlattenHead(nn.Module):
    """
    Flatten head for prediction.

    Flattens all patch embeddings and predicts.
    """

    def __init__(
        self,
        n_vars: int,
        num_patches: int,
        d_model: int,
        pred_len: int,
        head_dropout: float = 0.0
    ):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(num_patches * d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_vars, num_patches, d_model)

        Returns:
            Predictions (batch, num_vars, pred_len)
        """
        # Flatten patches
        x = self.flatten(x)  # (batch, num_vars, num_patches * d_model)

        # Linear projection to prediction length
        x = self.linear(x)  # (batch, num_vars, pred_len)
        x = self.dropout(x)

        return x


class PatchTST(nn.Module):
    """
    Complete PatchTST model.

    Time series forecasting with:
    - Patch-based representation
    - Transformer encoder
    - Channel independence
    - Efficient for long sequences
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config

        # Patching
        self.patching = Patching(config.patch_len, config.stride)

        # Patch embedding
        self.patch_embedding = PatchEmbedding(config.patch_len, config.d_model)

        # Positional encoding
        if config.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                config.d_model,
                max_len=config.num_patches
            )
        else:
            self.positional_encoding = None

        # Normalization
        if config.norm_type == 'batch':
            self.norm = nn.BatchNorm1d(config.num_variables)
        elif config.norm_type == 'layer':
            self.norm = nn.LayerNorm(config.seq_len)
        else:
            self.norm = None

        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                activation=config.activation
            )
            for _ in range(config.e_layers)
        ])

        # Prediction head
        if config.head_type == 'flatten':
            self.head = FlattenHead(
                n_vars=config.num_variables,
                num_patches=config.num_patches,
                d_model=config.d_model,
                pred_len=config.pred_len,
                head_dropout=config.dropout
            )
        else:
            raise ValueError(f"Unknown head type: {config.head_type}")

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input time series (batch, num_vars, seq_len)
            return_attention: Return attention weights

        Returns:
            Predictions (batch, num_vars, pred_len)
        """
        batch_size = x.shape[0]

        # Normalization
        if self.norm is not None:
            # Store mean and std for denormalization
            means = x.mean(dim=2, keepdim=True)
            stds = x.std(dim=2, keepdim=True) + 1e-5
            x = (x - means) / stds

            if isinstance(self.norm, nn.BatchNorm1d):
                x = self.norm(x)
            else:
                x = self.norm(x)

        # Patching
        patches = self.patching(x)  # (batch, num_vars, num_patches, patch_len)

        # Patch embedding
        x = self.patch_embedding(patches)  # (batch, num_vars, num_patches, d_model)

        # Positional encoding
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        # Process each variable independently (channel independence)
        if self.config.channel_independence:
            # Reshape to process all variables together
            num_vars = x.shape[1]
            num_patches = x.shape[2]

            # Merge batch and variable dimensions
            x = x.reshape(batch_size * num_vars, num_patches, self.config.d_model)

            # Transpose for transformer (seq, batch, d_model)
            x = x.transpose(0, 1)

            # Transformer encoder
            for layer in self.encoder_layers:
                x = layer(x)

            # Transpose back (batch, seq, d_model)
            x = x.transpose(0, 1)

            # Reshape back to separate variables
            x = x.reshape(batch_size, num_vars, num_patches, self.config.d_model)

        else:
            # Process all variables together
            # Flatten to (batch, num_vars * num_patches, d_model)
            num_vars = x.shape[1]
            num_patches = x.shape[2]

            x = x.reshape(batch_size, num_vars * num_patches, self.config.d_model)

            # Transpose for transformer
            x = x.transpose(0, 1)

            # Transformer encoder
            for layer in self.encoder_layers:
                x = layer(x)

            # Transpose back
            x = x.transpose(0, 1)

            # Reshape to separate variables and patches
            x = x.reshape(batch_size, num_vars, num_patches, self.config.d_model)

        # Prediction head
        output = self.head(x)  # (batch, num_vars, pred_len)

        # Denormalize
        if self.norm is not None:
            output = output * stds + means

        return output

    def predict(
        self,
        x: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Make predictions.

        Args:
            x: Input time series (batch, num_vars, seq_len)
            normalize: Whether to normalize input

        Returns:
            Predictions (batch, num_vars, pred_len)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class PatchTSTForPretraining(nn.Module):
    """
    PatchTST for self-supervised pre-training.

    Uses masked patch modeling (like BERT for text).
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config

        # Base model
        self.model = PatchTST(config)

        # Reconstruction head
        self.reconstruction_head = nn.Linear(
            config.d_model,
            config.patch_len
        )

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, config.d_model))

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with masking.

        Args:
            x: Input (batch, num_vars, seq_len)
            mask_ratio: Ratio of patches to mask

        Returns:
            reconstructed: Reconstructed patches
            mask: Binary mask
            original_patches: Original patches for loss computation
        """
        mask_ratio = mask_ratio or self.config.mask_ratio

        # Get patches
        patches = self.model.patching(x)  # (batch, num_vars, num_patches, patch_len)

        # Embed patches
        x_embed = self.model.patch_embedding(patches)

        # Random masking
        batch_size, num_vars, num_patches, d_model = x_embed.shape

        # Create random mask
        mask = torch.rand(batch_size, num_vars, num_patches, 1) > mask_ratio
        mask = mask.to(x_embed.device)

        # Apply mask
        x_masked = x_embed * mask + self.mask_token * (~mask)

        # Add positional encoding
        if self.model.positional_encoding is not None:
            x_masked = self.model.positional_encoding(x_masked)

        # Process through transformer
        if self.config.channel_independence:
            x_proc = x_masked.reshape(batch_size * num_vars, num_patches, d_model)
            x_proc = x_proc.transpose(0, 1)

            for layer in self.model.encoder_layers:
                x_proc = layer(x_proc)

            x_proc = x_proc.transpose(0, 1)
            x_proc = x_proc.reshape(batch_size, num_vars, num_patches, d_model)
        else:
            x_proc = x_masked.reshape(batch_size, num_vars * num_patches, d_model)
            x_proc = x_proc.transpose(0, 1)

            for layer in self.model.encoder_layers:
                x_proc = layer(x_proc)

            x_proc = x_proc.transpose(0, 1)
            x_proc = x_proc.reshape(batch_size, num_vars, num_patches, d_model)

        # Reconstruct patches
        reconstructed = self.reconstruction_head(x_proc)

        return reconstructed, mask.squeeze(-1), patches


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("PatchTST - Patch Time Series Transformer")
    print("="*80)

    # Create PatchTST model
    config = PatchTSTConfig(
        num_variables=7,
        seq_len=336,
        pred_len=96,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=8,
        e_layers=3
    )

    model = PatchTST(config)

    print(f"\nConfiguration:")
    print(f"  Input length: {config.seq_len}")
    print(f"  Prediction length: {config.pred_len}")
    print(f"  Number of variables: {config.num_variables}")
    print(f"  Patch length: {config.patch_len}")
    print(f"  Number of patches: {config.num_patches}")
    print(f"  Model dimension: {config.d_model}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config.num_variables, config.seq_len)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch={batch_size}, vars={config.num_variables}, pred_len={config.pred_len})")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {num_params:,}")

    # Test pre-training model
    print("\n" + "-"*80)
    print("Pre-training with Masked Patch Modeling")
    print("-"*80)

    config_pretrain = PatchTSTConfig(
        num_variables=7,
        seq_len=336,
        pred_len=96,
        mask_ratio=0.4
    )

    pretrain_model = PatchTSTForPretraining(config_pretrain)

    reconstructed, mask, original = pretrain_model(x, mask_ratio=0.4)

    print(f"\nReconstructed patches: {reconstructed.shape}")
    print(f"Mask: {mask.shape}")
    print(f"Original patches: {original.shape}")
    print(f"Masked ratio: {(~mask).float().mean():.2%}")

    print("\n" + "="*80)
