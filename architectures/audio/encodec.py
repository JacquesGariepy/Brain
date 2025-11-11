"""
Encodec - High-Fidelity Neural Audio Codec

Meta's state-of-the-art neural audio codec (2022-2025).
Used as foundation for MusicGen, AudioGen, and other audio generation models.

Key features:
- High-quality audio compression (24 kHz, 48 kHz)
- Real-time encoding/decoding
- Residual Vector Quantization (RVQ)
- Multi-scale STFT discriminator
- Bandwidth scalability (1.5-12 kbps)

Architecture:
- Encoder: 1D conv with residual blocks
- Quantizer: Residual Vector Quantization (RVQ)
- Decoder: Transpose conv with residual blocks

References:
- "High Fidelity Neural Audio Compression" (Défossez et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from dataclasses import dataclass
import math


@dataclass
class EncodecConfig:
    """Configuration for Encodec model"""
    # Audio
    sample_rate: int = 24000  # 24kHz (music) or 48kHz (general audio)
    channels: int = 1  # Mono

    # Model architecture
    encoder_dim: int = 128
    encoder_rates: List[int] = None  # Downsampling rates [8, 5, 4, 2]
    decoder_dim: int = 128
    decoder_rates: List[int] = None  # Upsampling rates [2, 4, 5, 8]

    # Residual blocks
    encoder_residual_layers: int = 2
    decoder_residual_layers: int = 2
    residual_kernel_size: int = 7
    dilation_growth_rate: int = 2

    # Quantization
    codebook_size: int = 1024  # Vocabulary size per codebook
    num_quantizers: int = 8  # Number of RVQ stages
    codebook_dim: int = 128

    # Compression
    bandwidth: float = 6.0  # Target bandwidth in kbps
    # num_quantizers used = bandwidth * 1000 / (sample_rate / hop_length)

    # Training
    use_ema: bool = True  # Exponential moving average for codebooks
    kmeans_init: bool = True  # Initialize codebooks with k-means

    def __post_init__(self):
        if self.encoder_rates is None:
            self.encoder_rates = [8, 5, 4, 2]
        if self.decoder_rates is None:
            self.decoder_rates = list(reversed(self.encoder_rates))


class ResidualUnit(nn.Module):
    """
    Residual unit with dilated convolutions.

    Used in both encoder and decoder.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        dilation: int = 1
    ):
        super().__init__()

        padding = (kernel_size * dilation - dilation) // 2

        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, dilation=dilation, padding=padding),
            nn.ELU(),
            nn.Conv1d(dim, dim, 1)  # 1x1 conv
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection"""
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """
    Encoder block with downsampling.

    Sequence: Conv (downsample) -> Residual units
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        num_residual_layers: int = 2,
        kernel_size: int = 7,
        dilation_growth_rate: int = 2
    ):
        super().__init__()

        # Downsampling convolution
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2
        )

        # Residual units with increasing dilation
        self.residual_units = nn.ModuleList([
            ResidualUnit(
                out_channels,
                kernel_size,
                dilation=dilation_growth_rate ** i
            )
            for i in range(num_residual_layers)
        ])

        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through encoder block"""
        x = self.conv(x)
        x = self.elu(x)

        for unit in self.residual_units:
            x = unit(x)

        return x


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling.

    Sequence: Residual units -> TransposeConv (upsample)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        num_residual_layers: int = 2,
        kernel_size: int = 7,
        dilation_growth_rate: int = 2
    ):
        super().__init__()

        # Residual units
        self.residual_units = nn.ModuleList([
            ResidualUnit(
                in_channels,
                kernel_size,
                dilation=dilation_growth_rate ** i
            )
            for i in range(num_residual_layers)
        ])

        # Upsampling transpose convolution
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2
        )

        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through decoder block"""
        for unit in self.residual_units:
            x = unit(x)

        x = self.conv_transpose(x)
        x = self.elu(x)

        return x


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantization (RVQ).

    Quantizes in multiple stages, each handling the residual from previous stages.
    This allows higher quality reconstruction at higher bandwidths.
    """

    def __init__(self, config: EncodecConfig):
        super().__init__()
        self.config = config

        # Create multiple quantizer layers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                dim=config.codebook_dim,
                codebook_size=config.codebook_size,
                use_ema=config.use_ema,
                kmeans_init=config.kmeans_init
            )
            for _ in range(config.num_quantizers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        num_quantizers: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input using RVQ.

        Args:
            x: Input tensor (batch, dim, time)
            num_quantizers: Number of quantizers to use (for bandwidth control)

        Returns:
            quantized: Quantized output
            codes: Quantization indices (batch, num_quantizers, time)
            commit_loss: Commitment loss for training
        """
        if num_quantizers is None:
            num_quantizers = len(self.quantizers)

        quantized = torch.zeros_like(x)
        residual = x
        codes_list = []
        commit_loss = 0.0

        # Quantize residuals progressively
        for i in range(num_quantizers):
            q, indices, loss = self.quantizers[i](residual)
            quantized = quantized + q
            residual = residual - q
            codes_list.append(indices)
            commit_loss += loss

        # Stack codes
        codes = torch.stack(codes_list, dim=1)  # (batch, num_quantizers, time)

        return quantized, codes, commit_loss

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode from quantization codes.

        Args:
            codes: Quantization indices (batch, num_quantizers, time)

        Returns:
            Reconstructed tensor (batch, dim, time)
        """
        quantized = None

        for i, quantizer in enumerate(self.quantizers):
            if i >= codes.shape[1]:
                break

            q = quantizer.decode(codes[:, i])

            if quantized is None:
                quantized = q
            else:
                quantized = quantized + q

        return quantized


class VectorQuantizer(nn.Module):
    """
    Single vector quantizer layer.

    Uses either EMA updates or gradient-based training.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        use_ema: bool = True,
        kmeans_init: bool = True,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.use_ema = use_ema
        self.decay = decay
        self.epsilon = epsilon

        # Codebook embeddings
        self.codebook = nn.Embedding(codebook_size, dim)

        if not kmeans_init:
            self.codebook.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

        # EMA variables (if using EMA)
        if use_ema:
            self.register_buffer('cluster_size', torch.zeros(codebook_size))
            self.register_buffer('embed_avg', self.codebook.weight.data.clone())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input tensor.

        Args:
            x: Input (batch, dim, time)

        Returns:
            quantized: Quantized output (batch, dim, time)
            indices: Codebook indices (batch, time)
            commit_loss: Commitment loss
        """
        # Reshape for quantization
        batch_size, dim, time = x.shape
        x_flat = x.permute(0, 2, 1).reshape(-1, dim)  # (batch*time, dim)

        # Compute distances to codebook vectors
        distances = (
            x_flat.pow(2).sum(1, keepdim=True) -
            2 * x_flat @ self.codebook.weight.t() +
            self.codebook.weight.pow(2).sum(1, keepdim=True).t()
        )

        # Find nearest codebook vectors
        indices = distances.argmin(dim=1)  # (batch*time,)
        quantized_flat = self.codebook(indices)

        # Reshape back
        quantized = quantized_flat.view(batch_size, time, dim).permute(0, 2, 1)
        indices = indices.view(batch_size, time)

        # Commitment loss (for gradient)
        commit_loss = F.mse_loss(quantized.detach(), x)

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        # EMA update (in training mode)
        if self.training and self.use_ema:
            self._ema_update(x_flat, indices)

        return quantized, indices, commit_loss

    def _ema_update(self, x_flat: torch.Tensor, indices: torch.Tensor):
        """Update codebook using exponential moving average"""
        # Compute cluster sizes
        indices_onehot = F.one_hot(indices, self.codebook_size).float()
        cluster_size = indices_onehot.sum(0)

        # EMA cluster size
        self.cluster_size.data.mul_(self.decay).add_(
            cluster_size, alpha=1 - self.decay
        )

        # EMA embedding average
        embed_sum = x_flat.t() @ indices_onehot
        self.embed_avg.data.mul_(self.decay).add_(
            embed_sum.t(), alpha=1 - self.decay
        )

        # Update codebook
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.epsilon) /
            (n + self.codebook_size * self.epsilon) * n
        )
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.codebook.weight.data.copy_(embed_normalized)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from indices.

        Args:
            indices: Codebook indices (batch, time)

        Returns:
            Decoded tensor (batch, dim, time)
        """
        quantized = self.codebook(indices)  # (batch, time, dim)
        return quantized.permute(0, 2, 1)  # (batch, dim, time)


class Encodec(nn.Module):
    """
    Complete Encodec model for neural audio compression.

    Can be used for:
    - High-quality audio compression
    - Audio generation (as discrete tokens)
    - Music generation (MusicGen)
    - General audio generation (AudioGen)
    """

    def __init__(self, config: EncodecConfig):
        super().__init__()
        self.config = config

        # Calculate total downsampling factor
        self.hop_length = 1
        for rate in config.encoder_rates:
            self.hop_length *= rate

        # Encoder
        self.encoder = self._build_encoder()

        # Quantizer
        self.quantizer = ResidualVectorQuantizer(config)

        # Decoder
        self.decoder = self._build_decoder()

    def _build_encoder(self) -> nn.Module:
        """Build encoder network"""
        encoder = nn.ModuleList()

        # Initial convolution
        encoder.append(
            nn.Conv1d(
                self.config.channels,
                self.config.encoder_dim,
                kernel_size=7,
                padding=3
            )
        )

        # Encoder blocks with downsampling
        in_channels = self.config.encoder_dim
        for stride in self.config.encoder_rates:
            out_channels = in_channels * 2
            encoder.append(
                EncoderBlock(
                    in_channels,
                    out_channels,
                    stride,
                    self.config.encoder_residual_layers,
                    self.config.residual_kernel_size,
                    self.config.dilation_growth_rate
                )
            )
            in_channels = out_channels

        # Final convolution to codebook dim
        encoder.append(
            nn.Conv1d(
                in_channels,
                self.config.codebook_dim,
                kernel_size=7,
                padding=3
            )
        )

        return nn.Sequential(*encoder)

    def _build_decoder(self) -> nn.Module:
        """Build decoder network"""
        decoder = nn.ModuleList()

        # Initial convolution from codebook dim
        in_channels = self.config.encoder_dim * (2 ** len(self.config.encoder_rates))
        decoder.append(
            nn.Conv1d(
                self.config.codebook_dim,
                in_channels,
                kernel_size=7,
                padding=3
            )
        )

        # Decoder blocks with upsampling
        for stride in self.config.decoder_rates:
            out_channels = in_channels // 2
            decoder.append(
                DecoderBlock(
                    in_channels,
                    out_channels,
                    stride,
                    self.config.decoder_residual_layers,
                    self.config.residual_kernel_size,
                    self.config.dilation_growth_rate
                )
            )
            in_channels = out_channels

        # Final convolution to audio channels
        decoder.append(
            nn.Conv1d(
                in_channels,
                self.config.channels,
                kernel_size=7,
                padding=3
            )
        )

        return nn.Sequential(*decoder)

    def forward(
        self,
        audio: torch.Tensor,
        bandwidth: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode and decode audio.

        Args:
            audio: Input audio (batch, channels, time)
            bandwidth: Target bandwidth in kbps (optional)

        Returns:
            reconstructed: Reconstructed audio (batch, channels, time)
            codes: Quantization codes (batch, num_quantizers, time)
            commit_loss: Commitment loss
        """
        # Encode
        encoded = self.encoder(audio)

        # Determine number of quantizers based on bandwidth
        if bandwidth is not None:
            num_quantizers = self._bandwidth_to_num_quantizers(bandwidth)
        else:
            num_quantizers = None

        # Quantize
        quantized, codes, commit_loss = self.quantizer(encoded, num_quantizers)

        # Decode
        reconstructed = self.decoder(quantized)

        return reconstructed, codes, commit_loss

    def encode(self, audio: torch.Tensor, bandwidth: Optional[float] = None) -> torch.Tensor:
        """
        Encode audio to discrete codes.

        Args:
            audio: Input audio (batch, channels, time)
            bandwidth: Target bandwidth in kbps

        Returns:
            codes: Discrete codes (batch, num_quantizers, encoded_time)
        """
        encoded = self.encoder(audio)

        num_quantizers = (
            self._bandwidth_to_num_quantizers(bandwidth)
            if bandwidth else None
        )

        _, codes, _ = self.quantizer(encoded, num_quantizers)

        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode from discrete codes.

        Args:
            codes: Discrete codes (batch, num_quantizers, encoded_time)

        Returns:
            audio: Reconstructed audio (batch, channels, time)
        """
        quantized = self.quantizer.decode(codes)
        audio = self.decoder(quantized)
        return audio

    def _bandwidth_to_num_quantizers(self, bandwidth: float) -> int:
        """Convert target bandwidth to number of quantizers"""
        # bandwidth (kbps) = num_quantizers * sample_rate / hop_length / 1000
        num_quantizers = int(
            bandwidth * 1000 * self.hop_length / self.config.sample_rate
        )
        return min(num_quantizers, self.config.num_quantizers)


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Encodec - High-Fidelity Neural Audio Codec")
    print("="*80)

    # Create Encodec model (24kHz music)
    config = EncodecConfig(
        sample_rate=24000,
        channels=1,
        encoder_rates=[8, 5, 4, 2],
        num_quantizers=8,
        bandwidth=6.0
    )

    model = Encodec(config)

    # Example audio (3 seconds)
    batch_size = 2
    audio_length = 3 * config.sample_rate
    audio = torch.randn(batch_size, config.channels, audio_length)

    # Encode and decode
    print(f"\nInput audio shape: {audio.shape}")
    reconstructed, codes, commit_loss = model(audio, bandwidth=6.0)

    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Codes shape: {codes.shape}")
    print(f"Commit loss: {commit_loss.item():.4f}")

    # Just encode (for tokenization)
    codes_only = model.encode(audio, bandwidth=6.0)
    print(f"\nEncoded codes shape: {codes_only.shape}")

    # Calculate compression ratio
    original_bits = audio_length * 16  # 16-bit audio
    compressed_bits = codes.shape[1] * codes.shape[2] * math.log2(config.codebook_size)
    compression_ratio = original_bits / compressed_bits

    print(f"\nCompression:")
    print(f"Original: {original_bits / 1000:.1f} kbits")
    print(f"Compressed: {compressed_bits / 1000:.1f} kbits")
    print(f"Compression ratio: {compression_ratio:.1f}x")

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "="*80)
