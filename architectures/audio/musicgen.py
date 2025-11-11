"""
MusicGen - Text-to-Music Generation

Meta's state-of-the-art music generation model (2023-2025).
Generates high-quality music from text descriptions.

Key features:
- Text-conditional music generation
- 32 kHz stereo output
- Up to 30 seconds of coherent music
- Multiple conditioning methods (text, melody, audio)
- Built on Encodec + Transformer architecture

Architecture:
- Text encoder: T5 or similar
- Audio codec: Encodec (compressed tokens)
- Decoder: Autoregressive transformer with delay pattern
- Conditioning: Cross-attention to text embeddings

References:
- "Simple and Controllable Music Generation" (Copet et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from .encodec import Encodec, EncodecConfig


@dataclass
class MusicGenConfig:
    """Configuration for MusicGen model"""
    # Audio
    sample_rate: int = 32000  # 32kHz for music
    channels: int = 2  # Stereo

    # Model architecture
    d_model: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    d_ff: int = 4096

    # Conditioning
    text_encoder_dim: int = 1024  # T5 embedding dimension
    use_melody_conditioning: bool = True

    # Encodec
    encodec_config: Optional[EncodecConfig] = None
    num_codebooks: int = 4  # Number of Encodec codebooks to use
    codebook_size: int = 2048

    # Generation
    max_duration: float = 30.0  # Maximum generation duration in seconds
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0  # 0 = disabled

    # Training
    dropout: float = 0.1
    use_cfg: bool = True  # Classifier-free guidance
    cfg_coef: float = 3.0  # Guidance coefficient


class DelayedPatternProvider:
    """
    Delay pattern for parallel prediction of multiple codebooks.

    MusicGen predicts all codebook tokens in parallel using a clever
    delay pattern that maintains causality while being efficient.
    """

    def __init__(self, num_codebooks: int):
        self.num_codebooks = num_codebooks
        # Codebook i is delayed by i steps
        self.delays = list(range(num_codebooks))

    def build_pattern_sequence(
        self,
        codes: torch.Tensor
    ) -> torch.Tensor:
        """
        Build delayed pattern sequence from codebook tokens.

        Args:
            codes: (batch, num_codebooks, time)

        Returns:
            Pattern sequence (batch, num_codebooks * time)
        """
        batch_size, num_codebooks, time = codes.shape
        assert num_codebooks == self.num_codebooks

        # Create sequence with delays
        sequence = []

        for t in range(time):
            for k in range(num_codebooks):
                if t >= self.delays[k]:
                    sequence.append(codes[:, k, t - self.delays[k]])
                else:
                    # Padding token
                    sequence.append(torch.zeros_like(codes[:, 0, 0]))

        return torch.stack(sequence, dim=1)

    def revert_pattern_sequence(
        self,
        sequence: torch.Tensor,
        time: int
    ) -> torch.Tensor:
        """
        Revert pattern sequence back to codebook format.

        Args:
            sequence: (batch, num_codebooks * time)
            time: Original time dimension

        Returns:
            codes: (batch, num_codebooks, time)
        """
        batch_size = sequence.shape[0]
        codes = torch.zeros(
            batch_size, self.num_codebooks, time,
            dtype=sequence.dtype, device=sequence.device
        )

        idx = 0
        for t in range(time):
            for k in range(self.num_codebooks):
                if t >= self.delays[k]:
                    codes[:, k, t - self.delays[k]] = sequence[:, idx]
                idx += 1

        return codes


class MusicGenTransformer(nn.Module):
    """
    Transformer decoder for MusicGen.

    Predicts audio tokens autoregressively conditioned on text.
    """

    def __init__(self, config: MusicGenConfig):
        super().__init__()
        self.config = config

        # Token embeddings (for each codebook)
        self.embeddings = nn.ModuleList([
            nn.Embedding(config.codebook_size, config.d_model)
            for _ in range(config.num_codebooks)
        ])

        # Position embeddings
        max_positions = int(config.max_duration * config.sample_rate / 320)  # Hop length
        self.pos_embedding = nn.Embedding(max_positions, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])

        # Output projection (for each codebook)
        self.output_proj = nn.ModuleList([
            nn.Linear(config.d_model, config.codebook_size)
            for _ in range(config.num_codebooks)
        ])

        # Layer norm
        self.ln = nn.LayerNorm(config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        melody_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of transformer.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            text_embeddings: Text condition (batch, text_len, d_model)
            melody_embeddings: Melody condition (batch, melody_len, d_model)
            attention_mask: Attention mask

        Returns:
            Logits for each codebook (batch, seq_len, num_codebooks, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # TODO: Implement pattern-based multi-codebook embedding
        # For now, simple embedding
        x = self.embeddings[0](input_ids)

        # Add position embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        x = x + self.pos_embedding(positions)

        x = self.dropout(x)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_hidden_states=text_embeddings,
                attention_mask=causal_mask
            )

        x = self.ln(x)

        # Project to vocabulary for each codebook
        logits_list = []
        for proj in self.output_proj:
            logits_list.append(proj(x))

        # Stack: (batch, seq_len, num_codebooks, vocab_size)
        logits = torch.stack(logits_list, dim=2)

        return logits


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with cross-attention"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(d_model)

        # Cross-attention (to text)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward with self-attention, cross-attention, and FFN"""
        # Self-attention
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=attention_mask, need_weights=False)
        x = residual + x

        # Cross-attention (if conditioning provided)
        if encoder_hidden_states is not None:
            residual = x
            x = self.cross_attn_norm(x)
            x, _ = self.cross_attn(x, encoder_hidden_states, encoder_hidden_states, need_weights=False)
            x = residual + x

        # FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x


class MusicGen(nn.Module):
    """
    Complete MusicGen model for text-to-music generation.

    Can generate music conditioned on:
    - Text descriptions
    - Melody (as audio or MIDI)
    - Both text and melody
    """

    def __init__(self, config: MusicGenConfig):
        super().__init__()
        self.config = config

        # Audio codec (Encodec)
        if config.encodec_config is None:
            config.encodec_config = EncodecConfig(
                sample_rate=config.sample_rate,
                channels=config.channels,
                num_quantizers=config.num_codebooks
            )
        self.codec = Encodec(config.encodec_config)

        # Text encoder (placeholder - would use T5)
        self.text_encoder = None  # Would load T5-base or similar

        # Transformer decoder
        self.transformer = MusicGenTransformer(config)

        # Delay pattern for multi-codebook prediction
        self.pattern_provider = DelayedPatternProvider(config.num_codebooks)

    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode text descriptions to embeddings.

        Args:
            text: List of text descriptions

        Returns:
            Text embeddings (batch, text_len, d_model)
        """
        # Would use T5 or similar text encoder
        # For now, return dummy embeddings
        batch_size = len(text)
        text_len = 77  # Max length
        return torch.randn(
            batch_size, text_len, self.config.text_encoder_dim,
            device=next(self.parameters()).device
        )

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to discrete codes.

        Args:
            audio: Audio waveform (batch, channels, time)

        Returns:
            Discrete codes (batch, num_codebooks, encoded_time)
        """
        codes = self.codec.encode(audio)
        return codes

    def decode_audio(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete codes to audio.

        Args:
            codes: Discrete codes (batch, num_codebooks, encoded_time)

        Returns:
            Audio waveform (batch, channels, time)
        """
        audio = self.codec.decode(codes)
        return audio

    @torch.no_grad()
    def generate(
        self,
        text: List[str],
        duration: float = 10.0,
        melody: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: float = 3.0
    ) -> torch.Tensor:
        """
        Generate music from text (and optional melody).

        Args:
            text: Text descriptions
            duration: Duration in seconds
            melody: Optional melody audio (batch, channels, time)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            cfg_coef: Classifier-free guidance coefficient

        Returns:
            Generated audio (batch, channels, time)
        """
        batch_size = len(text)
        device = next(self.parameters()).device

        # Encode text condition
        text_embeddings = self.encode_text(text)

        # Calculate number of tokens to generate
        num_tokens = int(duration * self.config.sample_rate / self.codec.hop_length)

        # Initialize with start token
        generated_ids = torch.zeros(
            batch_size, 1,
            dtype=torch.long, device=device
        )

        # Autoregressive generation
        for _ in range(num_tokens):
            # Forward pass
            logits = self.transformer(
                generated_ids,
                text_embeddings=text_embeddings
            )

            # Get logits for next token (first codebook for simplicity)
            next_logits = logits[:, -1, 0, :]  # (batch, vocab_size)

            # Apply temperature
            next_logits = next_logits / temperature

            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) sampling
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # Convert tokens to codes (simplified - would use pattern provider)
        # For now, replicate across codebooks
        codes = generated_ids[:, 1:].unsqueeze(1).repeat(1, self.config.num_codebooks, 1)

        # Decode to audio
        audio = self.decode_audio(codes)

        return audio

    def forward(
        self,
        audio: torch.Tensor,
        text: List[str],
        melody: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            audio: Target audio (batch, channels, time)
            text: Text descriptions
            melody: Optional melody conditioning

        Returns:
            loss: Training loss
            logits: Model logits
        """
        # Encode audio to tokens
        with torch.no_grad():
            codes = self.encode_audio(audio)

        # Encode text
        text_embeddings = self.encode_text(text)

        # Convert codes to pattern sequence
        input_ids = self.pattern_provider.build_pattern_sequence(codes)

        # Shift for teacher forcing
        input_ids = input_ids[:, :-1]
        target_ids = self.pattern_provider.build_pattern_sequence(codes)[:, 1:]

        # Forward through transformer
        logits = self.transformer(
            input_ids,
            text_embeddings=text_embeddings
        )

        # Compute loss (cross-entropy for each codebook)
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.codebook_size),
            target_ids.reshape(-1),
            ignore_index=-100
        )

        return loss, logits


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("MusicGen - Text-to-Music Generation")
    print("="*80)

    # Create MusicGen model
    config = MusicGenConfig(
        sample_rate=32000,
        channels=2,
        d_model=1024,
        num_layers=24,
        num_codebooks=4
    )

    model = MusicGen(config)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Example: Generate music from text
    text_prompts = [
        "upbeat electronic dance music with heavy bass",
        "calm piano melody with strings"
    ]

    print(f"\nGenerating music for prompts:")
    for prompt in text_prompts:
        print(f"  - {prompt}")

    # Generate (this would actually generate music)
    # generated_audio = model.generate(
    #     text=text_prompts,
    #     duration=10.0,
    #     temperature=1.0,
    #     top_k=250
    # )
    # print(f"\nGenerated audio shape: {generated_audio.shape}")

    print("\n" + "="*80)
