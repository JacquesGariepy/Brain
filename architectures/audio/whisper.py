"""
Whisper - Robust Speech Recognition

OpenAI's SOTA speech recognition system (2022-2025).
Trained on 680,000 hours of multilingual data.

Key features:
- Multilingual (99 languages)
- Robust to accents, background noise
- Multitask: transcription, translation, language detection
- Zero-shot transfer
- Timestamp prediction

Architecture: Encoder-decoder transformer
- Encoder: Processes audio features (log-Mel spectrogram)
- Decoder: Autoregressive text generation with special tokens

References:
- "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import math


@dataclass
class WhisperConfig:
    """Configuration for Whisper model"""
    # Audio
    n_mels: int = 80  # Number of Mel frequency bins
    n_audio_ctx: int = 1500  # Audio context length (30s at 50 fps)
    n_audio_state: int = 384  # Encoder hidden size (base model)
    n_audio_head: int = 6  # Encoder attention heads
    n_audio_layer: int = 4  # Encoder layers

    # Text
    n_vocab: int = 51865  # Vocabulary size (multilingual)
    n_text_ctx: int = 448  # Text context length
    n_text_state: int = 384  # Decoder hidden size
    n_text_head: int = 6  # Decoder attention heads
    n_text_layer: int = 4  # Decoder layers

    # Model sizes (tiny, base, small, medium, large)
    # This is "base" configuration
    # Larger models have more layers and larger hidden dimensions


class WhisperEncoder(nn.Module):
    """
    Whisper audio encoder.

    Processes log-Mel spectrogram into contextualized representations.
    Uses conv layers for downsampling then transformer blocks.
    """

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config

        # Conv layers for initial downsampling
        # 2 conv layers reduce sequence length by 2x each
        self.conv1 = nn.Conv1d(
            config.n_mels, config.n_audio_state,
            kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            config.n_audio_state, config.n_audio_state,
            kernel_size=3, stride=2, padding=1
        )

        # Position embeddings (sinusoidal)
        self.register_buffer(
            "positional_embedding",
            self._get_sinusoidal_embeddings(
                config.n_audio_ctx // 2, config.n_audio_state
            )
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            WhisperEncoderBlock(config)
            for _ in range(config.n_audio_layer)
        ])

        # Final layer norm
        self.ln_post = nn.LayerNorm(config.n_audio_state)

    def _get_sinusoidal_embeddings(
        self,
        length: int,
        channels: int
    ) -> torch.Tensor:
        """Generate sinusoidal position embeddings"""
        assert channels % 2 == 0
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, channels, 2) * -(math.log(10000.0) / channels)
        )

        embeddings = torch.zeros(length, channels)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)

        return embeddings

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Encode audio features.

        Args:
            mel: Log-Mel spectrogram (batch, n_mels, time)

        Returns:
            Encoded audio (batch, time//2, n_audio_state)
        """
        # Conv layers with GELU activation
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))

        # Transpose for transformer: (batch, channels, time) -> (batch, time, channels)
        x = x.permute(0, 2, 1)

        # Add positional embeddings
        x = x + self.positional_embedding[:x.shape[1]]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_post(x)

        return x


class WhisperEncoderBlock(nn.Module):
    """Single transformer block for Whisper encoder"""

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.n_audio_state,
            config.n_audio_head,
            batch_first=True
        )
        self.attn_ln = nn.LayerNorm(config.n_audio_state)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_audio_state, config.n_audio_state * 4),
            nn.GELU(),
            nn.Linear(config.n_audio_state * 4, config.n_audio_state)
        )
        self.mlp_ln = nn.LayerNorm(config.n_audio_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with pre-norm residual connections"""
        x = x + self.attn(self.attn_ln(x), self.attn_ln(x), self.attn_ln(x))[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class WhisperDecoder(nn.Module):
    """
    Whisper text decoder.

    Autoregressive decoder with cross-attention to encoder outputs.
    Generates text tokens conditioned on audio.
    """

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.n_vocab, config.n_text_state)

        # Position embeddings (learned)
        self.positional_embedding = nn.Parameter(
            torch.randn(config.n_text_ctx, config.n_text_state)
        )

        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList([
            WhisperDecoderBlock(config)
            for _ in range(config.n_text_layer)
        ])

        # Final layer norm and projection
        self.ln = nn.LayerNorm(config.n_text_state)

    def forward(
        self,
        tokens: torch.Tensor,
        audio_features: torch.Tensor,
        past_key_values: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Decode tokens conditioned on audio.

        Args:
            tokens: Input tokens (batch, seq_len)
            audio_features: Encoded audio (batch, audio_len, n_audio_state)
            past_key_values: Cached key-values for fast generation

        Returns:
            Hidden states (batch, seq_len, n_text_state)
            Updated key-value cache
        """
        # Token embeddings
        x = self.token_embedding(tokens)

        # Add positional embeddings
        x = x + self.positional_embedding[:tokens.shape[1]]

        # Transformer blocks with cross-attention
        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            kv = past_key_values[i] if past_key_values else None
            x, new_kv = block(x, audio_features, kv)
            new_kv_cache.append(new_kv)

        # Final layer norm
        x = self.ln(x)

        return x, new_kv_cache


class WhisperDecoderBlock(nn.Module):
    """Single transformer block for Whisper decoder with cross-attention"""

    def __init__(self, config: WhisperConfig):
        super().__init__()

        # Self-attention (causal)
        self.attn = nn.MultiheadAttention(
            config.n_text_state,
            config.n_text_head,
            batch_first=True
        )
        self.attn_ln = nn.LayerNorm(config.n_text_state)

        # Cross-attention to audio
        self.cross_attn = nn.MultiheadAttention(
            config.n_text_state,
            config.n_text_head,
            kdim=config.n_audio_state,
            vdim=config.n_audio_state,
            batch_first=True
        )
        self.cross_attn_ln = nn.LayerNorm(config.n_text_state)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.n_text_state, config.n_text_state * 4),
            nn.GELU(),
            nn.Linear(config.n_text_state * 4, config.n_text_state)
        )
        self.mlp_ln = nn.LayerNorm(config.n_text_state)

    def forward(
        self,
        x: torch.Tensor,
        audio_features: torch.Tensor,
        kv_cache: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """Forward with causal self-attention and cross-attention"""
        # Causal self-attention
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )

        attn_out, _ = self.attn(
            self.attn_ln(x), self.attn_ln(x), self.attn_ln(x),
            attn_mask=causal_mask,
            need_weights=False
        )
        x = x + attn_out

        # Cross-attention to audio
        cross_out, _ = self.cross_attn(
            self.cross_attn_ln(x),
            audio_features,
            audio_features,
            need_weights=False
        )
        x = x + cross_out

        # MLP
        x = x + self.mlp(self.mlp_ln(x))

        # Cache key-values (simplified)
        kv = (x, x)  # In practice, cache actual key-values

        return x, kv


class Whisper(nn.Module):
    """
    Complete Whisper model for speech recognition.

    Multitask capabilities:
    - Transcription (speech-to-text)
    - Translation (speech-to-English)
    - Language identification
    - Voice activity detection
    - Timestamp prediction
    """

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config

        # Encoder and decoder
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)

        # Output projection to vocabulary
        self.proj_out = nn.Linear(config.n_text_state, config.n_vocab, bias=False)

        # Special tokens (example IDs, actual depend on tokenizer)
        self.sot_token = 50258  # Start of transcript
        self.eot_token = 50257  # End of transcript
        self.translate_token = 50358  # Task: translate
        self.transcribe_token = 50359  # Task: transcribe
        self.no_speech_token = 50362  # No speech detected
        self.no_timestamps_token = 50363  # No timestamps

    def forward(
        self,
        mel: torch.Tensor,
        tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Whisper.

        Args:
            mel: Log-Mel spectrogram (batch, n_mels, time)
            tokens: Target tokens for training (batch, seq_len)

        Returns:
            Logits (batch, seq_len, n_vocab)
        """
        # Encode audio
        audio_features = self.encoder(mel)

        # Decode
        if tokens is None:
            # Generate start token
            batch_size = mel.shape[0]
            tokens = torch.full(
                (batch_size, 1),
                self.sot_token,
                dtype=torch.long,
                device=mel.device
            )

        hidden_states, _ = self.decoder(tokens, audio_features)

        # Project to vocabulary
        logits = self.proj_out(hidden_states)

        return logits

    def generate(
        self,
        mel: torch.Tensor,
        max_length: int = 448,
        temperature: float = 0.0,
        task: str = "transcribe",
        language: Optional[str] = None
    ) -> torch.Tensor:
        """
        Generate transcription or translation.

        Args:
            mel: Audio features (batch, n_mels, time)
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            task: "transcribe" or "translate"
            language: Optional language code

        Returns:
            Generated token IDs (batch, seq_len)
        """
        batch_size = mel.shape[0]

        # Encode audio
        audio_features = self.encoder(mel)

        # Initialize with special tokens
        # Format: <|startoftranscript|><|language|><|task|><|notimestamps|>
        tokens = [self.sot_token]
        if language:
            # Add language token (simplified)
            tokens.append(self.sot_token + 1)  # Placeholder
        if task == "translate":
            tokens.append(self.translate_token)
        else:
            tokens.append(self.transcribe_token)
        tokens.append(self.no_timestamps_token)

        tokens = torch.tensor([tokens] * batch_size, device=mel.device)

        # Greedy decoding
        for _ in range(max_length - tokens.shape[1]):
            # Forward
            hidden_states, _ = self.decoder(tokens, audio_features)
            logits = self.proj_out(hidden_states[:, -1:, :])

            # Sample next token
            if temperature == 0:
                next_token = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), 1)

            # Append token
            tokens = torch.cat([tokens, next_token], dim=1)

            # Stop if all sequences generated EOT
            if (next_token == self.eot_token).all():
                break

        return tokens

    def detect_language(self, mel: torch.Tensor) -> Dict[str, float]:
        """
        Detect language from audio.

        Returns dictionary of language probabilities.
        """
        # Encode audio
        audio_features = self.encoder(mel)

        # Decode with SOT token
        sot_tokens = torch.full(
            (mel.shape[0], 1),
            self.sot_token,
            device=mel.device
        )

        hidden_states, _ = self.decoder(sot_tokens, audio_features)
        logits = self.proj_out(hidden_states)

        # Language tokens are at specific positions
        # (simplified - actual implementation more complex)
        lang_logits = logits[:, 0, self.sot_token+1:self.sot_token+100]
        lang_probs = F.softmax(lang_logits, dim=-1)

        # Return as dict (placeholder)
        return {"detected_language": "en", "probability": 0.95}


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Whisper - Robust Speech Recognition")
    print("="*80)

    # Create Whisper model (base size)
    config = WhisperConfig()
    model = Whisper(config)

    # Example inputs
    batch_size = 2
    audio_length = 3000  # 30 seconds at 50 fps downsampled
    mel = torch.randn(batch_size, config.n_mels, audio_length)

    # Transcribe
    print("\nTranscribing audio...")
    outputs = model.generate(mel, max_length=100, task="transcribe")

    print(f"\nModel Statistics:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")
    print(f"Output shape: {outputs.shape}")

    # Language detection
    lang_info = model.detect_language(mel)
    print(f"\nDetected language: {lang_info}")

    print("\n" + "="*80)
